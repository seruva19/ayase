"""Near-duplicate detection across the dataset using perceptual hashing (pHash).

Each sample's representative frame (the image itself, or a video's middle frame)
is hashed during ``process`` (samples restored from a resumed run are hashed in
``post_process``); grouping happens in ``post_process`` once the whole dataset
has been seen. Two samples are linked when the Hamming distance between
their hashes is at most ``hamming_threshold`` bits (0 = exact hash match only),
and links are merged transitively into duplicate groups. Candidate pairs are
found with pigeonhole block indexing, so the dataset is never compared all
pairs against all pairs.

In each group one representative is kept: the sample with the highest
``priority_metric`` value (default ``technical_score`` from ``basic``), ties
and missing values falling back to processing order. Every other member gets a
``near_duplicate`` WARNING naming the representative and its Hamming distance to it.
Because grouping is transitive, a member can be farther than the threshold from
the representative when it is linked through a chain of closer samples.

pHash matches global low-frequency structure: re-encodes, resizes, mild crops
and colour edits are caught, while semantic near-duplicates (same scene, other
viewpoint) are not; use ``diversity_selection`` for those. Flat or near-black
frames hash alike and can be grouped even when their content differs. No
QualityMetrics field is written.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from ayase.image import load_representative_frame
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _popcount(value: int) -> int:
    return bin(value).count("1")


class _DisjointSet:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def find_duplicate_groups(hashes: List[int], n_bits: int, threshold: int) -> List[List[int]]:
    """Group hash indices whose Hamming distance is <= ``threshold`` (transitively).

    Returns groups of size >= 2, each sorted by index. Splitting the hash into
    ``threshold + 1`` bit blocks guarantees (pigeonhole) that any pair within the
    threshold agrees exactly on at least one block, so only bucket-mates are compared.
    """
    n = len(hashes)
    dsu = _DisjointSet(n)

    # Identical hashes are merged first, so blocking only works on unique values.
    first_by_hash: Dict[int, int] = {}
    for i, h in enumerate(hashes):
        if h in first_by_hash:
            dsu.union(first_by_hash[h], i)
        else:
            first_by_hash[h] = i

    threshold = max(0, min(int(threshold), n_bits))
    if threshold > 0 and len(first_by_hash) > 1:
        unique = list(first_by_hash.items())  # (hash, index)
        n_blocks = min(threshold + 1, n_bits)
        bounds = np.linspace(0, n_bits, n_blocks + 1, dtype=int)
        compared = set()
        for b in range(n_blocks):
            lo, hi = int(bounds[b]), int(bounds[b + 1])
            mask = ((1 << (hi - lo)) - 1) << lo
            buckets: Dict[int, List[Tuple[int, int]]] = {}
            for h, idx in unique:
                buckets.setdefault(h & mask, []).append((h, idx))
            for members in buckets.values():
                for a in range(len(members)):
                    ha, ia = members[a]
                    for c in range(a + 1, len(members)):
                        hc, ic = members[c]
                        pair = (ia, ic)
                        if pair in compared:
                            continue
                        compared.add(pair)
                        if _popcount(ha ^ hc) <= threshold:
                            dsu.union(ia, ic)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(dsu.find(i), []).append(i)
    return [sorted(g) for g in groups.values() if len(g) > 1]


class DeduplicationModule(PipelineModule):
    name = "deduplication"
    description = "Groups near-duplicate samples by pHash Hamming distance and flags all but the best one"
    default_config = {
        "hamming_threshold": 4,  # max differing bits (of hash_size**2) to count as duplicate
        "hash_size": 8,  # pHash grid side; 8 -> 64-bit hash
        "priority_metric": "technical_score",  # QualityMetrics field; highest is kept
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.hamming_threshold = int(self.config.get("hamming_threshold", 4))
        self.hash_size = int(self.config.get("hash_size", 8))
        self.priority_metric = self.config.get("priority_metric", "technical_score")
        self._hashes: Dict[str, int] = {}  # sample path -> pHash as int
        self._imagehash_available = False

        try:
            import imagehash
            self.imagehash = imagehash
            self._imagehash_available = True
        except ImportError:
            logger.warning("imagehash not installed. Deduplication disabled.")

    def setup(self) -> None:
        self._hashes = {}

    def process(self, sample: Sample) -> Sample:
        if not self._imagehash_available:
            return sample

        h = self._hash_sample(sample)
        if h is not None:
            self._hashes[str(sample.path)] = h
        return sample

    def post_process(self, all_samples: List[Sample]) -> None:
        if not self._imagehash_available or len(all_samples) < 2:
            return

        entries: List[Tuple[Sample, int]] = []
        for sample in all_samples:
            key = str(sample.path)
            h = self._hashes.get(key)
            if h is None:
                h = self._hash_sample(sample)
                if h is None:
                    continue
                self._hashes[key] = h
            entries.append((sample, h))
        if len(entries) < 2:
            return

        # Resumed runs restore samples with the previous run's verdicts; regroup from scratch.
        for sample, _ in entries:
            sample.validation_issues = [
                issue for issue in sample.validation_issues
                if issue.issue_type != "near_duplicate"
            ]

        n_bits = self.hash_size * self.hash_size
        groups = find_duplicate_groups([h for _, h in entries], n_bits, self.hamming_threshold)

        flagged = 0
        for group_id, members in enumerate(groups):
            # Highest priority first; stable sort keeps processing order on ties.
            ranked = sorted(members, key=lambda i: -self._priority(entries[i][0]))
            rep_sample, rep_hash = entries[ranked[0]]
            for i in ranked[1:]:
                sample, h = entries[i]
                distance = _popcount(h ^ rep_hash)
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        issue_type="near_duplicate",
                        message=(
                            f"Near-duplicate of {rep_sample.path.name} "
                            f"(pHash distance {distance}/{n_bits})"
                        ),
                        details={
                            "original": str(rep_sample.path),
                            "hamming_distance": distance,
                            "hash_bits": n_bits,
                            "duplicate_group": group_id,
                            "group_size": len(members),
                            "phash": format(h, f"0{n_bits // 4}x"),
                        },
                        recommendation="Remove this sample and keep the group representative.",
                    )
                )
                flagged += 1

        logger.info(
            f"Deduplication: {len(groups)} duplicate groups, {flagged} samples flagged "
            f"(hamming_threshold={self.hamming_threshold})."
        )

    def _priority(self, sample: Sample) -> float:
        if not self.priority_metric or sample.quality_metrics is None:
            return float("-inf")
        value = getattr(sample.quality_metrics, self.priority_metric, None)
        if not isinstance(value, (int, float)) or value != value:  # None / non-numeric / NaN
            return float("-inf")
        return float(value)

    def _hash_sample(self, sample: Sample) -> Optional[int]:
        image = self._load_image(sample)
        if image is None:
            return None
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return int(str(self.imagehash.phash(pil_image, hash_size=self.hash_size)), 16)
        except Exception as e:
            logger.warning(f"Dedup failed for {sample.path}: {e}")
            return None

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            # Middle frame for video, the image itself otherwise (shared cache).
            return load_representative_frame(sample.path, color="bgr")
        except Exception:
            return None


class DedupCompatModule(DeduplicationModule):
    """Compatibility alias matching filename-based discovery."""

    name = "dedup"

"""ST-GREED (Spatial-Temporal Generalized Entropic Difference) module.

Full-reference video quality metric based on natural scene statistics
in the spatial and temporal domains.

Backend tiers:
  1. **FR mode** — full-reference entropic difference when reference available
     (ported from ``github.com/pavancm/GREED``)
  2. **NR heuristic** — bandpass NSS statistics with heuristic quality mapping
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class STGREEDModule(PipelineModule):
    name = "st_greed"
    description = "Spatial-temporal entropic quality (FR entropic difference or NR heuristic fallback)"
    default_config = {"subsample": 16}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not sample.is_video:
            return sample

        reference_path = getattr(sample, "reference_path", None)
        has_reference = reference_path is not None and Path(str(reference_path)).exists()

        try:
            if has_reference:
                score = self._compute_fr(sample, Path(str(reference_path)))
            else:
                score = self._compute_nr(sample)

            if score is not None:
                sample.quality_metrics.st_greed_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("ST-GREED failed: %s", e)
        return sample

    def _compute_fr(self, sample: Sample, reference_path: Path) -> Optional[float]:
        """Full-reference ST-GREED: entropic difference between ref and dist."""
        import cv2

        subsample = self.config.get("subsample", 16)

        # Load distorted frames
        dist_frames = self._load_gray_frames(sample.path, subsample)
        ref_frames = self._load_gray_frames(reference_path, subsample)

        if len(dist_frames) < 2 or len(ref_frames) < 2:
            return self._compute_nr(sample)

        n_pairs = min(len(dist_frames), len(ref_frames))
        dist_frames = dist_frames[:n_pairs]
        ref_frames = ref_frames[:n_pairs]

        # Resize reference to match distorted
        for i in range(n_pairs):
            h, w = dist_frames[i].shape
            ref_frames[i] = cv2.resize(ref_frames[i], (w, h))

        # Spatial GREED: entropic difference per frame
        spatial_diffs = []
        for i in range(n_pairs):
            ref_ent = self._spatial_greed(ref_frames[i])
            dist_ent = self._spatial_greed(dist_frames[i])
            # Entropic difference (lower = better quality for distorted)
            diff = abs(ref_ent - dist_ent)
            spatial_diffs.append(diff)

        # Temporal GREED: entropic difference of temporal subbands
        ref_temporal = self._temporal_entropy(ref_frames)
        dist_temporal = self._temporal_entropy(dist_frames)
        temporal_diff = abs(ref_temporal - dist_temporal)

        # Combine: lower difference = better quality
        # Normalize to 0-1 (1 = perfect match)
        spatial_quality = max(0.0, 1.0 - np.mean(spatial_diffs) / 6.0)
        temporal_quality = max(0.0, 1.0 - temporal_diff / 4.0)

        # Geometric mean (as in original GREED paper)
        st_greed = np.sqrt(max(0.0, spatial_quality) * max(0.0, temporal_quality))
        return float(st_greed)

    def _compute_nr(self, sample: Sample) -> Optional[float]:
        """No-reference ST-GREED: bandpass NSS with heuristic mapping."""
        import cv2

        subsample = self.config.get("subsample", 16)
        cap = cv2.VideoCapture(str(sample.path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, total, max(1, total // subsample)))[:subsample]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64))
        cap.release()

        if len(frames) < 2:
            return None

        # Spatial GREED
        spatial_entropies = []
        for gray in frames:
            s_ent = self._spatial_greed(gray)
            spatial_entropies.append(s_ent)

        # Temporal GREED
        temporal_entropy = self._temporal_greed(frames, fps, total, subsample)

        spatial_score = float(np.mean(spatial_entropies))
        spatial_quality = min(1.0, spatial_score / 6.0)
        temporal_quality = min(1.0, temporal_entropy / 4.0)

        st_greed = np.sqrt(max(0.0, spatial_quality) * max(0.0, temporal_quality))
        return float(st_greed)

    def _spatial_greed(self, gray: np.ndarray) -> float:
        """Compute spatial entropic features using bandpass decomposition."""
        import cv2

        h, w = gray.shape
        entropies = []
        current = gray.copy()
        for scale in range(4):
            blurred = cv2.GaussianBlur(current, (0, 0), 2.0)
            bandpass = current - blurred

            mu = cv2.GaussianBlur(bandpass, (7, 7), 7 / 6)
            sigma = np.sqrt(cv2.GaussianBlur(bandpass ** 2, (7, 7), 7 / 6) - mu ** 2 + 1e-7)
            mscn = bandpass / (sigma + 1.0)

            hist, _ = np.histogram(mscn.flatten(), bins=64, range=(-3, 3))
            hist = hist / (hist.sum() + 1e-8)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            entropies.append(entropy)

            current = cv2.resize(current, (max(1, w // (2 ** (scale + 1))),
                                           max(1, h // (2 ** (scale + 1)))))

        return float(np.mean(entropies))

    def _temporal_entropy(self, frames) -> float:
        """Compute temporal entropy from frame differences."""
        if len(frames) < 2:
            return 2.0

        entropies = []
        for i in range(len(frames) - 1):
            diff = frames[i + 1] - frames[i]
            sigma = np.std(diff) + 1e-7
            mscn = (diff - np.mean(diff)) / sigma
            hist, _ = np.histogram(mscn.flatten(), bins=64, range=(-3, 3))
            hist = hist / (hist.sum() + 1e-8)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            entropies.append(entropy)

        return float(np.mean(entropies))

    def _temporal_greed(self, frames, fps, total_frames, subsample) -> float:
        """Compute temporal entropic features from frame differences."""
        frame_gap = max(1, total_frames // subsample)
        temporal_diffs = []

        for i in range(len(frames) - 1):
            diff = frames[i + 1] - frames[i]
            diff_normalized = diff / max(1.0, frame_gap)
            temporal_diffs.append(diff_normalized)

        if not temporal_diffs:
            return 2.0

        entropies = []
        for diff in temporal_diffs:
            sigma = np.std(diff) + 1e-7
            mscn = (diff - np.mean(diff)) / sigma
            hist, _ = np.histogram(mscn.flatten(), bins=64, range=(-3, 3))
            hist = hist / (hist.sum() + 1e-8)
            entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
            entropies.append(entropy)

        mean_entropy = float(np.mean(entropies))
        fr_factor = min(1.0, fps / 30.0)

        return mean_entropy * (0.7 + 0.3 * fr_factor)

    def _load_gray_frames(self, path, subsample: int) -> list:
        """Load grayscale frames from a video path."""
        import cv2

        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = list(range(0, total, max(1, total // subsample)))[:subsample]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64))
        cap.release()
        return frames

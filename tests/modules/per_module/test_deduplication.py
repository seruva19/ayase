"""Tests for deduplication module."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_deduplication_basics():
    from ayase.modules.dedup import DeduplicationModule
    _test_module_basics(DeduplicationModule, "deduplication")

def test_deduplication_image(image_sample):
    from ayase.modules.dedup import DeduplicationModule
    image_sample.quality_metrics = QualityMetrics()
    m = DeduplicationModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_deduplication_video(video_sample):
    from ayase.modules.dedup import DeduplicationModule
    video_sample.quality_metrics = QualityMetrics()
    m = DeduplicationModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_find_duplicate_groups_matches_brute_force():
    import random
    from ayase.modules.dedup import find_duplicate_groups

    rng = random.Random(0)
    base = [rng.getrandbits(64) for _ in range(12)]
    hashes = []
    for b in base:  # clusters of near variants: flip 0..6 random bits
        hashes.append(b)
        for _ in range(3):
            v = b
            for bit in rng.sample(range(64), rng.randint(0, 6)):
                v ^= 1 << bit
            hashes.append(v)

    for threshold in (0, 2, 4, 6):
        # Reference: all-pairs union-find.
        parent = list(range(len(hashes)))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        for i in range(len(hashes)):
            for j in range(i + 1, len(hashes)):
                if bin(hashes[i] ^ hashes[j]).count("1") <= threshold:
                    parent[find(j)] = find(i)
        expected = {}
        for i in range(len(hashes)):
            expected.setdefault(find(i), []).append(i)
        expected = sorted(sorted(g) for g in expected.values() if len(g) > 1)

        assert sorted(find_duplicate_groups(hashes, 64, threshold)) == expected


def test_find_duplicate_groups_is_transitive_and_exact_at_zero():
    from ayase.modules.dedup import find_duplicate_groups

    a, b, c = 0b0, 0b111, 0b111111  # a-b: 3 bits, b-c: 3 bits, a-c: 6 bits
    assert find_duplicate_groups([a, b, c], 64, 3) == [[0, 1, 2]]
    assert find_duplicate_groups([a, b, c], 64, 2) == []
    assert find_duplicate_groups([a, a, b], 64, 0) == [[0, 1]]


def _pattern_image(path, seed, size=128):
    import cv2
    import numpy as np

    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    img = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(path), img)
    return img


def test_deduplication_groups_near_duplicates_and_keeps_best(tmp_path):
    import cv2
    import pytest
    from ayase.models import Sample
    from ayase.modules.dedup import DeduplicationModule

    pytest.importorskip("imagehash")

    img = _pattern_image(tmp_path / "orig.png", seed=1)
    # Re-encoded, downscaled copy: not byte-identical, same structure.
    small = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(tmp_path / "copy.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    _pattern_image(tmp_path / "other.png", seed=2)

    samples = [
        Sample(path=tmp_path / name, is_video=False)
        for name in ("orig.png", "copy.jpg", "other.png")
    ]
    samples[0].quality_metrics = QualityMetrics(technical_score=40.0)
    samples[1].quality_metrics = QualityMetrics(technical_score=80.0)

    m = DeduplicationModule({"hamming_threshold": 6})
    m.on_mount()
    for s in samples:
        m.process(s)
    m.post_process(samples)

    dup_issues = lambda s: [i for i in s.validation_issues if i.issue_type == "near_duplicate"]
    # The higher-technical_score copy is kept; the original is flagged against it.
    assert len(dup_issues(samples[0])) == 1
    assert dup_issues(samples[0])[0].details["original"] == str(samples[1].path)
    assert dup_issues(samples[1]) == []
    assert dup_issues(samples[2]) == []

    # Regrouping (e.g. a resumed run) must not stack verdicts.
    m.post_process(samples)
    assert len(dup_issues(samples[0])) == 1


def test_deduplication_hashes_samples_not_seen_in_process(tmp_path):
    import shutil
    import pytest
    from ayase.models import Sample
    from ayase.modules.dedup import DeduplicationModule

    pytest.importorskip("imagehash")

    _pattern_image(tmp_path / "a.png", seed=3)
    shutil.copy(tmp_path / "a.png", tmp_path / "b.png")
    samples = [Sample(path=tmp_path / n, is_video=False) for n in ("a.png", "b.png")]

    m = DeduplicationModule()
    m.on_mount()
    m.process(samples[0])  # b.png arrives only via post_process (resumed-run cache)
    m.post_process(samples)

    flagged = [s for s in samples if any(i.issue_type == "near_duplicate" for i in s.validation_issues)]
    assert flagged == [samples[1]]  # ties keep processing order: a.png is the representative

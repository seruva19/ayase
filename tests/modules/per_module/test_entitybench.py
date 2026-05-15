"""Tests for entitybench module."""

import cv2
import numpy as np

from ..conftest import _test_module_basics
from ayase.models import Sample


def _write_image(tmp_dir, name: str, color: tuple) -> Sample:
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:] = color
    path = tmp_dir / name
    cv2.imwrite(str(path), img)
    return Sample(path=path, is_video=False)


def test_entitybench_basics():
    from ayase.modules.entitybench import EntityBenchModule
    _test_module_basics(EntityBenchModule, "entitybench")


def test_entitybench_group_key_uses_parent(tmp_dir):
    from ayase.modules.entitybench import _group_key
    sub = tmp_dir / "shot_a"
    sub.mkdir()
    s = _write_image(sub, "f0.png", (255, 0, 0))
    assert _group_key(s) == str(sub)


def test_entitybench_single_shot_group_returns_one(tmp_dir):
    from ayase.modules.entitybench import EntityBenchModule
    m = EntityBenchModule(config={"backend": "histogram"})
    m.on_mount()
    sub = tmp_dir / "shot_alone"
    sub.mkdir()
    s = _write_image(sub, "frame0.png", (200, 100, 50))
    m.extract_features(s)
    m.on_dispose()
    # Single-shot group → trivial consistency = 1.0
    assert s.quality_metrics is not None
    assert s.quality_metrics.entitybench_identity_consistency == 1.0
    assert s.quality_metrics.entitybench_appearance_consistency == 1.0


def test_entitybench_histogram_consistency_high_for_same_color(tmp_dir):
    from ayase.modules.entitybench import EntityBenchModule
    m = EntityBenchModule(config={"backend": "histogram"})
    m.on_mount()
    sub = tmp_dir / "shot_same"
    sub.mkdir()
    a = _write_image(sub, "a.png", (180, 90, 30))
    b = _write_image(sub, "b.png", (180, 90, 30))
    m.extract_features(a)
    m.extract_features(b)
    m.on_dispose()
    assert a.quality_metrics.entitybench_appearance_consistency >= 0.9


def test_entitybench_histogram_consistency_low_for_different_colors(tmp_dir):
    from ayase.modules.entitybench import EntityBenchModule
    m = EntityBenchModule(config={"backend": "histogram"})
    m.on_mount()
    sub = tmp_dir / "shot_diff"
    sub.mkdir()
    a = _write_image(sub, "a.png", (255, 0, 0))
    b = _write_image(sub, "b.png", (0, 0, 255))
    m.extract_features(a)
    m.extract_features(b)
    m.on_dispose()
    # Different colors → lower consistency than identical case
    assert a.quality_metrics.entitybench_appearance_consistency < 0.9

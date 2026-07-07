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
    """Single-shot group → trivial consistency = 1.0 with a real backend;
    without one (histogram proxy removed) the fields must stay unset."""
    from ayase.modules.entitybench import EntityBenchModule
    m = EntityBenchModule()
    m.on_mount()
    sub = tmp_dir / "shot_alone"
    sub.mkdir()
    s = _write_image(sub, "frame0.png", (200, 100, 50))
    m.extract_features(s)
    m.on_dispose()
    if m.active_backend in ("dinov2_face", "clip"):
        assert s.quality_metrics is not None
        assert s.quality_metrics.entitybench_identity_consistency == 1.0
        assert s.quality_metrics.entitybench_appearance_consistency == 1.0
    else:
        # Honest state: no real embedding backend → no fabricated consistency
        assert m.active_backend == "unavailable"
        assert (s.quality_metrics is None
                or s.quality_metrics.entitybench_identity_consistency is None)


def test_entitybench_consistency_high_for_same_color(tmp_dir):
    """Identical images score high with a real backend; unset otherwise
    (the color-histogram proxy tier was removed)."""
    from ayase.modules.entitybench import EntityBenchModule
    m = EntityBenchModule()
    m.on_mount()
    sub = tmp_dir / "shot_same"
    sub.mkdir()
    a = _write_image(sub, "a.png", (180, 90, 30))
    b = _write_image(sub, "b.png", (180, 90, 30))
    m.extract_features(a)
    m.extract_features(b)
    m.on_dispose()
    if m.active_backend in ("dinov2_face", "clip"):
        assert a.quality_metrics.entitybench_appearance_consistency >= 0.9
    else:
        assert m.active_backend == "unavailable"
        assert (a.quality_metrics is None
                or a.quality_metrics.entitybench_appearance_consistency is None)


def test_entitybench_unavailable_backend_writes_nothing(tmp_dir):
    """With the proxy tier removed, an unavailable backend must not emit
    any entitybench fields at all."""
    from ayase.modules.entitybench import EntityBenchModule
    m = EntityBenchModule()
    m.active_backend = "unavailable"  # force honest-unavailable state
    sub = tmp_dir / "shot_diff"
    sub.mkdir()
    a = _write_image(sub, "a.png", (255, 0, 0))
    b = _write_image(sub, "b.png", (0, 0, 255))
    assert m.extract_features(a) is None
    assert m.extract_features(b) is None
    m.on_dispose()
    assert a.quality_metrics is None
    assert b.quality_metrics is None

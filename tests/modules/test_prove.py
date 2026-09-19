"""Deterministic tests for PROVE RC-S and RC-T."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from ayase.models import Sample
from ayase.modules import prove
from ayase.modules.prove import PROVEModule, _PatchGrid


class _FakeEncoder:
    def __init__(self):
        self.shapes = []

    def encode(self, image: np.ndarray) -> _PatchGrid:
        self.shapes.append(image.shape[:2])
        features = np.zeros((32, 32, 2), dtype=np.float32)
        features[:, :, 0] = 1.0
        return _PatchGrid(features, image.shape[:2], image.shape[:2])


def _grid(features: np.ndarray) -> _PatchGrid:
    return _PatchGrid(features, features.shape[:2], features.shape[:2])


def test_prove_metadata_and_defaults():
    module = PROVEModule()
    assert module.name == "prove"
    assert module.max_frames == 81
    assert module._backend is None
    assert module.models[0]["id"] == "facebook/dinov2-giant"
    assert "611a9d42f2335e0f921f1e313ad3c1b7178d206d" in module.models[0]["notes"]
    assert module.metric_groups == {
        "prove_rc_s_score": "nr_quality",
        "prove_rc_t_score": "temporal",
    }


def test_rbf_mmd_is_biased_normalized_and_scaled():
    x = np.asarray([[2.0, 0.0]], dtype=np.float32)
    y = np.asarray([[0.0, 7.0]], dtype=np.float32)
    expected = (2.0 - 2.0 * np.exp(-2.0 / (2.0 * 10.0**2))) * 1000.0
    assert prove._rbf_mmd(x, y) == pytest.approx(expected, rel=1e-5)
    assert prove._rbf_mmd(x, x) == pytest.approx(0.0, abs=1e-6)


def test_resize_longest_side_and_pad_with_rgb_mean():
    image = np.zeros((100, 33, 3), dtype=np.uint8)
    padded, resized_hw = prove._resize_and_pad_rgb(image)
    assert resized_hw == (448, 147)
    assert padded.shape == (448, 154, 3)
    assert np.all(padded[:, 147:] == np.asarray([123, 116, 103], dtype=np.uint8))


def test_mask_threshold_and_four_connected_component_filtering():
    assert prove._to_binary_mask(np.asarray([[127, 128]], dtype=np.uint8)).tolist() == [
        [0, 1]
    ]

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:5, 3:5] = 255
    mask[5, 5] = 255  # diagonal-only contact: a separate component
    assert prove._component_boxes(mask) == [(2, 2, 6, 6)]

    large = np.zeros((50, 50), dtype=np.uint8)
    large[5:25, 5:25] = 255
    large[35:45, 35:45] = 255
    boxes = prove._component_boxes(large)
    assert len(boxes) == 1  # the 100-pixel component is ignored beside a >=200 one


def test_rc_s_preserves_nonzero_soft_mask_pixels_for_feature_pooling(monkeypatch):
    module = PROVEModule()
    module._encoder = _FakeEncoder()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    mask[15, 16:48] = 1
    seen = []

    def fake_score(encoded, cropped_mask):
        seen.append(cropped_mask.copy())
        return 1.0

    monkeypatch.setattr(prove, "_rc_s_from_grid", fake_score)
    assert module._score_spatial_frame(image, mask) == 1.0
    assert any(np.any(crop == 1) for crop in seen)


def test_expanded_square_bbox_is_centered_expanded_and_bounded():
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[4:8, 6:10] = 1
    assert prove._expanded_square_bbox(mask) == (5, 3, 11, 9)

    edge = np.zeros((12, 16), dtype=np.uint8)
    edge[0:4, 0:2] = 1
    assert prove._expanded_square_bbox(edge) == (0, 0, 4, 4)


def test_rc_s_window_routing_and_nonnegative_presentation_clamp(monkeypatch):
    features = np.zeros((32, 32, 2), dtype=np.float32)
    features[:, :, 0] = 1.0
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[:, 16:] = 255
    calls = []

    def fake_mmd(x, y, sigma=10.0):
        calls.append((len(x), len(y), sigma))
        return -2.0

    monkeypatch.setattr(prove, "_rbf_mmd", fake_mmd)
    score = prove._rc_s_from_grid(_grid(features), mask)

    assert score == pytest.approx(1.0)
    assert (32, 32, 10.0) in calls  # mixed window: local foreground vs local background
    assert (64, 512, 10.0) in calls  # full foreground window vs all global background


def test_rc_t_accepts_exactly_five_intersection_points():
    features = np.zeros((32, 32, 2), dtype=np.float32)
    features[:, :, 0] = 1.0
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[0, :5] = 255
    score = prove._rc_t_from_grids(_grid(features), _grid(features.copy()), mask, mask)
    assert score == pytest.approx(0.0, abs=1e-6)


def test_image_process_uses_generated_image_and_mask_without_reference(tmp_path: Path):
    image_path = tmp_path / "result.png"
    mask_path = tmp_path / "result.mask.png"
    cv2.imwrite(str(image_path), np.full((64, 64, 3), 80, dtype=np.uint8))
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    cv2.imwrite(str(mask_path), mask)

    module = PROVEModule()
    module._encoder = _FakeEncoder()
    module._backend = "fake-dinov2"
    sample = Sample(path=image_path, is_video=False)
    result = module.process(sample)

    assert result is sample
    assert sample.reference_path is None
    assert sample.quality_metrics.prove_rc_s_score == pytest.approx(1.0)
    assert sample.quality_metrics.prove_rc_t_score is None


def test_video_process_keeps_sequential_frames_and_reports_both_metrics(
    tmp_path: Path, monkeypatch
):
    video_path = tmp_path / "result.mp4"
    mask_path = tmp_path / "mask.mp4"
    video_path.touch()
    mask_path.touch()
    frames = [
        np.full((64, 64, 3), value, dtype=np.uint8) for value in (20, 40, 60)
    ]
    masks = []
    for offset in (0, 1, 2):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:48, 16 + offset : 48 + offset] = 255
        masks.append(mask)

    module = PROVEModule({"max_frames": 3})
    module._encoder = _FakeEncoder()
    module._backend = "fake-dinov2"
    calls = []

    def fake_load(path, color):
        calls.append((Path(path), color))
        return frames if color == "rgb" else masks

    monkeypatch.setattr(module, "_load_video_frames", fake_load)
    sample = Sample(path=video_path, is_video=True, reference_mask_path=mask_path)
    result = module.process(sample)

    assert result is sample
    assert calls == [(video_path, "rgb"), (mask_path, "gray")]
    assert sample.quality_metrics.prove_rc_s_score == pytest.approx(1.0)
    assert sample.quality_metrics.prove_rc_t_score == pytest.approx(0.0, abs=1e-6)


def test_mask_resolution_priority_config_directory_and_sidecar(tmp_path: Path):
    media = tmp_path / "clip.png"
    media.touch()
    sidecar = tmp_path / "clip.mask.png"
    sidecar.touch()
    module = PROVEModule()
    sample = Sample(path=media, is_video=False)
    assert module._resolve_mask_path(sample) == sidecar

    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    directory_mask = mask_dir / "clip.png"
    directory_mask.touch()
    configured = PROVEModule({"mask_path": str(mask_dir)})
    assert configured._resolve_mask_path(sample) == directory_mask

    explicit = tmp_path / "explicit.png"
    explicit.touch()
    sample.reference_mask_path = explicit
    assert configured._resolve_mask_path(sample) == explicit


def test_process_gracefully_skips_without_backend_or_mask(tmp_path: Path):
    sample = Sample(path=tmp_path / "missing.png", is_video=False)
    assert PROVEModule().process(sample) is sample
    assert sample.quality_metrics is None

"""Tests for the camerabench module (CameraBench camera-motion classification)."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


# ---------------------------------------------------------------------------
# Module basics
# ---------------------------------------------------------------------------


def test_camerabench_basics():
    from ayase.modules.camerabench import CameraBenchModule

    _test_module_basics(CameraBenchModule, "camerabench")


# ---------------------------------------------------------------------------
# Taxonomy / label-set sanity
# ---------------------------------------------------------------------------


def test_camera_motion_label_set():
    from ayase.modules.camerabench import CAMERA_MOTION_LABELS

    expected = {
        "static",
        "pan_left",
        "pan_right",
        "tilt_up",
        "tilt_down",
        "roll",
        "zoom_in",
        "zoom_out",
        "dolly_in",
        "dolly_out",
        "truck",
        "pedestal",
        "follow",
    }
    assert expected <= set(CAMERA_MOTION_LABELS.keys())
    # Every label maps to a non-empty natural-language description.
    assert all(isinstance(v, str) and v for v in CAMERA_MOTION_LABELS.values())


# ---------------------------------------------------------------------------
# Metadata write with a mocked classifier (no model download)
# ---------------------------------------------------------------------------


def test_store_camera_motion_class_helper(video_sample):
    from ayase.modules.camerabench import _store_camera_motion_class

    _store_camera_motion_class(video_sample, "pan_left")
    meta = getattr(video_sample, "metadata", None)
    assert isinstance(meta, dict)
    assert meta["camera_motion_class"] == "pan_left"


def test_process_writes_label_and_confidence(video_sample):
    from ayase.modules.camerabench import CameraBenchModule

    m = CameraBenchModule()
    m._ml_available = True
    m._backend = "qwen2.5-vl-camerabench"
    # Mock the classifier so no model is loaded.
    m._classify = lambda sample: ("tilt_up", 0.87)

    video_sample.quality_metrics = QualityMetrics()
    result = m.process(video_sample)
    assert result is video_sample
    assert getattr(video_sample, "metadata")["camera_motion_class"] == "tilt_up"
    assert video_sample.quality_metrics.camera_motion_class_confidence == pytest.approx(0.87)


# ---------------------------------------------------------------------------
# Graceful unavailable: no VLM -> _backend "unavailable", both outputs unset
# ---------------------------------------------------------------------------


def test_graceful_unavailable(video_sample):
    from ayase.modules.camerabench import CameraBenchModule

    m = CameraBenchModule()

    def _fail():
        raise ImportError("transformers Qwen2.5-VL unavailable")

    m._load_qwen = _fail  # avoid any real model download
    m.setup()
    assert m._backend == "unavailable"
    assert m._ml_available is False

    video_sample.quality_metrics = QualityMetrics()
    result = m.process(video_sample)
    assert result is video_sample
    assert video_sample.quality_metrics.camera_motion_class_confidence is None
    assert getattr(video_sample, "metadata", None) is None

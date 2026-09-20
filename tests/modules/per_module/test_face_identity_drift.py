"""Face-identity drift diagnostics without model downloads or GPU use."""

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from ayase.models import Sample


def test_face_identity_drift_basics():
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule
    from tests.modules.conftest import _test_module_basics

    _test_module_basics(FaceIdentityDriftModule, "face_identity_drift")


def test_declares_only_transparent_component_metrics():
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule

    expected = {
        "face_identity_detection_coverage",
        "face_identity_similarity_p05",
        "face_identity_similarity_min",
        "face_identity_below_threshold_fraction",
        "face_identity_longest_below_threshold_run_fraction",
        "face_identity_drift_slope",
    }
    assert set(FaceIdentityDriftModule.metric_info) == expected
    assert FaceIdentityDriftModule.metric_groups == {field: "face" for field in expected}
    assert FaceIdentityDriftModule.models[0]["id"] == "insightface/buffalo_l"
    assert "non-commercial research" in FaceIdentityDriftModule.models[0]["notes"]


def test_threshold_is_opt_in_and_invalid_values_disable_it(caplog):
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule

    default = FaceIdentityDriftModule()
    assert default.similarity_threshold is None
    assert default.subsample == 32
    assert default.device == "auto"
    assert FaceIdentityDriftModule({"similarity_threshold": 0.42}).similarity_threshold == 0.42
    assert FaceIdentityDriftModule({"similarity_threshold": 1.1}).similarity_threshold is None
    assert "outside [0, 1]" in caplog.text


def test_setup_is_noop_in_test_mode():
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule

    module = FaceIdentityDriftModule({"test_mode": True})
    module.setup()
    assert module._face_app is None
    assert module._backend == "unavailable"


@pytest.mark.parametrize(
    ("device", "available", "providers", "ctx_id", "backend_suffix"),
    [
        (
            "cpu",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CPUExecutionProvider"],
            -1,
            ":cpu",
        ),
        (
            "auto",
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            ["CUDAExecutionProvider", "CPUExecutionProvider"],
            0,
            ":cuda",
        ),
        (
            "cuda",
            ["CPUExecutionProvider"],
            ["CPUExecutionProvider"],
            -1,
            ":cpu",
        ),
    ],
)
def test_setup_selects_available_provider_without_loading_models(
    monkeypatch, device, available, providers, ctx_id, backend_suffix
):
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule
    from ayase.pipeline import PipelineModule

    # The light test suite globally enables test mode.  Disable only for this
    # fully mocked setup call; monkeypatch restores both controls afterwards.
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)
    monkeypatch.setattr(PipelineModule, "_global_test_mode", False)

    created = []

    class FakeFaceAnalysis:
        def __init__(self, name, providers):
            self.name = name
            self.providers = providers
            self.prepare_args = None
            created.append(self)

        def prepare(self, **kwargs):
            self.prepare_args = kwargs

    ort = ModuleType("onnxruntime")
    ort.get_available_providers = lambda: list(available)
    insightface = ModuleType("insightface")
    insightface.__path__ = []
    insightface_app = ModuleType("insightface.app")
    insightface_app.FaceAnalysis = FakeFaceAnalysis
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "insightface", insightface)
    monkeypatch.setitem(sys.modules, "insightface.app", insightface_app)

    module = FaceIdentityDriftModule({"device": device})
    module.setup()

    assert len(created) == 1
    assert created[0].name == "buffalo_l"
    assert created[0].providers == providers
    assert created[0].prepare_args == {"ctx_id": ctx_id, "det_size": (640, 640)}
    assert module._face_app is created[0]
    assert module._backend.endswith(backend_suffix)


def test_process_skips_without_backend_or_reference(image_sample):
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule

    module = FaceIdentityDriftModule()
    assert module.process(image_sample) is image_sample
    assert image_sample.quality_metrics is None

    module._face_app = object()
    assert module.process(image_sample) is image_sample
    assert image_sample.quality_metrics is None


def test_summary_preserves_missing_frames_and_clips_cosines():
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule

    module = FaceIdentityDriftModule({"similarity_threshold": 0.5})
    series = [(0, 0.9), (1, 0.8), (2, 0.2), (3, 0.1), (5, 0.7), (6, -0.4), (7, 1.2)]

    result = module._summarize_series(series, sampled_frame_count=8)
    clipped = np.asarray([0.9, 0.8, 0.2, 0.1, 0.7, 0.0, 1.0])
    positions = np.asarray([0, 1, 2, 3, 5, 6, 7], dtype=float) / 7.0

    assert result["face_identity_detection_coverage"] == pytest.approx(7 / 8)
    assert result["face_identity_similarity_p05"] == pytest.approx(
        np.percentile(clipped, 5.0, method="linear")
    )
    assert result["face_identity_similarity_min"] == 0.0
    assert result["face_identity_below_threshold_fraction"] == pytest.approx(3 / 7)
    # Frames 2-3 are one run; missing frame 4 breaks it before bad frame 6.
    assert result["face_identity_longest_below_threshold_run_fraction"] == pytest.approx(2 / 8)
    assert result["face_identity_drift_slope"] == pytest.approx(
        np.polyfit(positions, clipped, 1)[0]
    )


def test_threshold_metrics_stay_unset_without_explicit_operating_point():
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule

    result = FaceIdentityDriftModule()._summarize_series(
        [(0, 0.8), (1, 0.7), (2, 0.6), (3, 0.5)],
        sampled_frame_count=4,
    )

    assert result["face_identity_below_threshold_fraction"] is None
    assert result["face_identity_longest_below_threshold_run_fraction"] is None


def test_slope_requires_four_detected_chronological_samples():
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule

    module = FaceIdentityDriftModule()
    short = module._summarize_series([(0, 0.9), (2, 0.7), (3, 0.6)], 4)
    enough = module._summarize_series(
        [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)], 4
    )

    assert short["face_identity_drift_slope"] is None
    assert enough["face_identity_drift_slope"] == pytest.approx(-0.3)


def test_empty_detection_series_reports_zero_coverage_only():
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule

    result = FaceIdentityDriftModule(
        {"similarity_threshold": 0.5}
    )._summarize_series([], sampled_frame_count=8)

    assert result["face_identity_detection_coverage"] == 0.0
    assert all(value is None for key, value in result.items() if key != "face_identity_detection_coverage")


class _Face:
    bbox = np.asarray([0.0, 0.0, 10.0, 10.0])

    def __init__(self, embedding):
        self.normed_embedding = np.asarray(embedding, dtype=float)


class _FaceApp:
    def __init__(self, frame_faces):
        self._frame_faces = list(frame_faces)

    def get(self, _frame):
        return self._frame_faces.pop(0)


def test_process_writes_all_applicable_metrics_without_loading_weights(monkeypatch):
    import ayase.modules.face_identity_drift as drift

    module = drift.FaceIdentityDriftModule({"similarity_threshold": 0.75})
    module._face_app = _FaceApp(
        [
            [_Face([1.0, 0.0])],
            [],
            [_Face([0.6, 0.8])],
            [_Face([0.0, 1.0])],
        ]
    )
    monkeypatch.setattr(module, "_reference_embedding", lambda _path: np.asarray([1.0, 0.0]))
    monkeypatch.setattr(
        drift,
        "sample_frames",
        lambda *_args, **_kwargs: [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(4)],
    )
    sample = Sample(
        path=Path("candidate.mp4"),
        is_video=True,
        reference_path=Path("reference.png"),
    )

    assert module.process(sample) is sample
    metrics = sample.quality_metrics
    assert metrics.face_identity_detection_coverage == pytest.approx(0.75)
    assert metrics.face_identity_similarity_p05 == pytest.approx(0.06)
    assert metrics.face_identity_similarity_min == 0.0
    assert metrics.face_identity_below_threshold_fraction == pytest.approx(2 / 3)
    # Missing frame 1 separates the acceptable first frame from bad frames 2-3.
    assert metrics.face_identity_longest_below_threshold_run_fraction == pytest.approx(0.5)
    assert metrics.face_identity_drift_slope is None


def test_models_catalog_renders_explicit_other_backend():
    from ayase.models_doc import generate_models_doc

    catalog = generate_models_doc(fetch_licenses=False)
    assert "## Other Models" in catalog
    assert "`insightface/buffalo_l`" in catalog
    assert "Non-commercial research" in catalog
    assert "`face_identity_drift`" in catalog


def test_metric_docs_do_not_invent_huggingface_url_for_other_model():
    import inspect

    from ayase.metrics_doc import _detect_source_links
    from ayase.modules.face_identity_drift import FaceIdentityDriftModule

    links = _detect_source_links(
        inspect.getsource(FaceIdentityDriftModule), FaceIdentityDriftModule
    )
    assert "github.com/deepinsight/insightface" in links
    assert "huggingface.co/insightface/buffalo_l" not in links

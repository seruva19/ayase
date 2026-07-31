"""AdaFace identity-similarity module contract tests (no weights required)."""

import numpy as np
import pytest

from ayase.models import QualityMetrics, Sample


def test_adaface_basics():
    from ayase.modules.adaface import AdaFaceModule
    from .conftest import _test_module_basics

    _test_module_basics(AdaFaceModule, "adaface")


def test_adaface_config():
    from ayase.modules.adaface import AdaFaceModule

    m = AdaFaceModule()
    for key in ("checkpoint", "face_model", "subsample", "warning_threshold", "device"):
        assert key in m.default_config
    assert m.default_config["checkpoint"] in _known_checkpoints()


def _known_checkpoints():
    from ayase.modules.adaface import _ADAFACE_CHECKPOINTS

    return set(_ADAFACE_CHECKPOINTS)


def test_adaface_checkpoints_are_pinned():
    """Every checkpoint must pin a full 40-hex HF revision and a known arch."""
    from ayase.modules.adaface import _ADAFACE_CHECKPOINTS, _weight_url

    for key, entry in _ADAFACE_CHECKPOINTS.items():
        assert entry["repo"].startswith("minchul/cvlface_adaface_"), key
        assert len(entry["revision"]) == 40, key
        assert all(c in "0123456789abcdef" for c in entry["revision"]), key
        assert entry["arch"] in ("ir18", "ir50", "ir101"), key
        assert _weight_url(entry).endswith("/model.safetensors"), key


def test_adaface_declares_models_and_metric():
    from ayase.modules.adaface import _ADAFACE_CHECKPOINTS, AdaFaceModule

    assert len(AdaFaceModule.models) == len(_ADAFACE_CHECKPOINTS)
    assert all(entry["type"] == "huggingface" for entry in AdaFaceModule.models)
    assert "adaface_identity_similarity" in AdaFaceModule.metric_info


def test_adaface_skip_without_backend(image_sample):
    from ayase.modules.adaface import AdaFaceModule

    m = AdaFaceModule()  # setup() not called → no net, no detector
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.adaface_identity_similarity is None


def test_adaface_skip_without_reference(image_sample):
    from ayase.modules.adaface import AdaFaceModule

    m = AdaFaceModule()
    m._net = object()  # pretend the backbone is loaded
    m._face_app = object()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.adaface_identity_similarity is None


def test_adaface_unknown_checkpoint_disables_module(caplog):
    from ayase.modules.adaface import AdaFaceModule

    m = AdaFaceModule(config={"checkpoint": "does_not_exist"})
    m.setup()
    assert m._net is None
    assert m._backend == "unavailable"


def test_adaface_setup_is_noop_in_test_mode():
    from ayase.modules.adaface import AdaFaceModule

    m = AdaFaceModule(config={"test_mode": True})
    assert m.test_mode
    m.setup()
    assert m._backend == "unavailable"
    assert m._net is None


def test_adaface_scores_reference_pair(tmp_dir, monkeypatch):
    """End-to-end process() with stubbed detector/backbone: mean cosine, clipped."""
    import cv2

    from ayase.modules.adaface import AdaFaceModule

    img = np.full((64, 64, 3), 127, dtype=np.uint8)
    ref_path = tmp_dir / "ref.png"
    gen_path = tmp_dir / "gen.png"
    cv2.imwrite(str(ref_path), img)
    cv2.imwrite(str(gen_path), img)

    m = AdaFaceModule()
    m._net = object()
    m._face_app = object()

    embeddings = iter([
        np.array([1.0, 0.0, 0.0]),  # reference
        np.array([0.6, 0.8, 0.0]),  # frame → cosine 0.6
    ])
    monkeypatch.setattr(m, "_embed_face", lambda frame: next(embeddings))

    sample = Sample(path=gen_path, is_video=False, reference_path=ref_path)
    result = m.process(sample)

    assert result.quality_metrics is not None
    assert result.quality_metrics.adaface_identity_similarity == pytest.approx(0.6, abs=1e-6)


def test_adaface_negative_cosine_clipped_to_zero(tmp_dir, monkeypatch):
    import cv2

    from ayase.modules.adaface import AdaFaceModule

    img = np.full((64, 64, 3), 127, dtype=np.uint8)
    ref_path = tmp_dir / "ref.png"
    gen_path = tmp_dir / "gen.png"
    cv2.imwrite(str(ref_path), img)
    cv2.imwrite(str(gen_path), img)

    m = AdaFaceModule()
    m._net = object()
    m._face_app = object()
    embeddings = iter([np.array([1.0, 0.0]), np.array([-1.0, 0.0])])
    monkeypatch.setattr(m, "_embed_face", lambda frame: next(embeddings))

    sample = Sample(path=gen_path, is_video=False, reference_path=ref_path)
    result = m.process(sample)
    assert result.quality_metrics.adaface_identity_similarity == 0.0
    assert any("Low AdaFace identity similarity" in i.message for i in result.validation_issues)


def test_adaface_no_face_detected_warns(tmp_dir, monkeypatch):
    import cv2

    from ayase.modules.adaface import AdaFaceModule

    img = np.full((64, 64, 3), 127, dtype=np.uint8)
    ref_path = tmp_dir / "ref.png"
    gen_path = tmp_dir / "gen.png"
    cv2.imwrite(str(ref_path), img)
    cv2.imwrite(str(gen_path), img)

    m = AdaFaceModule()
    m._net = object()
    m._face_app = object()
    calls = iter([np.array([1.0, 0.0]), None])
    monkeypatch.setattr(m, "_embed_face", lambda frame: next(calls))

    sample = Sample(path=gen_path, is_video=False, reference_path=ref_path)
    result = m.process(sample)
    assert result.quality_metrics is None or result.quality_metrics.adaface_identity_similarity is None
    assert any("no face detected" in i.message.lower() for i in result.validation_issues)


class _FakeFace:
    def __init__(self, bbox, kps):
        self.bbox = bbox
        self.kps = kps


class _PaddingOnlyDetector:
    """Detects a face only once the frame has been padded (tight-crop case)."""

    def __init__(self, tight_size):
        self.tight_size = tight_size
        self.calls = []

    def get(self, image):
        self.calls.append(image.shape[:2])
        if image.shape[0] <= self.tight_size:
            return []
        return [_FakeFace((0, 0, 10, 10), np.zeros((5, 2), dtype=np.float32))]


def test_adaface_pad_retry_rescues_tight_crop():
    from ayase.modules.adaface import AdaFaceModule

    m = AdaFaceModule()
    m._face_app = _PaddingOnlyDetector(tight_size=112)

    frame = np.zeros((112, 112, 3), dtype=np.uint8)
    face, detect_image = m._detect_largest_face(frame)

    assert face is not None
    assert detect_image.shape[0] == 112 + 2 * int(112 * m.pad_retry)
    assert len(m._face_app.calls) == 2  # original, then padded


def test_adaface_pad_retry_disabled():
    from ayase.modules.adaface import AdaFaceModule

    m = AdaFaceModule(config={"pad_retry": 0})
    m._face_app = _PaddingOnlyDetector(tight_size=112)

    face, detect_image = m._detect_largest_face(np.zeros((112, 112, 3), dtype=np.uint8))
    assert face is None
    assert detect_image.shape[0] == 112
    assert len(m._face_app.calls) == 1


def test_adaface_field_and_group():
    qm = QualityMetrics()
    assert hasattr(qm, "adaface_identity_similarity")
    assert qm.adaface_identity_similarity is None

    from ayase.pipeline import ModuleRegistry

    ModuleRegistry.discover_modules()  # folds module metric_groups into the registry
    assert qm._FIELD_GROUPS.get("adaface_identity_similarity") == "face"


def test_adaface_vendored_backbone_shapes():
    """The vendored IResNet must expose the AdaFace 112x112 → 512-d contract."""
    torch = pytest.importorskip("torch")

    from ayase.third_party.adaface import IR_18

    net = IR_18(input_size=(112, 112), output_dim=512).eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 112, 112))
    assert out.shape == (1, 512)

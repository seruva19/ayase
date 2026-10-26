"""Tests for ocr_fidelity module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_ocr_fidelity_basics():
    from ayase.modules.ocr_fidelity import OCRFidelityModule
    _test_module_basics(OCRFidelityModule, "ocr_fidelity")


@pytest.mark.parametrize("api", ["v2", "v3"])
def test_missing_ocr_text_is_deletion_error(image_sample, monkeypatch, api):
    import numpy as np
    from types import SimpleNamespace
    from ayase.modules.ocr_fidelity import OCRFidelityModule

    module = OCRFidelityModule({"expected_text": "HELLO WORLD"})
    module._ocr_available = True
    module._ocr_api = api
    module._ocr = SimpleNamespace(
        predict=lambda frame: [{"rec_texts": []}],
        ocr=lambda frame, cls: [None],
    )
    monkeypatch.setattr(module, "_load_frames", lambda sample: [np.zeros((8, 8, 3))])

    assert module.process(image_sample) is image_sample
    metrics = image_sample.quality_metrics
    assert metrics.ocr_score == pytest.approx(1.0)
    assert metrics.ocr_fidelity == pytest.approx(1.0)
    assert metrics.ocr_cer == pytest.approx(1.0)
    assert metrics.ocr_wer == pytest.approx(1.0)


def test_character_error_rate_fallback_uses_reference_length(monkeypatch):
    import sys
    from ayase.modules.ocr_fidelity import _character_error_rate

    monkeypatch.setitem(sys.modules, "cer", None)
    assert _character_error_rate("ab", "abcd") == pytest.approx(0.5)
    assert _character_error_rate("abcdef", "abcd") == pytest.approx(0.5)
    assert _character_error_rate("", "abcd") == pytest.approx(1.0)
    assert _character_error_rate("", "") == pytest.approx(0.0)

def test_ocr_fidelity_image(image_sample):
    from ayase.modules.ocr_fidelity import OCRFidelityModule
    image_sample.quality_metrics = QualityMetrics()
    m = OCRFidelityModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_ocr_fidelity_video(video_sample):
    from ayase.modules.ocr_fidelity import OCRFidelityModule
    video_sample.quality_metrics = QualityMetrics()
    m = OCRFidelityModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_ocr_fidelity_paddleocr_v2_api(tmp_path, monkeypatch):
    """PaddleOCR 2.x exposes ``.ocr(img, cls=True)`` returning nested lists —
    the module must extract texts through that API when ``predict`` is absent."""
    import sys
    import types
    import cv2
    import numpy as np
    import ayase._compat
    from ayase.models import Sample, CaptionMetadata
    from ayase.modules.ocr_fidelity import OCRFidelityModule

    monkeypatch.setattr(ayase._compat, "ensure_paddle_gpu", lambda: None)

    class _FakeV2PaddleOCR:
        """API-surface stand-in for paddleocr<3.0 (no .predict)."""
        def __init__(self, **kw):
            assert kw.get("use_angle_cls") is True

        def ocr(self, img, cls=True):
            return [[[None, ("HELLO", 0.99)]]]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = _FakeV2PaddleOCR
    sys.modules["paddleocr"] = fake_module
    img_path = tmp_path / "frame.png"
    cv2.imwrite(str(img_path), np.zeros((64, 256, 3), dtype=np.uint8))
    caption = 'a sign that says "HELLO"'
    sample = Sample(
        path=img_path,
        is_video=False,
        caption=CaptionMetadata(text=caption, length=len(caption)),
    )
    try:
        m = OCRFidelityModule()
        m.setup()
        assert m._ocr_api == "v2"
        m.process(sample)
        assert sample.quality_metrics is not None
        assert sample.quality_metrics.ocr_score == pytest.approx(0.0)
    finally:
        sys.modules.pop("paddleocr", None)


def test_ocr_fidelity_paddleocr_v3_api(tmp_path, monkeypatch):
    """PaddleOCR 3.x exposes ``.predict(img)`` returning rec_texts dicts."""
    import sys
    import types
    import cv2
    import numpy as np
    import ayase._compat
    from ayase.models import Sample, CaptionMetadata
    from ayase.modules.ocr_fidelity import OCRFidelityModule

    monkeypatch.setattr(ayase._compat, "ensure_paddle_gpu", lambda: None)

    class _FakeV3PaddleOCR:
        def __init__(self, **kw):
            pass

        def predict(self, img):
            return [{"rec_texts": ["HELLO"]}]

    fake_module = types.ModuleType("paddleocr")
    fake_module.PaddleOCR = _FakeV3PaddleOCR
    sys.modules["paddleocr"] = fake_module
    img_path = tmp_path / "frame.png"
    cv2.imwrite(str(img_path), np.zeros((64, 256, 3), dtype=np.uint8))
    caption = 'a sign that says "HELLO"'
    sample = Sample(
        path=img_path,
        is_video=False,
        caption=CaptionMetadata(text=caption, length=len(caption)),
    )
    try:
        m = OCRFidelityModule()
        m.setup()
        assert m._ocr_api == "v3"
        m.process(sample)
        assert sample.quality_metrics is not None
        assert sample.quality_metrics.ocr_score == pytest.approx(0.0)
    finally:
        sys.modules.pop("paddleocr", None)

"""Tests for text_detection module."""

import numpy as np

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_text_detection_basics():
    from ayase.modules.text import TextDetectionModule
    _test_module_basics(TextDetectionModule, "text_detection")

def test_text_detection_image(image_sample):
    from ayase.modules.text import TextDetectionModule
    image_sample.quality_metrics = QualityMetrics()
    m = TextDetectionModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_text_detection_video(video_sample):
    from ayase.modules.text import TextDetectionModule
    video_sample.quality_metrics = QualityMetrics()
    m = TextDetectionModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_text_detection_parses_paddleocr_v2_output():
    from ayase.modules.text import TextDetectionModule

    class FakePaddleV2:
        def ocr(self, image, cls=True):
            assert cls is True
            return [[
                [
                    [[1, 2], [5, 2], [5, 6], [1, 6]],
                    ("AYASE", 0.95),
                ]
            ]]

    module = TextDetectionModule()
    module._model = FakePaddleV2()
    module._ocr_api = "v2"

    detections = module._paddle_detections(np.zeros((8, 8, 3), dtype=np.uint8))

    assert detections == [
        ([[1, 2], [5, 2], [5, 6], [1, 6]], "AYASE", 0.95)
    ]


def test_text_detection_v2_output_feeds_existing_area_metric(image_sample, monkeypatch):
    from ayase.modules.text import TextDetectionModule
    from ayase.utils.sampling import FrameSampler

    class FakePaddleV2:
        def ocr(self, image, cls=True):
            return [[
                [
                    [[1, 2], [5, 2], [5, 6], [1, 6]],
                    ("AYASE", 0.95),
                ]
            ]]

    monkeypatch.setattr(
        FrameSampler,
        "sample_frames",
        lambda path, num_frames: [np.zeros((10, 10, 3), dtype=np.uint8)],
    )
    module = TextDetectionModule()
    module._model = FakePaddleV2()
    module._ocr_api = "v2"
    module._engine = "paddle"
    module._ocr_available = True

    result = module.process(image_sample)

    assert result is image_sample
    assert result.quality_metrics is not None
    assert result.quality_metrics.ocr_area_ratio == 0.16
    assert result.validation_issues[-1].details["detected_text"] == ["AYASE"]


def test_text_detection_parses_paddleocr_v3_numpy_output():
    from ayase.modules.text import TextDetectionModule

    class FakePaddleV3:
        def predict(self, image):
            return [{
                "dt_polys": np.asarray([[[2, 3], [8, 3], [8, 7], [2, 7]]]),
                "rec_texts": ["TEXT"],
                "rec_scores": np.asarray([0.9]),
            }]

    module = TextDetectionModule()
    module._model = FakePaddleV3()
    module._ocr_api = "v3"

    detections = module._paddle_detections(np.zeros((10, 10, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert np.array_equal(detections[0][0], [[2, 3], [8, 3], [8, 7], [2, 7]])
    assert detections[0][1] == "TEXT"
    assert detections[0][2] == 0.9


def test_text_detection_empty_paddle_outputs_are_real_empty_detections():
    from ayase.modules.text import TextDetectionModule

    class FakePaddleV2:
        def ocr(self, image, cls=True):
            return [None]

    class FakePaddleV3:
        def predict(self, image):
            return [{
                "dt_polys": np.empty((0, 4, 2)),
                "rec_texts": [],
                "rec_scores": np.empty((0,)),
            }]

    module = TextDetectionModule()
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    module._model = FakePaddleV2()
    module._ocr_api = "v2"
    assert module._paddle_detections(image) == []

    module._model = FakePaddleV3()
    module._ocr_api = "v3"
    assert module._paddle_detections(image) == []

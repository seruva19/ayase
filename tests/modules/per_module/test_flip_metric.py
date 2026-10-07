"""FLIP (ꟻLIP, Andersson et al. HPG 2020) module tests.

Official ``flip_evaluator.evaluate(reference, test, "LDR")`` returns a
``(error_map, mean_error, parameters)`` tuple — the module must pass the
tone-mapping mode argument and take the mean, not cast the tuple to float.
"""

import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from ayase.models import QualityMetrics
from ayase.modules.flip_metric import FLIPModule


class _FakeFlipEvaluator:
    def __init__(self, mean=0.42):
        self.calls = []
        self.mean = mean

    def evaluate(self, reference, test, mode):
        self.calls.append(mode)
        return np.zeros((4, 4)), self.mean, {}


def test_flip_evaluator_uses_ldr_and_mean(monkeypatch, image_sample, tmp_path):
    fake = _FakeFlipEvaluator(mean=0.42)
    monkeypatch.setitem(sys.modules, "flip_evaluator", fake)

    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), np.zeros((32, 32, 3), dtype=np.uint8))
    image_sample.reference_path = ref
    image_sample.quality_metrics = QualityMetrics()

    m = FLIPModule()
    m._backend = "flip_evaluator"
    m._ml_available = True
    out = m.process(image_sample)

    assert fake.calls == ["LDR"]
    assert out.quality_metrics.flip_score == pytest.approx(0.42)


def test_flip_unavailable_returns_none(monkeypatch, image_sample, tmp_path):
    m = FLIPModule()
    m._backend = "unavailable"
    m._ml_available = False
    image_sample.reference_path = tmp_path / "ref.png"
    out = m.process(image_sample)
    assert out.quality_metrics is None or out.quality_metrics.flip_score is None

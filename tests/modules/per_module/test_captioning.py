"""Tests for captioning module."""

import sys
import types

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_captioning_basics():
    from ayase.modules.captioning import CaptioningModule
    _test_module_basics(CaptioningModule, "captioning")

def test_captioning_image(image_sample):
    from ayase.modules.captioning import CaptioningModule
    image_sample.quality_metrics = QualityMetrics()
    m = CaptioningModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_captioning_video(video_sample):
    from ayase.modules.captioning import CaptioningModule
    video_sample.quality_metrics = QualityMetrics()
    m = CaptioningModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_blip_bleu_selects_requested_order_from_pycocoevalcap(monkeypatch):
    """Each BLEU-n maximum uses component n, matching pycocoevalcap semantics."""
    from ayase.modules.captioning import CaptioningModule

    class FakeBleu:
        def __init__(self, n):
            self.n = n

        def compute_score(self, references, hypotheses):
            assert references == {0: ["reference"]}
            value = 0.1 if hypotheses == {0: ["weak"]} else 0.2
            return [value * order for order in range(1, self.n + 1)], None

    bleu_module = types.ModuleType("pycocoevalcap.bleu.bleu")
    bleu_module.Bleu = FakeBleu
    monkeypatch.setitem(sys.modules, "pycocoevalcap", types.ModuleType("pycocoevalcap"))
    monkeypatch.setitem(sys.modules, "pycocoevalcap.bleu", types.ModuleType("pycocoevalcap.bleu"))
    monkeypatch.setitem(sys.modules, "pycocoevalcap.bleu.bleu", bleu_module)

    module = CaptioningModule()
    module._bleu_available = True

    # Best strong-caption BLEU orders are .2, .4, .6 and .8; their mean is .5.
    assert module._compute_blip_bleu("reference", ["weak", "strong"]) == 0.5


def test_blip_bleu_rejects_incomplete_component_vector(monkeypatch):
    """Malformed scorer output leaves the metric unset instead of changing orders."""
    from ayase.modules.captioning import CaptioningModule

    class IncompleteBleu:
        def __init__(self, n):
            self.n = n

        def compute_score(self, references, hypotheses):
            return [0.25], None

    bleu_module = types.ModuleType("pycocoevalcap.bleu.bleu")
    bleu_module.Bleu = IncompleteBleu
    monkeypatch.setitem(sys.modules, "pycocoevalcap", types.ModuleType("pycocoevalcap"))
    monkeypatch.setitem(sys.modules, "pycocoevalcap.bleu", types.ModuleType("pycocoevalcap.bleu"))
    monkeypatch.setitem(sys.modules, "pycocoevalcap.bleu.bleu", bleu_module)

    module = CaptioningModule()
    module._bleu_available = True
    assert module._compute_blip_bleu("reference", ["hypothesis"]) is None

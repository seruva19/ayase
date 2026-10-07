"""Tests for dnsmos module."""

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_dnsmos_basics():
    from ayase.modules.dnsmos import DNSMOSModule
    _test_module_basics(DNSMOSModule, "dnsmos")

def test_dnsmos_image(image_sample):
    from ayase.modules.dnsmos import DNSMOSModule
    image_sample.quality_metrics = QualityMetrics()
    m = DNSMOSModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample

def test_dnsmos_video(video_sample):
    from ayase.modules.dnsmos import DNSMOSModule
    video_sample.quality_metrics = QualityMetrics()
    m = DNSMOSModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


class _FakeDNSMOS4:
    """torchmetrics-style metric returning [p808_mos, mos_sig, mos_bak, mos_ovr]."""

    def __init__(self, fs=None, personalized=False):
        pass

    def __call__(self, audio):
        import torch

        return torch.tensor([[1.1, 2.2, 3.3, 4.4]])


class _FakeDNSMOSSingle:
    def __init__(self, fs=None, personalized=False):
        pass

    def __call__(self, audio):
        import torch

        return torch.tensor([3.5])


def _audio():
    import numpy as np

    return np.zeros(16000, dtype=np.float32)


def test_dnsmos_torchmetrics_field_mapping():
    """torchmetrics returns [p808, sig, bak, ovr] — each field must map to its
    own index; P.808 lands in the dedicated dnsmos_p808 field."""
    from ayase.modules.dnsmos import DNSMOSModule

    m = DNSMOSModule()
    m._metric_cls = _FakeDNSMOS4
    scores = m._compute_torchmetrics(_audio(), 16000)
    assert scores == pytest.approx(
        {"p808": 1.1, "sig": 2.2, "bak": 3.3, "ovrl": 4.4}
    )


def test_dnsmos_single_value_not_copied():
    """A single-value backend result must not be copied into all fields."""
    from ayase.modules.dnsmos import DNSMOSModule

    m = DNSMOSModule()
    m._metric_cls = _FakeDNSMOSSingle
    scores = m._compute_torchmetrics(_audio(), 16000)
    assert scores is None or all(v is None for v in scores.values())


def test_dnsmos_dict_result_named_keys():
    """Dict results map by named key; absent keys stay None (no 0.0 fill)."""
    from ayase.modules.dnsmos import DNSMOSModule

    class _FakeDict:
        def __init__(self, fs=None, personalized=False):
            pass

        def __call__(self, audio):
            return {"mos_sig": 2.5, "mos_bak": 3.5, "mos_ovr": 4.0, "p808_mos": 1.5}

    m = DNSMOSModule()
    m._metric_cls = _FakeDict
    scores = m._compute_torchmetrics(_audio(), 16000)
    assert scores["sig"] == 2.5
    assert scores["bak"] == 3.5
    assert scores["ovrl"] == 4.0
    assert scores["p808"] == 1.5

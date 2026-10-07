"""Tests for head_beat_align (Bailando kernel on head-pose velocity)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_head_beat_align_basics():
    from ayase.modules.head_beat_align import HeadBeatAlignModule
    _test_module_basics(HeadBeatAlignModule, "head_beat_align")


def test_head_beat_align_provenance():
    from ayase.modules.head_beat_align import HeadBeatAlignModule
    meta = HeadBeatAlignModule.get_metadata()
    assert meta["provenance"]["head_beat_align"] == "adapted"
    assert "Bailando" in meta["sources"]["head_beat_align"]


def test_head_beat_align_no_backend(tmp_path):
    from ayase.modules.head_beat_align import HeadBeatAlignModule
    m = HeadBeatAlignModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.head_beat_align is None


def test_kinematic_beats_finds_minima():
    from ayase.modules.head_beat_align import _head_kinematic_beats
    t = np.arange(101, dtype=np.float32)
    params = np.zeros((101, 12), dtype=np.float32)
    params[:, 0] = np.sin(t * 0.3)  # oscillating pose -> real velocity minima
    beats = _head_kinematic_beats(params, np.arange(101))
    assert len(beats) >= 1


def test_kinematic_beats_short_sequence_empty():
    from ayase.modules.head_beat_align import _head_kinematic_beats
    params = np.zeros((3, 12), dtype=np.float32)
    assert len(_head_kinematic_beats(params, np.arange(3))) == 0


def test_head_beat_align_no_audio_skips(tmp_path, monkeypatch):
    from ayase.modules import head_beat_align as hba
    m = hba.HeadBeatAlignModule()
    m._backend = "tddfa"
    m._extractor = object()
    monkeypatch.setattr(hba, "_has_audio_stream", lambda p: False)
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.head_beat_align is None


def test_head_beat_align_image_skips(tmp_path):
    from ayase.modules.head_beat_align import HeadBeatAlignModule
    m = HeadBeatAlignModule()
    m._backend = "tddfa"
    m._extractor = object()
    sample = Sample(path=tmp_path / "i.png", is_video=False)
    assert m.process(sample).quality_metrics is None

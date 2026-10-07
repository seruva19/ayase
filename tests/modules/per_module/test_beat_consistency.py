"""Tests for beat_consistency (BEAT/EMAGE BC kernel on MediaPipe pose)."""

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_beat_consistency_basics():
    from ayase.modules.beat_consistency import BeatConsistencyModule
    _test_module_basics(BeatConsistencyModule, "beat_consistency")


def test_beat_consistency_provenance():
    from ayase.modules.beat_consistency import BeatConsistencyModule
    meta = BeatConsistencyModule.get_metadata()
    assert meta["provenance"]["beat_consistency"] == "adapted"
    assert "BEAT" in meta["sources"]["beat_consistency"]


def test_beat_consistency_no_backend(tmp_path):
    from ayase.modules.beat_consistency import BeatConsistencyModule
    m = BeatConsistencyModule()
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.beat_consistency is None


def test_kinematic_beats_sec_finds_minima():
    from ayase.modules.beat_consistency import BeatConsistencyModule
    t = np.arange(200, dtype=np.float32)
    seq = np.zeros((200, 33, 4), dtype=np.float32)
    seq[:, 0, 0] = np.sin(t * 0.3)  # oscillating joint -> real minima
    beats = BeatConsistencyModule._kinematic_beats_sec(seq, t.astype(int), 30.0)
    assert len(beats) >= 1
    assert np.all((beats >= 0) & (beats <= 200 / 30))


def test_beat_consistency_no_audio_skips(tmp_path, monkeypatch):
    from ayase.modules import beat_consistency as bc
    m = bc.BeatConsistencyModule()
    m._backend = "mediapipe"
    m._pose = object()
    monkeypatch.setattr(bc, "_has_audio_stream", lambda p: False)
    sample = Sample(path=tmp_path / "v.mp4", is_video=True)
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.beat_consistency is None

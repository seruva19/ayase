"""Tests for the speaker_sim module (SIM-o, WavLM-TDNN)."""

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample


def test_speaker_sim_basics():
    from ayase.modules.speaker_sim import SpeakerSimModule
    _test_module_basics(SpeakerSimModule, "speaker_sim")


def test_speaker_sim_without_backend_leaves_field_unset(tmp_path):
    from ayase.modules.speaker_sim import SpeakerSimModule
    m = SpeakerSimModule()
    sample = Sample(path=tmp_path / "a.wav", is_video=False, reference_path=tmp_path / "b.wav")
    sample.quality_metrics = QualityMetrics()
    assert m.process(sample).quality_metrics.sim_o is None


def test_speaker_sim_uses_vendored_wavlm():
    from ayase.modules.speaker_sim import SpeakerSimModule
    assert "s3prl_hub" not in SpeakerSimModule().config  # WavLM upstream is vendored, no torch.hub

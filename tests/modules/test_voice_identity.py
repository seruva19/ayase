"""Tests for speaker-identity similarity against a reference set.

The properties held here are the ones that make the number trustworthy rather than
merely present: a reference set is required, a clip with no usable speech yields no
score instead of a zero, and coverage reports how much of the reference set was
actually usable.
"""

from pathlib import Path

import numpy as np

from tests.modules.conftest import _test_module_basics


def _module(**config):
    from ayase.modules.voice_identity import VoiceIdentityModule

    return VoiceIdentityModule(config or None)


def test_voice_identity_basics():
    from ayase.modules.voice_identity import VoiceIdentityModule

    _test_module_basics(VoiceIdentityModule, "voice_identity")


def test_no_encoder_leaves_sample_untouched():
    from ayase.models import Sample

    module = _module()
    sample = Sample(path=Path("clip.mp4"), is_video=True, reference_path=Path("ref"))
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_reference_is_required():
    from ayase.models import Sample

    module = _module()
    module._encoder = object()
    sample = Sample(path=Path("clip.mp4"), is_video=True)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_reference_directory_lists_media_only(tmp_path):
    module = _module()
    for name in ("a.wav", "b.mp4", "c.txt", "d.flac"):
        (tmp_path / name).write_bytes(b"")
    files = module._reference_files(tmp_path)
    assert [f.name for f in files] == ["a.wav", "b.mp4", "d.flac"]


def test_reference_directory_is_capped(tmp_path):
    module = _module(max_references=2)
    for index in range(5):
        (tmp_path / f"{index}.wav").write_bytes(b"")
    assert len(module._reference_files(tmp_path)) == 2


def test_missing_reference_yields_no_files(tmp_path):
    module = _module()
    assert module._reference_files(tmp_path / "absent.wav") == []


def test_scores_and_coverage_are_reported(monkeypatch, tmp_path):
    """Coverage must state how much of the reference set was usable.

    A reference set where half the files carry no speech is not the same evidence
    as one where all of them do, and the score alone cannot say which happened.
    """
    from ayase.models import Sample

    module = _module()
    module._encoder = object()
    for name in ("r0.wav", "r1.wav", "r2.wav", "r3.wav"):
        (tmp_path / name).write_bytes(b"")

    same = np.array([1.0, 0.0], dtype=np.float64)
    near = np.array([0.8, 0.6], dtype=np.float64)

    def fake_embed(self, path, workdir, tag):
        if tag == "sample":
            return same
        return {"ref0": same, "ref1": near}.get(tag)

    monkeypatch.setattr(type(module), "_embed", fake_embed)
    sample = Sample(path=tmp_path / "clip.mp4", is_video=True, reference_path=tmp_path)
    sample = module.process(sample)

    qm = sample.quality_metrics
    assert qm is not None
    assert qm.voice_identity_max == 1.0
    assert qm.voice_identity == 0.9
    assert qm.voice_identity_coverage == 0.5


def test_sample_without_speech_yields_no_score(monkeypatch, tmp_path):
    from ayase.models import Sample

    module = _module()
    module._encoder = object()
    (tmp_path / "r0.wav").write_bytes(b"")

    monkeypatch.setattr(type(module), "_embed", lambda self, path, workdir, tag: None)
    sample = Sample(path=tmp_path / "clip.mp4", is_video=True, reference_path=tmp_path)
    sample = module.process(sample)
    assert sample.quality_metrics is None

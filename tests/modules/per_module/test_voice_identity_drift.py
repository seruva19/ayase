"""Temporal speaker-identity drift tests without model downloads."""

from pathlib import Path

import numpy as np
import pytest

from ayase.models import Sample


def test_voice_identity_drift_basics():
    from ayase.modules.voice_identity_drift import VoiceIdentityDriftModule
    from tests.modules.conftest import _test_module_basics

    _test_module_basics(VoiceIdentityDriftModule, "voice_identity_drift")


def test_contract_is_component_only_and_threshold_is_opt_in(caplog):
    from ayase.modules.voice_identity_drift import VoiceIdentityDriftModule

    expected = {
        "voice_identity_window_coverage",
        "voice_identity_reference_coverage",
        "voice_identity_similarity_p05",
        "voice_identity_similarity_min",
        "voice_identity_below_threshold_fraction",
        "voice_identity_longest_below_threshold_run_fraction",
        "voice_identity_drift_slope",
    }
    assert set(VoiceIdentityDriftModule.metric_info) == expected
    assert VoiceIdentityDriftModule.metric_groups == {field: "audio" for field in expected}
    assert VoiceIdentityDriftModule.models[0]["id"] == "speechbrain/spkrec-ecapa-voxceleb"

    default = VoiceIdentityDriftModule()
    assert default.similarity_threshold is None
    assert default.window_seconds == 3.0
    assert default.hop_seconds == 1.5
    assert default.min_window_seconds == 1.0
    assert default.silence_rms_threshold == 1e-4
    assert VoiceIdentityDriftModule({"similarity_threshold": -0.25}).similarity_threshold == -0.25
    assert VoiceIdentityDriftModule({"similarity_threshold": 1.1}).similarity_threshold is None
    assert "outside [-1, 1]" in caplog.text


def test_setup_is_noop_in_test_mode():
    from ayase.modules.voice_identity_drift import VoiceIdentityDriftModule

    module = VoiceIdentityDriftModule({"test_mode": True})
    module.setup()
    assert module._encoder is None
    assert module._backend == "unavailable"


def test_exact_overlapping_window_schedule_and_short_tail():
    from ayase.modules.voice_identity_drift import TARGET_RATE, VoiceIdentityDriftModule

    module = VoiceIdentityDriftModule()
    # 7.2 s: four full 3 s windows plus a final 1.2 s window at t=6.0 s.
    assert module._window_bounds(int(7.2 * TARGET_RATE)) == [
        (0, 48_000),
        (24_000, 72_000),
        (48_000, 96_000),
        (72_000, 115_200),
        (96_000, 115_200),
    ]
    assert module._window_bounds(TARGET_RATE) == [(0, TARGET_RATE)]
    assert module._window_bounds(TARGET_RATE - 1) == []


def test_rms_eligibility_is_strict_and_rejects_nonfinite_audio():
    from ayase.modules.voice_identity_drift import TARGET_RATE, VoiceIdentityDriftModule

    module = VoiceIdentityDriftModule()
    assert not module._eligible(np.zeros(TARGET_RATE, dtype=np.float32), 1.0)
    assert not module._eligible(np.full(TARGET_RATE, 1e-4, dtype=np.float32), 1.0)
    assert module._eligible(np.full(TARGET_RATE, 1.01e-4, dtype=np.float32), 1.0)
    broken = np.ones(TARGET_RATE, dtype=np.float32)
    broken[0] = np.nan
    assert not module._eligible(broken, 1.0)


def test_reference_centroid_is_normalized_and_coverage_is_explicit(monkeypatch):
    from ayase.modules.voice_identity_drift import TARGET_RATE, VoiceIdentityDriftModule

    module = VoiceIdentityDriftModule()
    references = [Path("a.wav"), Path("b.wav"), Path("missing.wav")]
    monkeypatch.setattr(
        module,
        "_load_waveform",
        lambda path: None
        if path.name == "missing.wav"
        else np.ones(TARGET_RATE, dtype=np.float32),
    )
    embeddings = iter([np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])])
    monkeypatch.setattr(module, "_encode_waveform", lambda _waveform: next(embeddings))

    target, coverage = module._reference_target(references)

    assert coverage == pytest.approx(2 / 3)
    assert target == pytest.approx(np.asarray([2**-0.5, 2**-0.5]))
    assert np.linalg.norm(target) == pytest.approx(1.0)


def test_summary_preserves_missing_windows_tail_run_and_signed_cosines():
    from ayase.modules.voice_identity_drift import VoiceIdentityDriftModule

    module = VoiceIdentityDriftModule({"similarity_threshold": 0.3})
    series = [(0, 0.9), (1, 0.8), (2, 0.2), (3, 0.1), (5, 0.7), (6, -1.4), (7, 1.2)]
    result = module._summarize_series(series, scheduled_window_count=8)
    clipped = np.asarray([0.9, 0.8, 0.2, 0.1, 0.7, -1.0, 1.0])
    positions = np.asarray([0, 1, 2, 3, 5, 6, 7], dtype=float) / 7.0

    assert result["voice_identity_window_coverage"] == pytest.approx(7 / 8)
    assert result["voice_identity_similarity_p05"] == pytest.approx(
        np.percentile(clipped, 5.0, method="linear")
    )
    assert result["voice_identity_similarity_min"] == -1.0
    assert result["voice_identity_below_threshold_fraction"] == pytest.approx(3 / 7)
    # Missing window 4 breaks the two-window low run before low window 6.
    assert result["voice_identity_longest_below_threshold_run_fraction"] == pytest.approx(2 / 8)
    assert result["voice_identity_drift_slope"] == pytest.approx(
        np.polyfit(positions, clipped, 1)[0]
    )


def test_threshold_fields_are_unset_without_operating_point():
    from ayase.modules.voice_identity_drift import VoiceIdentityDriftModule

    result = VoiceIdentityDriftModule()._summarize_series(
        [(0, 0.9), (1, 0.8), (2, 0.7), (3, 0.6)], 4
    )

    assert result["voice_identity_below_threshold_fraction"] is None
    assert result["voice_identity_longest_below_threshold_run_fraction"] is None
    assert result["voice_identity_drift_slope"] == pytest.approx(-0.3)


def test_slope_requires_four_valid_windows_and_zero_coverage_is_observable():
    from ayase.modules.voice_identity_drift import VoiceIdentityDriftModule

    module = VoiceIdentityDriftModule({"similarity_threshold": 0.5})
    short = module._summarize_series([(0, 0.9), (2, 0.7), (3, 0.6)], 4)
    empty = module._summarize_series([], 4)

    assert short["voice_identity_drift_slope"] is None
    assert empty["voice_identity_window_coverage"] == 0.0
    assert all(
        value is None
        for field, value in empty.items()
        if field != "voice_identity_window_coverage"
    )


def test_process_uses_reference_centroid_and_records_exact_tail_metrics(
    monkeypatch, tmp_path
):
    import ayase.modules.voice_identity_drift as drift

    references = tmp_path / "refs"
    references.mkdir()
    (references / "a.wav").touch()
    (references / "b.wav").touch()
    candidate = tmp_path / "candidate.wav"
    candidate.touch()

    module = drift.VoiceIdentityDriftModule({"similarity_threshold": 0.5})
    module._encoder = object()
    module._backend = "test-ecapa"
    audio = np.ones(6 * drift.TARGET_RATE, dtype=np.float32)
    monkeypatch.setattr(drift, "load_audio", lambda *_args, **_kwargs: audio.copy())
    # Two references, then four candidate windows (the last is 1.5 seconds).
    embeddings = iter(
        [
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([1.0, 0.0]),
            np.asarray([0.8, 0.6]),
            np.asarray([0.0, 1.0]),
            np.asarray([-1.0, 0.0]),
        ]
    )
    monkeypatch.setattr(module, "_encode_waveform", lambda _waveform: next(embeddings))
    sample = Sample(path=candidate, is_video=False, reference_path=references)

    assert module.process(sample) is sample
    metrics = sample.quality_metrics
    assert metrics.voice_identity_reference_coverage == 1.0
    assert metrics.voice_identity_window_coverage == 1.0
    assert metrics.voice_identity_similarity_p05 == pytest.approx(-0.85)
    assert metrics.voice_identity_similarity_min == -1.0
    assert metrics.voice_identity_below_threshold_fraction == 0.5
    assert metrics.voice_identity_longest_below_threshold_run_fraction == 0.5
    assert metrics.voice_identity_drift_slope == pytest.approx(-2.04)
    for field in drift.VoiceIdentityDriftModule.metric_info:
        assert metrics.metric_backends[field] == "test-ecapa"


def test_failed_window_is_missing_and_breaks_threshold_run(monkeypatch, tmp_path):
    import ayase.modules.voice_identity_drift as drift

    reference = tmp_path / "reference.wav"
    reference.touch()
    candidate = tmp_path / "candidate.wav"
    candidate.touch()
    module = drift.VoiceIdentityDriftModule({"similarity_threshold": 0.5})
    module._encoder = object()
    module._backend = "test-ecapa"
    audio = np.ones(6 * drift.TARGET_RATE, dtype=np.float32)
    monkeypatch.setattr(drift, "load_audio", lambda *_args, **_kwargs: audio.copy())
    embeddings = iter(
        [
            np.asarray([1.0, 0.0]),
            np.asarray([0.0, 1.0]),
            None,
            np.asarray([0.0, 1.0]),
            np.asarray([1.0, 0.0]),
        ]
    )
    monkeypatch.setattr(module, "_encode_waveform", lambda _waveform: next(embeddings))
    sample = Sample(path=candidate, is_video=False, reference_path=reference)

    assert module.process(sample) is sample
    metrics = sample.quality_metrics
    assert metrics.voice_identity_window_coverage == 0.75
    assert metrics.voice_identity_below_threshold_fraction == pytest.approx(2 / 3)
    assert metrics.voice_identity_longest_below_threshold_run_fraction == 0.25
    assert metrics.voice_identity_drift_slope is None


def test_process_returns_same_sample_on_decode_or_backend_failure(monkeypatch, tmp_path):
    import ayase.modules.voice_identity_drift as drift

    reference = tmp_path / "reference.wav"
    reference.touch()
    candidate = tmp_path / "candidate.wav"
    candidate.touch()
    module = drift.VoiceIdentityDriftModule()

    sample = Sample(path=candidate, is_video=False, reference_path=reference)
    assert module.process(sample) is sample
    assert sample.quality_metrics is None

    module._encoder = object()
    monkeypatch.setattr(drift, "load_audio", lambda *_args, **_kwargs: None)
    assert module.process(sample) is sample
    # Reference coverage remains observable even when no file can be decoded.
    assert sample.quality_metrics.voice_identity_reference_coverage == 0.0
    assert sample.quality_metrics.voice_identity_window_coverage is None

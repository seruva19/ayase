"""Focused tests for exact paired-speech pause-rhythm diagnostics."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from ayase.models import Sample
from ..conftest import _test_module_basics


def test_speech_pause_rhythm_basics_and_metadata():
    from ayase.modules.speech_pause_rhythm import SpeechPauseRhythmModule

    _test_module_basics(SpeechPauseRhythmModule, "speech_pause_rhythm")
    expected = {
        "speech_span_duration_ratio",
        "speech_activity_fraction_difference",
        "speech_pause_count_difference",
        "speech_pause_duration_wasserstein_ms",
        "speech_activity_pattern_disagreement",
    }
    assert set(SpeechPauseRhythmModule.metric_groups) == expected
    assert set(SpeechPauseRhythmModule.metric_info) == expected
    assert SpeechPauseRhythmModule.models == [
        {
            "id": "silero-vad",
            "type": "pip_package",
            "install": "pip install silero-vad",
            "task": "Speech activity timestamps for paired pause-rhythm diagnostics",
            "url": "https://github.com/snakers4/silero-vad",
            "notes": "Official package; MIT license",
        }
    ]
    metadata = SpeechPauseRhythmModule.get_metadata()
    assert metadata["input_type"] == "audio +ref"
    assert set(metadata["output_fields"]) == expected


def test_setup_uses_official_package_and_records_distribution_version(monkeypatch):
    import ayase.modules.speech_pause_rhythm as implementation
    from ayase.modules.speech_pause_rhythm import SpeechPauseRhythmModule

    torch_module = ModuleType("torch")
    vad_module = ModuleType("silero_vad")
    model = object()
    detector = object()
    vad_module.load_silero_vad = lambda: model
    vad_module.get_speech_timestamps = detector
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "silero_vad", vad_module)
    monkeypatch.setattr(implementation, "version", lambda name: "6.2.0")

    module = SpeechPauseRhythmModule()
    module.setup()

    assert module._torch is torch_module
    assert module._vad_model is model
    assert module._get_speech_timestamps is detector
    assert module._backend == "silero_vad:6.2.0"


@pytest.mark.parametrize(
    "timestamps,sample_count",
    [
        (None, 100),
        ([], 100),
        ({"start": 0, "end": 10}, 100),
        ([{"start": 0, "end": 10}], 0),
        ([{"start": True, "end": 10}], 100),
        ([{"start": 0.0, "end": 10}], 100),
        ([{"start": -1, "end": 10}], 100),
        ([{"start": 10, "end": 10}], 100),
        ([{"start": 0, "end": 101}], 100),
        ([{"start": 20, "end": 30}, {"start": 10, "end": 15}], 100),
        ([{"start": 10, "end": 30}, {"start": 20, "end": 40}], 100),
        ([{"start": 0}], 100),
        (["not-a-mapping"], 100),
    ],
)
def test_timestamp_validation_rejects_malformed_sequences(timestamps, sample_count):
    from ayase.modules.speech_pause_rhythm import _speech_activity

    assert _speech_activity(timestamps, sample_count) is None


def test_activity_uses_half_open_intervals_and_excludes_outer_silence():
    from ayase.modules.speech_pause_rhythm import _speech_activity

    # Touching half-open intervals form one union component, not a zero pause.
    activity = _speech_activity(
        [
            {"start": 100, "end": 200},
            {"start": 200, "end": 300},
            {"start": 500, "end": 700},
        ],
        sample_count=1000,
    )

    assert activity is not None
    assert activity.span_samples == 600
    assert activity.pause_durations_samples == (200,)
    assert activity.speech_fraction == pytest.approx(2 / 3)
    assert np.allclose(activity.normalized_intervals, ((0.0, 1 / 3), (2 / 3, 1.0)))


def test_comparison_reports_exact_interval_and_pause_distribution_metrics():
    from ayase.modules.speech_pause_rhythm import _compare_activity, _speech_activity

    reference = _speech_activity(
        [
            {"start": 100, "end": 300},
            {"start": 500, "end": 700},
            {"start": 900, "end": 1100},
        ],
        sample_count=1400,
    )
    candidate = _speech_activity(
        [
            {"start": 200, "end": 500},
            {"start": 700, "end": 900},
            {"start": 1200, "end": 1400},
        ],
        sample_count=1600,
    )

    comparison = _compare_activity(reference, candidate)

    assert comparison is not None
    assert comparison.span_duration_ratio == 1.2
    assert comparison.activity_fraction_difference == pytest.approx(1 / 60)
    assert comparison.pause_count_difference == 0.0
    # Reference pauses: 200, 200 samples; candidate: 200, 300 samples.
    # Their empirical W1 distance is 50 samples at 16 kHz = 3.125 ms.
    assert comparison.pause_duration_wasserstein_ms == 3.125

    # Independent span normalization gives unions
    # ref=[0,.2] U [.4,.6] U [.8,1]
    # cand=[0,.25] U [5/12,7/12] U [5/6,1].
    assert comparison.activity_pattern_disagreement == pytest.approx(7 / 60)


def test_pattern_disagreement_is_exact_not_rasterized():
    from ayase.modules.speech_pause_rhythm import _symmetric_difference_measure

    disagreement = _symmetric_difference_measure(
        ((0.0, 1 / 3), (2 / 3, 1.0)),
        ((0.0, 1 / 2), (2 / 3, 1.0)),
    )

    assert disagreement == pytest.approx(1 / 6)


def test_wasserstein_is_exact_for_unequal_empirical_sample_counts():
    from ayase.modules.speech_pause_rhythm import _empirical_wasserstein

    # scipy.stats.wasserstein_distance([0, 4], [1, 2, 9]) == 8/3.
    assert _empirical_wasserstein((0, 4), (1, 2, 9)) == pytest.approx(8 / 3)
    assert _empirical_wasserstein((), (1,)) is None
    assert _empirical_wasserstein((1,), ()) is None


def test_process_uses_fixed_official_vad_protocol_and_sets_metrics(
    monkeypatch, tmp_path
):
    import ayase.modules.speech_pause_rhythm as implementation
    from ayase.modules.speech_pause_rhythm import SpeechPauseRhythmModule

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )

    reference_audio = np.ones(16000, dtype=np.float32)
    candidate_audio = np.ones(19200, dtype=np.float32)
    load_calls = []

    def fake_load(path, **kwargs):
        load_calls.append((path, kwargs))
        return reference_audio if path == reference_path else candidate_audio

    monkeypatch.setattr(implementation, "load_audio", fake_load)
    vad_calls = []

    def fake_vad(waveform, model, **kwargs):
        vad_calls.append((waveform, model, kwargs))
        if waveform.size == reference_audio.size:
            return [
                {"start": 1600, "end": 4800},
                {"start": 8000, "end": 14400},
            ]
        return [
            {"start": 800, "end": 5600},
            {"start": 10400, "end": 17600},
        ]

    module = SpeechPauseRhythmModule()
    module._torch = SimpleNamespace(from_numpy=lambda values: values)
    module._vad_model = object()
    module._get_speech_timestamps = fake_vad
    module._backend = "silero_vad:test-version"

    result = module.process(sample)

    assert result is sample
    assert load_calls == [
        (reference_path, {"target_sr": 16000, "mono": True}),
        (candidate_path, {"target_sr": 16000, "mono": True}),
    ]
    assert len(vad_calls) == 2
    for _waveform, model, kwargs in vad_calls:
        assert model is module._vad_model
        assert kwargs == {
            "sampling_rate": 16000,
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 0,
            "return_seconds": False,
        }

    metrics = result.quality_metrics
    assert metrics.speech_span_duration_ratio == 1.3125
    assert metrics.speech_activity_fraction_difference == pytest.approx(1 / 28)
    assert metrics.speech_pause_count_difference == 0.0
    assert metrics.speech_pause_duration_wasserstein_ms == 100.0
    assert metrics.speech_activity_pattern_disagreement == pytest.approx(3 / 28)
    for field in SpeechPauseRhythmModule._metric_fields:
        assert metrics.metric_backends[field] == "silero_vad:test-version"


def test_process_leaves_wasserstein_unset_when_either_side_has_no_pause(
    monkeypatch, tmp_path
):
    import ayase.modules.speech_pause_rhythm as implementation
    from ayase.modules.speech_pause_rhythm import SpeechPauseRhythmModule

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )
    monkeypatch.setattr(
        implementation,
        "load_audio",
        lambda *_args, **_kwargs: np.ones(16000, dtype=np.float32),
    )
    outputs = iter(
        [
            [{"start": 1000, "end": 15000}],
            [
                {"start": 1000, "end": 7000},
                {"start": 9000, "end": 15000},
            ],
        ]
    )
    module = SpeechPauseRhythmModule()
    module._torch = SimpleNamespace(from_numpy=lambda values: values)
    module._vad_model = object()
    module._get_speech_timestamps = lambda *_args, **_kwargs: next(outputs)
    module._backend = "silero_vad:test-version"

    module.process(sample)

    assert sample.quality_metrics.speech_pause_duration_wasserstein_ms is None
    assert "speech_pause_duration_wasserstein_ms" not in sample.quality_metrics.metric_backends


@pytest.mark.parametrize(
    "bad_audio",
    [
        None,
        np.array([], dtype=np.float32),
        np.zeros((10, 2), dtype=np.float32),
        np.array([0.0, np.nan], dtype=np.float32),
        np.zeros(16000 * 30 + 1, dtype=np.float32),
    ],
)
def test_invalid_or_overlong_audio_is_rejected_without_vad(
    monkeypatch, tmp_path, bad_audio
):
    import ayase.modules.speech_pause_rhythm as implementation
    from ayase.modules.speech_pause_rhythm import SpeechPauseRhythmModule

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )
    monkeypatch.setattr(implementation, "load_audio", lambda *_args, **_kwargs: bad_audio)
    module = SpeechPauseRhythmModule()
    module._torch = SimpleNamespace(from_numpy=lambda values: values)
    module._vad_model = object()
    module._get_speech_timestamps = lambda *_args, **_kwargs: pytest.fail(
        "VAD must not run for invalid audio"
    )
    module._backend = "silero_vad:test-version"

    assert module.process(sample) is sample
    assert sample.quality_metrics is None

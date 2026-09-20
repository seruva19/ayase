"""Focused tests for the constrained DTW log-F0 metric."""

from types import SimpleNamespace

import numpy as np

from ayase.models import Sample
from ..conftest import _test_module_basics


def _features(f0, voiced, frames=None):
    from ayase.modules.audio_log_f0_dtw import _PitchFeatures

    f0 = np.asarray(f0, dtype=np.float64)
    voiced = np.asarray(voiced, dtype=bool)
    if frames is None:
        frames = f0.size
    axis = np.linspace(-1.0, 1.0, frames, dtype=np.float64)
    mfcc = np.vstack([(index + 1) * axis for index in range(12)])
    return _PitchFeatures(f0=f0, voiced=voiced, mfcc=mfcc)


def _module_with_diagonal_dtw(call_log=None):
    from ayase.modules.audio_log_f0_dtw import AudioLogF0DTWModule

    def dtw(**kwargs):
        if call_log is not None:
            call_log.append(kwargs)
        count = min(kwargs["X"].shape[1], kwargs["Y"].shape[1])
        path = np.column_stack([np.arange(count), np.arange(count)])[::-1]
        return np.zeros((kwargs["X"].shape[1], kwargs["Y"].shape[1])), path

    module = AudioLogF0DTWModule()
    module._librosa = SimpleNamespace(sequence=SimpleNamespace(dtw=dtw))
    module._backend = "librosa_pyin"
    return module


def test_audio_log_f0_dtw_basics():
    from ayase.modules.audio_log_f0_dtw import AudioLogF0DTWModule

    _test_module_basics(AudioLogF0DTWModule, "audio_log_f0_dtw")
    assert set(AudioLogF0DTWModule.metric_groups) == {
        "audio_log_f0_rmse_cents",
        "audio_f0_voicing_error",
        "audio_f0_joint_coverage",
    }


def test_identical_contours_are_zero_and_use_constrained_mfcc_dtw():
    calls = []
    module = _module_with_diagonal_dtw(calls)
    track = _features(np.full(40, 220.0), np.ones(40, dtype=bool))

    result = module._align_and_score(track, track)

    assert result == (0.0, 0.0, 1.0)
    assert len(calls) == 1
    call = calls[0]
    assert set(call) == {
        "X",
        "Y",
        "metric",
        "subseq",
        "backtrack",
        "global_constraints",
        "band_rad",
    }
    assert call["X"] is track.mfcc
    assert call["Y"] is track.mfcc
    assert call["metric"] == "euclidean"
    assert call["subseq"] is False
    assert call["backtrack"] is True
    assert call["global_constraints"] is True
    assert call["band_rad"] == 0.10


def test_octave_shift_is_1200_cents():
    module = _module_with_diagonal_dtw()
    reference = _features(np.full(40, 220.0), np.ones(40, dtype=bool))
    candidate = _features(np.full(40, 440.0), np.ones(40, dtype=bool))

    rmse, voicing_error, coverage = module._align_and_score(reference, candidate)

    assert rmse == 1200.0
    assert voicing_error == 0.0
    assert coverage == 1.0


def test_voicing_diagnostics_and_joint_coverage_are_path_based():
    module = _module_with_diagonal_dtw()
    reference = _features(np.full(30, 220.0), np.ones(30, dtype=bool))
    candidate_voiced = np.ones(30, dtype=bool)
    candidate_voiced[10:20] = False
    candidate_f0 = np.full(30, 440.0)
    candidate_f0[~candidate_voiced] = np.nan
    candidate = _features(candidate_f0, candidate_voiced)

    rmse, voicing_error, coverage = module._align_and_score(reference, candidate)

    assert rmse == 1200.0
    assert voicing_error == 10 / 30
    assert coverage == 20 / 30


def test_low_unique_joint_coverage_leaves_rmse_unset():
    module = _module_with_diagonal_dtw()
    reference = _features(np.full(60, 220.0), np.ones(60, dtype=bool))
    candidate_voiced = np.zeros(60, dtype=bool)
    candidate_voiced[:10] = True
    candidate_voiced[-10:] = True
    candidate_f0 = np.full(60, np.nan)
    candidate_f0[candidate_voiced] = 440.0
    candidate = _features(candidate_f0, candidate_voiced)

    rmse, voicing_error, coverage = module._align_and_score(reference, candidate)

    assert rmse is None
    assert voicing_error == 40 / 60
    assert coverage == 20 / 60


def test_fewer_than_twenty_voiced_frames_is_no_result():
    module = _module_with_diagonal_dtw()
    reference = _features(np.full(30, 220.0), np.ones(30, dtype=bool))
    candidate_voiced = np.zeros(30, dtype=bool)
    candidate_voiced[:19] = True
    candidate = _features(np.full(30, 220.0), candidate_voiced)

    assert module._align_and_score(reference, candidate) is None


def test_extract_features_uses_fixed_pyin_grid_trims_edges_and_cmn():
    from ayase.modules.audio_log_f0_dtw import AudioLogF0DTWModule

    calls = {}
    f0 = np.full(25, np.nan)
    f0[2:23] = 220.0
    voiced = np.isfinite(f0)
    raw_mfcc = np.arange(13 * 25, dtype=np.float64).reshape(13, 25)

    def pyin(audio, **kwargs):
        calls["pyin"] = (audio, kwargs)
        return f0, voiced, voiced.astype(float)

    def mfcc(**kwargs):
        calls["mfcc"] = kwargs
        return raw_mfcc

    module = AudioLogF0DTWModule()
    module._librosa = SimpleNamespace(
        pyin=pyin,
        feature=SimpleNamespace(mfcc=mfcc),
    )

    result = module._extract_features(np.ones(16000, dtype=np.float32))

    assert result is not None
    assert result.f0.shape == (21,)
    assert result.voiced.all()
    assert result.mfcc.shape == (12, 21)
    assert np.allclose(np.mean(result.mfcc, axis=1), 0.0)
    pyin_kwargs = calls["pyin"][1]
    assert pyin_kwargs == {
        "fmin": 50.0,
        "fmax": 800.0,
        "sr": 16000,
        "frame_length": 2048,
        "hop_length": 160,
        "center": True,
        "fill_na": np.nan,
    }
    mfcc_kwargs = calls["mfcc"]
    assert mfcc_kwargs["sr"] == 16000
    assert mfcc_kwargs["n_mfcc"] == 13
    assert mfcc_kwargs["n_fft"] == 512
    assert mfcc_kwargs["win_length"] == 400
    assert mfcc_kwargs["hop_length"] == 160
    assert mfcc_kwargs["center"] is True


def test_extract_features_fully_unvoiced_is_no_result():
    from ayase.modules.audio_log_f0_dtw import AudioLogF0DTWModule

    module = AudioLogF0DTWModule()
    module._librosa = SimpleNamespace(
        pyin=lambda *_args, **_kwargs: (
            np.full(30, np.nan),
            np.zeros(30, dtype=bool),
            np.zeros(30),
        )
    )

    assert module._extract_features(np.zeros(16000, dtype=np.float32)) is None


def test_process_populates_three_fields_for_valid_pair(monkeypatch, tmp_path):
    import ayase.modules.audio_log_f0_dtw as implementation

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )
    module = _module_with_diagonal_dtw()
    reference = _features(np.full(40, 220.0), np.ones(40, dtype=bool))
    candidate = _features(np.full(40, 440.0), np.ones(40, dtype=bool))
    tracks = iter([reference, candidate])
    monkeypatch.setattr(
        implementation,
        "load_audio",
        lambda *_args, **_kwargs: np.ones(16000, dtype=np.float32),
    )
    monkeypatch.setattr(module, "_extract_features", lambda _audio: next(tracks))

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics.audio_log_f0_rmse_cents == 1200.0
    assert result.quality_metrics.audio_f0_voicing_error == 0.0
    assert result.quality_metrics.audio_f0_joint_coverage == 1.0


def test_process_below_coverage_writes_diagnostics_only(monkeypatch, tmp_path):
    import ayase.modules.audio_log_f0_dtw as implementation

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )
    module = _module_with_diagonal_dtw()
    reference = _features(np.full(60, 220.0), np.ones(60, dtype=bool))
    candidate_voiced = np.zeros(60, dtype=bool)
    candidate_voiced[:10] = True
    candidate_voiced[-10:] = True
    candidate_f0 = np.full(60, np.nan)
    candidate_f0[candidate_voiced] = 220.0
    candidate = _features(candidate_f0, candidate_voiced)
    tracks = iter([reference, candidate])
    monkeypatch.setattr(
        implementation,
        "load_audio",
        lambda *_args, **_kwargs: np.ones(16000, dtype=np.float32),
    )
    monkeypatch.setattr(module, "_extract_features", lambda _audio: next(tracks))

    result = module.process(sample)

    assert result.quality_metrics.audio_log_f0_rmse_cents is None
    assert result.quality_metrics.audio_f0_voicing_error == 0.6667
    assert result.quality_metrics.audio_f0_joint_coverage == 0.3333


def test_missing_reference_and_overlong_audio_are_no_ops(monkeypatch, tmp_path):
    import ayase.modules.audio_log_f0_dtw as implementation

    candidate_path = tmp_path / "candidate.wav"
    reference_path = tmp_path / "reference.wav"
    candidate_path.touch()
    reference_path.touch()
    module = _module_with_diagonal_dtw()

    missing_reference = Sample(path=candidate_path, is_video=False)
    assert module.process(missing_reference) is missing_reference
    assert missing_reference.quality_metrics is None

    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )
    monkeypatch.setattr(
        implementation,
        "load_audio",
        lambda *_args, **_kwargs: np.ones(30 * 16000 + 1, dtype=np.float32),
    )
    monkeypatch.setattr(
        module,
        "_extract_features",
        lambda _audio: (_ for _ in ()).throw(AssertionError("must not extract")),
    )

    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_backend_unavailable_and_runtime_failure_degrade_gracefully(
    monkeypatch, tmp_path
):
    import ayase.modules.audio_log_f0_dtw as implementation
    from ayase.modules.audio_log_f0_dtw import AudioLogF0DTWModule

    path = tmp_path / "audio.wav"
    path.touch()
    sample = Sample(path=path, is_video=False, reference_path=path)
    unavailable = AudioLogF0DTWModule()
    assert unavailable.process(sample) is sample
    assert sample.quality_metrics is None

    module = _module_with_diagonal_dtw()
    monkeypatch.setattr(
        implementation,
        "load_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode")),
    )
    assert module.process(sample) is sample
    assert sample.quality_metrics is None

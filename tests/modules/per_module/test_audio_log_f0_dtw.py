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
    mcep = np.vstack([(index + 1) * axis for index in range(24)])
    return _PitchFeatures(f0=f0, voiced=voiced, mcep=mcep)


def _module_with_diagonal_dtw(call_log=None):
    from ayase.modules.audio_log_f0_dtw import AudioLogF0DTWModule

    def dtw(x, y):
        """fastdtw signature: (gen_frames x dims, gt_frames x dims)."""
        if call_log is not None:
            call_log.append((x, y))
        count = min(x.shape[0], y.shape[0])
        path = list(zip(np.arange(count), np.arange(count)))
        return 0.0, path

    module = AudioLogF0DTWModule()
    module._dtw = dtw
    module._backend = "espnet_f0"
    return module


def test_audio_log_f0_dtw_basics():
    from ayase.modules.audio_log_f0_dtw import AudioLogF0DTWModule

    _test_module_basics(AudioLogF0DTWModule, "audio_log_f0_dtw")
    assert set(AudioLogF0DTWModule.metric_groups) == {
        "audio_log_f0_rmse_cents",
        "audio_f0_voiced_mismatch",
    }


def test_identical_contours_are_zero_and_use_fastdtw_on_mcep():
    calls = []
    module = _module_with_diagonal_dtw(calls)
    track = _features(np.full(40, 220.0), np.ones(40, dtype=bool))

    result = module._align_and_score(track, track)

    assert result == (0.0, 0.0, 1.0)
    assert len(calls) == 1
    x, y = calls[0]
    # fastdtw is called with frames x dims mel-cepstra (candidate, reference).
    assert x.shape == (40, 24)
    assert y.shape == (40, 24)


def test_octave_shift_is_1200_cents():
    module = _module_with_diagonal_dtw()
    reference = _features(np.full(40, 220.0), np.ones(40, dtype=bool))
    candidate = _features(np.full(40, 440.0), np.ones(40, dtype=bool))

    rmse, voicing_error, coverage = module._align_and_score(reference, candidate)

    assert np.isclose(rmse, 1200.0)
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

    assert np.isclose(rmse, 1200.0)
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


def test_extract_features_uses_espnet_world_params_and_trims_edges(monkeypatch):
    from ayase.modules.audio_log_f0_dtw import AudioLogF0DTWModule

    calls = {}
    f0 = np.zeros(25, dtype=np.float64)
    f0[2:23] = 220.0
    raw_mcep = np.arange(25 * 24, dtype=np.float64).reshape(25, 24)

    def harvest(x, fs, f0_floor, f0_ceil, frame_period):
        calls["harvest"] = (x, fs, f0_floor, f0_ceil, frame_period)
        return f0, np.arange(25)

    def cheaptrick(x, f0v, time_axis, fs, fft_size):
        calls["cheaptrick"] = fft_size
        return np.zeros((25, 513))

    def sp2mc(sp, dim, alpha):
        calls["sp2mc"] = (dim, alpha)
        return raw_mcep

    monkeypatch.setitem(
        __import__("sys").modules,
        "pyworld",
        SimpleNamespace(harvest=harvest, cheaptrick=cheaptrick),
    )
    monkeypatch.setitem(__import__("sys").modules, "pysptk", SimpleNamespace(sp2mc=sp2mc))

    module = AudioLogF0DTWModule()
    result = module._extract_features(np.ones(16000, dtype=np.float32))

    assert result is not None
    assert result.f0.shape == (21,)
    assert result.voiced.all()
    assert result.mcep.shape == (24, 21)
    x, fs, f0_floor, f0_ceil, frame_period = calls["harvest"]
    assert fs == 16000
    assert f0_floor == 40.0
    assert f0_ceil == 800.0
    assert frame_period == 16.0
    assert calls["cheaptrick"] == 1024
    assert calls["sp2mc"] == (23, 0.42)


def test_extract_features_fully_unvoiced_is_no_result(monkeypatch):
    from ayase.modules.audio_log_f0_dtw import AudioLogF0DTWModule

    monkeypatch.setitem(
        __import__("sys").modules,
        "pyworld",
        SimpleNamespace(
            harvest=lambda *a, **k: (np.zeros(30), np.arange(30)),
        ),
    )

    module = AudioLogF0DTWModule()
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
    assert result.quality_metrics.audio_f0_voiced_mismatch == 0.0


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
    assert result.quality_metrics.audio_f0_voiced_mismatch == 0.6667


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

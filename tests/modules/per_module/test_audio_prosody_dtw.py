"""Focused tests for paired-speech relative-energy DTW diagnostics."""

from types import SimpleNamespace

import numpy as np

from ayase.models import Sample
from ..conftest import _test_module_basics


def _features(energy_db, voiced=None, mfcc=None):
    from ayase.modules.audio_prosody_dtw import _ProsodyFeatures

    energy_db = np.asarray(energy_db, dtype=np.float64)
    if voiced is None:
        voiced = np.ones(energy_db.size, dtype=bool)
    if mfcc is None:
        axis = np.linspace(-1.0, 1.0, energy_db.size, dtype=np.float64)
        mfcc = np.vstack([(index + 1) * axis for index in range(12)])
    return _ProsodyFeatures(
        energy_db=energy_db,
        voiced=np.asarray(voiced, dtype=bool),
        mfcc=np.asarray(mfcc, dtype=np.float64),
    )


def _module_with_path(path_factory=None, call_log=None):
    from ayase.modules.audio_prosody_dtw import AudioProsodyDTWModule

    def dtw(**kwargs):
        if call_log is not None:
            call_log.append(kwargs)
        if path_factory is None:
            count = min(kwargs["X"].shape[1], kwargs["Y"].shape[1])
            path = np.column_stack([np.arange(count), np.arange(count)])
        else:
            path = np.asarray(path_factory(kwargs), dtype=np.int64)
        # librosa returns the path from the end point to the start point.
        return np.zeros((kwargs["X"].shape[1], kwargs["Y"].shape[1])), path[::-1]

    module = AudioProsodyDTWModule()
    module._librosa = SimpleNamespace(sequence=SimpleNamespace(dtw=dtw))
    module._backend = "librosa_pyin_dtw"
    return module


def test_audio_prosody_dtw_basics_and_metadata():
    from ayase.modules.audio_prosody_dtw import AudioProsodyDTWModule

    _test_module_basics(AudioProsodyDTWModule, "audio_prosody_dtw")
    expected = {
        "audio_relative_energy_rmse_db",
        "audio_energy_contour_correlation",
        "audio_voiced_fraction_difference",
        "audio_duration_ratio",
        "audio_prosody_warp_ratio",
    }
    assert set(AudioProsodyDTWModule.metric_groups) == expected
    assert set(AudioProsodyDTWModule.metric_info) == expected
    metadata = AudioProsodyDTWModule.get_metadata()
    assert metadata["input_type"] == "audio +ref"
    assert set(metadata["output_fields"]) == expected
    assert set(metadata["metric_info"]) == expected


def test_identity_is_zero_with_unit_correlation_and_constrained_mfcc_dtw():
    calls = []
    module = _module_with_path(call_log=calls)
    energy = -30.0 + 5.0 * np.sin(np.linspace(0.0, 4.0 * np.pi, 40))
    track = _features(energy)

    result = module._align_and_score(track, track)

    assert result[0] == 0.0
    assert np.isclose(result[1], 1.0)
    assert result[2] == 1.0
    assert len(calls) == 1
    call = calls[0]
    assert call["X"] is track.mfcc
    assert call["Y"] is track.mfcc
    assert call["metric"] == "euclidean"
    assert call["subseq"] is False
    assert call["backtrack"] is True
    assert call["global_constraints"] is True
    assert call["band_rad"] == 0.10


def test_mean_centering_makes_relative_energy_gain_invariant():
    module = _module_with_path()
    reference_energy = np.linspace(-46.0, -18.0, 40)
    reference = _features(reference_energy)
    candidate = _features(reference_energy + 12.0)

    rmse, correlation, warp_ratio = module._align_and_score(reference, candidate)

    assert np.isclose(rmse, 0.0, atol=1e-12)
    assert np.isclose(correlation, 1.0)
    assert warp_ratio == 1.0


def test_inverted_contour_reports_shape_change_without_magic_aggregate():
    module = _module_with_path()
    energy = np.linspace(-8.0, 8.0, 40)
    reference = _features(energy)
    candidate = _features(-energy)

    rmse, correlation, warp_ratio = module._align_and_score(reference, candidate)

    assert rmse > 0.0
    assert np.isclose(correlation, -1.0)
    assert warp_ratio == 1.0


def test_time_stretch_dtw_preserves_shape_and_duration_ratio_direction(
    monkeypatch, tmp_path
):
    import ayase.modules.audio_prosody_dtw as implementation

    def stretched_path(kwargs):
        # Every candidate frame is assigned to its source reference frame.
        candidate_indices = np.arange(kwargs["Y"].shape[1])
        reference_indices = candidate_indices // 2
        return np.column_stack([reference_indices, candidate_indices])

    module = _module_with_path(stretched_path)
    reference_energy = np.linspace(-50.0, -10.0, 30)
    reference_features = _features(reference_energy)
    candidate_features = _features(np.repeat(reference_energy, 2))
    tracks = iter([reference_features, candidate_features])
    monkeypatch.setattr(module, "_extract_features", lambda _audio: next(tracks))

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )

    def load(path, **_kwargs):
        size = 16000 if path == reference_path else 24000
        return np.ones(size, dtype=np.float32)

    monkeypatch.setattr(implementation, "load_audio", load)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics.audio_duration_ratio == 1.5
    assert result.quality_metrics.audio_relative_energy_rmse_db == 0.0
    assert result.quality_metrics.audio_energy_contour_correlation == 1.0
    assert result.quality_metrics.audio_prosody_warp_ratio == 0.5


def test_voiced_fraction_difference_uses_whole_trimmed_span(monkeypatch, tmp_path):
    import ayase.modules.audio_prosody_dtw as implementation

    module = _module_with_path()
    reference = _features(np.linspace(-30.0, -10.0, 40))
    candidate_voiced = np.zeros(40, dtype=bool)
    candidate_voiced[:19] = True
    candidate_voiced[-1] = True
    candidate = _features(
        np.linspace(-30.0, -10.0, 40), voiced=candidate_voiced
    )
    tracks = iter([reference, candidate])
    monkeypatch.setattr(module, "_extract_features", lambda _audio: next(tracks))
    monkeypatch.setattr(
        implementation,
        "load_audio",
        lambda *_args, **_kwargs: np.ones(16000, dtype=np.float32),
    )

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )

    module.process(sample)

    assert result_value(sample, "audio_voiced_fraction_difference") == 0.5


def test_zero_variance_leaves_correlation_unset_but_keeps_rmse():
    module = _module_with_path()
    reference = _features(np.full(40, -24.0))
    candidate = _features(np.full(40, -12.0))

    rmse, correlation, warp_ratio = module._align_and_score(reference, candidate)

    assert rmse == 0.0
    assert correlation is None
    assert warp_ratio == 1.0


def test_excessive_repeated_frame_warping_leaves_contour_outputs_unset():
    def heavily_warped_path(_kwargs):
        reference_indices = np.arange(60)
        candidate_indices = reference_indices // 3
        return np.column_stack([reference_indices, candidate_indices])

    module = _module_with_path(heavily_warped_path)
    reference = _features(np.linspace(-40.0, -10.0, 60))
    candidate = _features(np.linspace(-40.0, -10.0, 20))

    rmse, correlation, warp_ratio = module._align_and_score(reference, candidate)

    assert rmse is None
    assert correlation is None
    assert warp_ratio == 1 / 3


def test_extract_features_uses_fixed_grid_floor_trim_and_mfcc_cmn():
    from ayase.modules.audio_prosody_dtw import AudioProsodyDTWModule

    calls = {}
    f0 = np.full(25, np.nan)
    f0[2:23] = 220.0
    voiced = np.isfinite(f0)
    rms_values = np.linspace(0.0, 0.1, 25)[None, :]
    rms_values[0, 2] = 0.0
    raw_mfcc = np.arange(13 * 25, dtype=np.float64).reshape(13, 25)

    def pyin(audio, **kwargs):
        calls["pyin"] = (audio, kwargs)
        return f0, voiced, voiced.astype(float)

    def rms(**kwargs):
        calls["rms"] = kwargs
        return rms_values

    def mfcc(**kwargs):
        calls["mfcc"] = kwargs
        return raw_mfcc

    module = AudioProsodyDTWModule()
    module._librosa = SimpleNamespace(
        pyin=pyin,
        feature=SimpleNamespace(rms=rms, mfcc=mfcc),
    )

    result = module._extract_features(np.ones(16000, dtype=np.float32))

    assert result is not None
    assert result.energy_db.shape == (21,)
    assert result.voiced.all()
    assert result.mfcc.shape == (12, 21)
    assert np.allclose(np.mean(result.mfcc, axis=1), 0.0)
    assert result.energy_db[0] == -80.0
    assert np.min(result.energy_db) >= -80.0
    assert calls["pyin"][1] == {
        "fmin": 50.0,
        "fmax": 800.0,
        "sr": 16000,
        "frame_length": 2048,
        "hop_length": 160,
        "center": True,
        "fill_na": np.nan,
    }
    assert calls["rms"]["frame_length"] == 400
    assert calls["rms"]["hop_length"] == 160
    assert calls["rms"]["center"] is True
    assert calls["mfcc"]["n_mfcc"] == 13
    assert calls["mfcc"]["n_fft"] == 512
    assert calls["mfcc"]["win_length"] == 400
    assert calls["mfcc"]["hop_length"] == 160


def test_fully_unvoiced_and_fewer_than_twenty_voiced_frames_are_rejected():
    from ayase.modules.audio_prosody_dtw import AudioProsodyDTWModule

    def feature_namespace(voiced_count):
        f0 = np.full(30, np.nan)
        f0[:voiced_count] = 220.0
        voiced = np.isfinite(f0)
        return SimpleNamespace(
            pyin=lambda *_args, **_kwargs: (f0, voiced, voiced.astype(float)),
            feature=SimpleNamespace(
                rms=lambda **_kwargs: np.ones((1, 30)),
                mfcc=lambda **_kwargs: np.ones((13, 30)),
            ),
        )

    module = AudioProsodyDTWModule()
    module._librosa = feature_namespace(0)
    assert module._extract_features(np.ones(16000, dtype=np.float32)) is None
    module._librosa = feature_namespace(19)
    assert module._extract_features(np.ones(16000, dtype=np.float32)) is None


def test_duration_survives_feature_rejection_and_no_op_error_paths(
    monkeypatch, tmp_path
):
    import ayase.modules.audio_prosody_dtw as implementation
    from ayase.modules.audio_prosody_dtw import AudioProsodyDTWModule

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()

    module = _module_with_path()
    monkeypatch.setattr(
        implementation,
        "load_audio",
        lambda path, **_kwargs: np.ones(
            16000 if path == reference_path else 8000, dtype=np.float32
        ),
    )
    monkeypatch.setattr(module, "_extract_features", lambda _audio: None)
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )

    assert module.process(sample) is sample
    assert result_value(sample, "audio_duration_ratio") == 0.5
    assert result_value(sample, "audio_voiced_fraction_difference") is None

    missing_reference = Sample(path=candidate_path, is_video=False)
    assert module.process(missing_reference) is missing_reference
    assert missing_reference.quality_metrics is None

    unavailable = AudioProsodyDTWModule()
    untouched = Sample(
        path=candidate_path, is_video=False, reference_path=reference_path
    )
    assert unavailable.process(untouched) is untouched
    assert untouched.quality_metrics is None

    monkeypatch.setattr(
        implementation,
        "load_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("decode")),
    )
    failed = Sample(
        path=candidate_path, is_video=False, reference_path=reference_path
    )
    assert module.process(failed) is failed
    assert failed.quality_metrics is None


def test_overlong_non_mono_and_nonfinite_audio_are_no_ops(monkeypatch, tmp_path):
    import ayase.modules.audio_prosody_dtw as implementation

    reference_path = tmp_path / "reference.wav"
    candidate_path = tmp_path / "candidate.wav"
    reference_path.touch()
    candidate_path.touch()
    sample = Sample(
        path=candidate_path,
        is_video=False,
        reference_path=reference_path,
    )
    module = _module_with_path()

    invalid_inputs = [
        np.ones(30 * 16000 + 1, dtype=np.float32),
        np.ones((100, 2), dtype=np.float32),
        np.array([np.nan], dtype=np.float32),
    ]
    for invalid in invalid_inputs:
        sample.quality_metrics = None
        monkeypatch.setattr(
            implementation,
            "load_audio",
            lambda *_args, value=invalid, **_kwargs: value,
        )
        assert module.process(sample) is sample
        assert sample.quality_metrics is None


def result_value(sample, field):
    """Read a metric while keeping assertions concise under Pydantic."""

    if sample.quality_metrics is None:
        return None
    return getattr(sample.quality_metrics, field)

"""Tests for lip_sync module."""

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_lip_sync_basics():
    from ayase.modules.lip_sync import LipSyncSyncNetModule, LipSyncVerseModule

    _test_module_basics(LipSyncVerseModule, "lip_sync_verse")
    _test_module_basics(LipSyncSyncNetModule, "lip_sync_syncnet")


def test_canonical_metadata_reports_only_its_own_output_pair():
    from ayase.modules.lip_sync import LipSyncSyncNetModule, LipSyncVerseModule

    assert set(LipSyncVerseModule.get_metadata()["output_fields"]) == {
        "lse_c_verse",
        "lse_d_verse",
    }
    assert set(LipSyncSyncNetModule.get_metadata()["output_fields"]) == {
        "lse_c_syncnet",
        "lse_d_syncnet",
    }


def test_lip_sync_video(video_sample):
    from ayase.modules.lip_sync import LipSyncModule

    video_sample.quality_metrics = QualityMetrics()
    m = LipSyncModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


def test_lip_sync_default_protocol_is_verse_bench():
    from ayase.modules.lip_sync import LipSyncModule, LipSyncVerseModule

    m = LipSyncModule({"test_mode": True, "protocol": "verse_bench"})
    assert isinstance(m, LipSyncVerseModule)
    assert "protocol" not in m.config
    assert m._requested_module_name == "lip_sync"
    assert m._legacy_lip_sync_protocol == "verse_bench"
    assert m._legacy_output_aliases == {"lse_c_verse": "lse_c", "lse_d_verse": "lse_d"}


def test_wav2lip_protocol_is_opt_in():
    from ayase.modules.lip_sync import LipSyncModule, LipSyncSyncNetModule

    m = LipSyncModule({"protocol": "wav2lip", "test_mode": True})
    assert isinstance(m, LipSyncSyncNetModule)
    assert "protocol" not in m.config
    assert m._legacy_output_aliases == {"lse_c_syncnet": "lse_c", "lse_d_syncnet": "lse_d"}


def test_lip_sync_unknown_protocol_is_rejected():
    from ayase.modules.lip_sync import LipSyncModule

    with pytest.raises(ValueError, match="verse_bench.*wav2lip"):
        LipSyncModule({"protocol": "nope"})


def test_legacy_metadata_delegates_to_default_canonical_module():
    from ayase.modules.lip_sync import LipSyncModule, LipSyncVerseModule

    assert LipSyncModule.get_metadata() == LipSyncVerseModule.get_metadata()
    assert LipSyncModule.get_metadata()["name"] == "lip_sync_verse"


def test_checkpoint_identities_are_declared_in_model_metadata():
    from ayase.modules.lip_sync import LipSyncSyncNetModule, LipSyncVerseModule

    verse_models = {model["id"]: model for model in LipSyncVerseModule.models}
    syncnet_models = {model["id"]: model for model in LipSyncSyncNetModule.models}
    assert verse_models["syncnet_v2.model"]["sha256"] == (
        "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
    )
    assert syncnet_models["sfd_face.pth"]["sha256"] == (
        "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491"
    )
    assert all("license is not established" in model["notes"] for model in syncnet_models.values())


def test_weight_fetch_rejects_same_size_wrong_hash(monkeypatch, tmp_path):
    from ayase.modules import lip_sync

    data = b"wrong bytes"
    path = tmp_path / "syncnet_v2.model"
    path.write_bytes(data)
    monkeypatch.setattr("ayase.config.download_model_file", lambda *args: path)
    module = lip_sync.LipSyncVerseModule({"models_dir": str(tmp_path)})

    assert module._fetch_weight(lip_sync._SYNCNET_REL, "unused", len(data)) is None


def test_weight_fetch_accepts_matching_hash(monkeypatch, tmp_path):
    from ayase.modules import lip_sync

    data = b"small checkpoint fixture"
    path = tmp_path / "syncnet_v2.model"
    path.write_bytes(data)
    monkeypatch.setattr("ayase.config.download_model_file", lambda *args: path)
    monkeypatch.setattr(lip_sync, "_SYNCNET_SHA256", hashlib.sha256(data).hexdigest())
    module = lip_sync.LipSyncVerseModule({"models_dir": str(tmp_path)})

    assert module._fetch_weight(lip_sync._SYNCNET_REL, "unused", len(data)) == path


def _without_test_mode(monkeypatch):
    from ayase.pipeline import PipelineModule

    monkeypatch.setattr(PipelineModule, "_global_test_mode", False)
    monkeypatch.delenv("AYASE_TEST_MODE", raising=False)


class _FixedTracks:
    """Stand-in SyncNet scorer returning fixed per-track scores."""

    def __init__(self, tracks):
        self.tracks = tracks
        self.calls = []

    def score_video(self, path, params, **kw):
        self.calls.append((path, params, kw))
        return self.tracks

    def close(self):
        pass


class _FixedVerse:
    def __init__(self, scores=(4.25, 7.5)):
        self.scores = scores

    def score(self, path):
        return self.scores

    def close(self):
        pass


def test_verse_module_assigns_explicit_canonical_fields(video_sample):
    from ayase.modules.lip_sync import LipSyncVerseModule

    module = LipSyncVerseModule()
    module._impl = _FixedVerse()
    module._ml_available = True
    module.process(video_sample)

    metrics = video_sample.quality_metrics
    assert metrics.lse_c_verse == 4.25
    assert metrics.lse_d_verse == 7.5
    assert "lse_c" not in type(metrics).model_fields
    assert "lse_d" not in type(metrics).model_fields
    assert metrics.non_null_metrics() == {"lse_c_verse": 4.25, "lse_d_verse": 7.5}
    canonical = metrics.canonical_model_dump(exclude_none=True)
    assert {key for key in canonical if key.startswith("lse_")} == {
        "lse_c_verse",
        "lse_d_verse",
    }


@pytest.mark.parametrize(
    ("protocol", "canonical_fields"),
    [
        ("verse_bench", ("lse_c_verse", "lse_d_verse")),
        ("wav2lip", ("lse_c_syncnet", "lse_d_syncnet")),
    ],
)
def test_legacy_facade_process_sets_canonical_values_and_legacy_json(
    video_sample, protocol, canonical_fields
):
    from ayase.modules.lip_sync import LipSyncModule

    module = LipSyncModule({"protocol": protocol})
    module._impl = _FixedVerse((2.5, 8.75))
    module._ml_available = True
    assert module.process(video_sample) is video_sample

    metrics = video_sample.quality_metrics
    assert getattr(metrics, canonical_fields[0]) == 2.5
    assert getattr(metrics, canonical_fields[1]) == 8.75
    assert metrics.non_null_metrics() == {
        canonical_fields[0]: 2.5,
        canonical_fields[1]: 8.75,
    }
    dumped = metrics.model_dump(exclude_none=True)
    assert dumped["lse_c"] == 2.5
    assert dumped["lse_d"] == 8.75
    assert canonical_fields[0] not in dumped
    assert canonical_fields[1] not in dumped
    assert {key for key in dumped if key.startswith("lse_")} == {"lse_c", "lse_d"}
    dumped_json = json.loads(metrics.model_dump_json())
    assert {key for key in dumped_json if key.startswith("lse_")} == {"lse_c", "lse_d"}


@pytest.mark.parametrize("scores", [(object(), 1.0), (1.0,), "bad"])
def test_malformed_scores_return_sample_without_metrics(video_sample, scores):
    from ayase.modules.lip_sync import LipSyncVerseModule

    module = LipSyncVerseModule()
    module._impl = _FixedVerse(scores)
    module._ml_available = True

    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_protocol_error_returns_sample_without_metrics(video_sample):
    from ayase.modules.lip_sync import LipSyncVerseModule

    class _BrokenProtocol:
        def score(self, path):
            raise ValueError("broken score")

    module = LipSyncVerseModule()
    module._impl = _BrokenProtocol()
    module._ml_available = True

    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


@pytest.mark.parametrize("protocol", ["verse_bench", "wav2lip"])
@pytest.mark.parametrize("backend_state", ["unready", "no_score"])
def test_legacy_facade_stamps_existing_metrics_when_score_is_unavailable(
    video_sample, protocol, backend_state
):
    from ayase.modules.lip_sync import LipSyncModule

    video_sample.quality_metrics = QualityMetrics(blur_score=1.0)
    module = LipSyncModule({"protocol": protocol})
    if backend_state == "no_score":
        module._impl = _FixedVerse(None)
        module._ml_available = True

    assert module.process(video_sample) is video_sample
    metrics = video_sample.quality_metrics
    assert metrics.non_null_metrics() == {"blur_score": 1.0}
    dumped = metrics.model_dump()
    assert dumped["lse_c"] is None
    assert dumped["lse_d"] is None
    assert {key for key in dumped if key.startswith("lse_")} == {"lse_c", "lse_d"}
    dumped_json = json.loads(metrics.model_dump_json())
    assert dumped_json["lse_c"] is None
    assert dumped_json["lse_d"] is None
    assert {key for key in dumped_json if key.startswith("lse_")} == {"lse_c", "lse_d"}


def _wav2lip_module(tracks):
    from ayase.modules.lip_sync import LipSyncSyncNetModule, Wav2LipProtocol
    from ayase.vendor.syncnet_python.face_tracks import FaceTrackParams

    m = LipSyncSyncNetModule({"min_face_size": 32, "facedet_scale": 0.5})
    impl = Wav2LipProtocol(m.config)
    impl._scorer = _FixedTracks(tracks)
    impl._track_params = FaceTrackParams(min_face_size=32, facedet_scale=0.5)
    m._impl = impl
    m._ml_available = True
    m._backend = impl.name
    return m, impl._scorer


def test_wav2lip_tracks_are_averaged(video_sample):
    from ayase.vendor.syncnet_python.scoring import TrackScore

    m, scorer = _wav2lip_module(
        [TrackScore(lse_d=8.0, lse_c=2.0, offset=0), TrackScore(lse_d=10.0, lse_c=5.0, offset=1)]
    )
    m.process(video_sample)
    assert video_sample.quality_metrics.lse_c_syncnet == 3.5
    assert video_sample.quality_metrics.lse_d_syncnet == 9.0
    path, params, kw = scorer.calls[0]
    assert path == video_sample.path
    assert (params.min_face_size, params.facedet_scale, params.min_track) == (32, 0.5, 100)
    assert kw == {"batch_size": 20, "vshift": 15}


def test_wav2lip_without_face_track_leaves_lse_unset(video_sample):
    m, _ = _wav2lip_module([])
    m.process(video_sample)
    assert (
        video_sample.quality_metrics is None or video_sample.quality_metrics.lse_c_syncnet is None
    )


def test_wav2lip_protocol_reads_tunables_from_config():
    from ayase.modules.lip_sync import Wav2LipProtocol

    impl = Wav2LipProtocol({"min_track": 50, "vshift": 10, "device": "cpu"})
    assert (impl.min_track, impl.vshift, impl.device) == (50, 10, "cpu")
    assert (
        impl.min_face_size,
        impl.facedet_scale,
        impl.crop_scale,
        impl.num_failed_det,
        impl.batch_size,
    ) == (100, 0.25, 0.40, 25, 20)


# -- vendored syncnet_python numerics ---------------------------------------------


def test_lse_from_features_worked_example():
    """Mean distances over shifts (-1, 0, +1) are (5/3, 10/3, 0): LSE-D 0, LSE-C 5/3, offset -1."""
    import torch
    from ayase.vendor.syncnet_python.scoring import lse_from_features

    im_feat = torch.tensor([[0.0, 0.0], [3.0, 4.0], [0.0, 0.0]])
    cc_feat = torch.tensor([[0.0, 0.0], [0.0, 0.0], [3.0, 4.0]])
    score = lse_from_features(im_feat, cc_feat, vshift=1)
    assert score.lse_d == pytest.approx(0.0, abs=1e-4)
    assert score.lse_c == pytest.approx(5.0 / 3.0, abs=1e-4)
    assert score.offset == -1


class _WindowEcho:
    """Stand-in encoder returning slices of its input, to check the window layout."""

    def forward_lip(self, x):
        return x[:, 0, :, 0, 0]

    def forward_aud(self, x):
        return x[:, 0, 0, :]


def test_extract_features_window_layout():
    """12 frames give 7 windows of 5 frames; audio windows step by 4 MFCC frames."""
    import torch
    from ayase.vendor.syncnet_python.scoring import extract_features

    frames = np.stack([np.full((8, 8, 3), i, dtype=np.uint8) for i in range(12)])
    audio = np.random.default_rng(0).integers(-3000, 3000, size=12 * 640).astype(np.int16)
    im_feat, cc_feat = extract_features(
        _WindowEcho(), frames, audio, 16000, batch_size=3, device=torch.device("cpu")
    )
    assert im_feat.shape == (7, 5)
    assert im_feat[3].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert cc_feat.shape == (7, 20)
    assert torch.equal(cc_feat[1][:16], cc_feat[0][4:])


def test_extract_features_rejects_too_short_track():
    import torch
    from ayase.vendor.syncnet_python.scoring import extract_features

    frames = np.zeros((5, 8, 8, 3), dtype=np.uint8)
    audio = np.zeros(5 * 640, dtype=np.int16)
    with pytest.raises(ValueError, match="too short"):
        extract_features(
            _WindowEcho(), frames, audio, 16000, batch_size=20, device=torch.device("cpu")
        )


def _detections(frames, box):
    return [[{"frame": i, "bbox": list(box), "conf": 0.99}] for i in frames]


def test_track_shot_filters_and_interpolates():
    from ayase.vendor.syncnet_python.face_tracks import (
        FaceTrackParams,
        bb_intersection_over_union,
        track_shot,
    )

    params = FaceTrackParams()
    assert bb_intersection_over_union([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    # a track must be strictly longer than min_track
    assert track_shot(_detections(range(100), [100, 100, 300, 300]), params) == []
    assert len(track_shot(_detections(range(101), [100, 100, 300, 300]), params)) == 1
    # small faces need a lower min_face_size
    assert track_shot(_detections(range(120), [100, 100, 150, 150]), params) == []
    assert (
        len(
            track_shot(
                _detections(range(120), [100, 100, 150, 150]), FaceTrackParams(min_face_size=32)
            )
        )
        == 1
    )
    # a 10-frame detection gap is bridged by linear interpolation
    moving = [
        [{"frame": i, "bbox": [100.0 + i, 100.0, 300.0 + i, 300.0], "conf": 0.99}]
        for i in range(130)
    ]
    for i in range(60, 70):
        moving[i] = []
    tracks = track_shot(moving, params)
    assert len(tracks) == 1 and tracks[0]["frame"].tolist() == list(range(130))
    assert tracks[0]["bbox"][65].tolist() == pytest.approx([165.0, 100.0, 365.0, 300.0])


def test_s3fd_nms_keeps_best_of_overlapping_pair():
    from ayase.vendor.syncnet_python.s3fd.box_utils import nms_

    dets = np.array(
        [[0, 0, 10, 10, 0.8], [1, 1, 11, 11, 0.9], [100, 100, 110, 110, 0.7]], dtype=float
    )
    assert nms_(dets, 0.1).tolist() == [1, 2]
    assert nms_(np.empty((0, 5)), 0.1).tolist() == []


def test_syncnet_model_shapes_and_checkpoint_loading(tmp_path):
    import torch
    from ayase.vendor.syncnet_python.model import SyncNetModel, load_syncnet

    source = SyncNetModel()
    state = {k: v for k, v in source.state_dict().items() if not k.endswith("num_batches_tracked")}
    torch.save(state, tmp_path / "syncnet.model")
    loaded = load_syncnet(tmp_path / "syncnet.model", torch.device("cpu"))
    assert loaded.training is False
    assert torch.equal(
        loaded.state_dict()["netfclip.3.weight"], source.state_dict()["netfclip.3.weight"]
    )
    with torch.no_grad():
        assert loaded.forward_aud(torch.zeros(2, 1, 13, 20)).shape == (2, 1024)
        assert loaded.forward_lip(torch.zeros(2, 3, 5, 224, 224)).shape == (2, 1024)
    del state["netfclip.3.weight"]
    torch.save(state, tmp_path / "broken.model")
    with pytest.raises(RuntimeError, match="netfclip.3.weight"):
        load_syncnet(tmp_path / "broken.model", torch.device("cpu"))


def test_run_ffmpeg_reports_exit_code_and_stderr(monkeypatch):
    from ayase.vendor.syncnet_python import ffmpeg

    monkeypatch.setattr(
        ffmpeg.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="no such file\n"),
    )
    with pytest.raises(RuntimeError, match="code 1: no such file"):
        ffmpeg.run_ffmpeg(["-i", "x.mp4"])


def test_run_ffmpeg_reports_missing_binary(monkeypatch):
    from ayase.vendor.syncnet_python import ffmpeg

    monkeypatch.setattr(ffmpeg, "FFMPEG_BINARY", "ffmpeg-binary-that-does-not-exist")
    with pytest.raises(RuntimeError, match="not found"):
        ffmpeg.run_ffmpeg(["-version"])


# -- verse_bench protocol backend --------------------------------------------------


def _syncnet_inferencer():
    vendor = Path(__file__).resolve().parents[3] / "src" / "ayase" / "vendor" / "verse_bench"
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    try:
        from syncnet.syncnet_inferencer import SyncnetInferencer
    except Exception as e:  # SyncNet dependencies (insightface, moviepy) are optional
        pytest.skip(f"SyncNet backend unavailable: {e}")
    # Skip __init__: it loads the face detector, which frame extraction does not need.
    inferencer = SyncnetInferencer.__new__(SyncnetInferencer)
    inferencer.fps, inferencer.sr = 25, 16000
    return inferencer


def test_syncnet_extracts_path_with_spaces(tmp_path):
    """Paths with spaces and non-ASCII characters remain intact in shell calls."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found")
    inferencer = _syncnet_inferencer()
    clip = tmp_path / "René Descartes" / "clip 1.mp4"
    clip.parent.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=160x120:rate=25:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-shortest",
            str(clip),
        ],
        check=True,
    )
    work = tmp_path / "work"
    inferencer.video_to_frames_audio(clip, str(work))
    assert len(list((work / "images").glob("frame-*.jpg"))) >= 20
    assert (work / "audio.wav").stat().st_size > 0


def test_syncnet_extraction_failure_is_raised(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found")
    inferencer = _syncnet_inferencer()
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        inferencer.video_to_frames_audio(tmp_path / "no such file.mp4", str(tmp_path / "work"))

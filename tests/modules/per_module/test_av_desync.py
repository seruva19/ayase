"""Tests for the av_desync module (Synchformer DeSync).

Covers module basics, image/no-audio/no-backend graceful behaviour, and the
DeSync = |predicted offset| semantics via an injected stub inferencer. No model
weights are downloaded and no real Synchformer forward pass is run.
"""

import shutil
import subprocess
import wave
from pathlib import Path

import cv2
import numpy as np
import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics, Sample

_FPS = 25
_N = 25
_SIZE = 128
_SR = 22050


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build a tiny muxed audio+video clip
# ─────────────────────────────────────────────────────────────────────────────

def _write_silent_video(path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(_FPS), (_SIZE, _SIZE))
    for i in range(_N):
        frame = np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8)
        x = 10 + i * 3
        cv2.rectangle(frame, (x, 40), (x + 30, 70), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def _write_wav(path):
    n = int(_SR * (_N / _FPS))
    rng = np.random.RandomState(0)
    y = (0.2 * rng.randn(n)).astype(np.float32)
    ints = np.clip(y, -1, 1)
    ints = (ints * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SR)
        wf.writeframes(ints.tobytes())


def _make_av_clip(tmp_dir):
    """Return a path to a muxed audio+video mp4, or None if ffmpeg is missing."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        return None
    silent = tmp_dir / "silent.mp4"
    wav = tmp_dir / "audio.wav"
    muxed = tmp_dir / "clip.mp4"
    _write_silent_video(silent)
    _write_wav(wav)
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(silent), "-i", str(wav),
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(muxed),
        ],
        capture_output=True, timeout=60,
    )
    if result.returncode != 0 or not muxed.exists():
        return None
    return muxed


class _FakeInferencer:
    """Stand-in for Synchformer that returns a fixed predicted offset (seconds)."""

    def __init__(self, offset_sec):
        self.offset_sec = offset_sec

    def infer(self, path):
        return self.offset_sec


# ─────────────────────────────────────────────────────────────────────────────
# Basics / graceful behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_av_desync_basics():
    from ayase.modules.av_desync import AVDesyncModule
    _test_module_basics(AVDesyncModule, "av_desync")


def test_av_desync_image(image_sample):
    from ayase.modules.av_desync import AVDesyncModule
    image_sample.quality_metrics = QualityMetrics()
    m = AVDesyncModule({"test_mode": True})
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert image_sample.quality_metrics.desync_score is None


def test_has_audio_stream(video_sample, tmp_dir):
    from ayase.modules.av_desync import _has_audio_stream

    # cv2-written synthetic video has no audio track.
    assert _has_audio_stream(str(video_sample.path)) is False

    clip = _make_av_clip(tmp_dir)
    if clip is None:
        pytest.skip("ffmpeg/ffprobe not available")
    assert _has_audio_stream(str(clip)) is True


def test_av_desync_no_audio(video_sample):
    """A video without an audio stream leaves desync_score unset."""
    from ayase.modules.av_desync import AVDesyncModule

    video_sample.quality_metrics = QualityMetrics()
    m = AVDesyncModule({"test_mode": True})
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert video_sample.quality_metrics.desync_score is None


def test_av_desync_unavailable_backend(tmp_dir):
    """With no Synchformer backend, an A/V clip still yields desync_score None."""
    from ayase.modules.av_desync import AVDesyncModule

    clip = _make_av_clip(tmp_dir)
    if clip is None:
        pytest.skip("ffmpeg/ffprobe not available")

    m = AVDesyncModule({"test_mode": True})
    m.on_mount()  # test_mode skips setup -> backend stays "unavailable"
    assert m._backend == "unavailable"

    sample = Sample(path=clip, is_video=True)
    sample.quality_metrics = QualityMetrics()
    result = m.process(sample)
    assert result is sample
    assert sample.quality_metrics.desync_score is None


def test_av_desync_compute_semantics(tmp_dir):
    """DeSync = |predicted offset|: a stub inferencer returning -1.6 -> 1.6."""
    from ayase.modules.av_desync import AVDesyncModule

    clip = _make_av_clip(tmp_dir)
    if clip is None:
        pytest.skip("ffmpeg/ffprobe not available")
    pytest.importorskip("torch")

    m = AVDesyncModule()
    # Inject a fake backend directly (no weights download, no real forward pass).
    m._backend = "synchformer"
    m._inferencer = _FakeInferencer(-1.6)

    sample = Sample(path=clip, is_video=True)
    sample.quality_metrics = QualityMetrics()
    result = m.process(sample)
    assert result is sample
    assert sample.quality_metrics.desync_score == pytest.approx(1.6)

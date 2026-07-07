"""Tests for the av_align module (AV-Align, TempoTokens / Yariv et al. 2024).

Covers module basics, graceful no-audio behaviour, an exact unit test of the
faithful IoU port, and a synthetic matched-vs-mismatched signal test that
exercises the real optical-flow + librosa onset pipeline. No model downloads.
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

# Synthetic-signal parameters (validated to yield matched IoU=1.0, mismatched=0.0)
_FPS = 25
_N = 75
_SIZE = 128
_EVENTS = [15, 30, 45, 60]  # frames where a large block jumps -> flow peaks
_SR = 22050


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic-signal builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_burst_frames():
    """Frames with a large white block that jumps at each event frame.

    A jump creates a large mean optical-flow magnitude between the previous and
    current frame (a local maximum); the block is static otherwise, so flow is
    ~0 elsewhere. Video peaks land at ``event_frame / fps`` seconds.
    """
    targets = [(60, 60), (10, 60), (60, 10), (35, 35)]
    ev_map = dict(zip(_EVENTS, targets))
    frames = []
    cur = (10, 10)
    for i in range(_N):
        if i in ev_map:
            cur = ev_map[i]
        frame = np.zeros((_SIZE, _SIZE, 3), dtype=np.uint8)
        x, y = cur
        cv2.rectangle(frame, (x, y), (x + 55, y + 55), (255, 255, 255), -1)
        frames.append(frame)
    return frames


def _write_click_wav(path, times, sr=_SR, duration=_N / _FPS):
    """Write a mono WAV with short broadband clicks at the given times (stdlib wave)."""
    n = int(sr * duration)
    y = np.zeros(n, dtype=np.float32)
    burst = int(0.03 * sr)
    rng = np.random.RandomState(0)
    for t in times:
        s = int(t * sr)
        e = min(n, s + burst)
        if e <= s:
            continue
        env = np.exp(-np.linspace(0, 6, e - s))
        y[s:e] += (rng.randn(e - s) * env).astype(np.float32)
    y = np.clip(y, -1.0, 1.0)
    ints = (y * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(ints.tobytes())


def _write_silent_video(path, frames, fps=_FPS):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (_SIZE, _SIZE))
    for f in frames:
        writer.write(f)
    writer.release()


def _mux(video_path, wav_path, out_path):
    """Mux a silent video with a WAV into an mp4 with an audio stream."""
    result = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(video_path), "-i", str(wav_path),
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path),
        ],
        capture_output=True, timeout=60,
    )
    return result.returncode == 0 and Path(out_path).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Basics / graceful behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_av_align_basics():
    from ayase.modules.av_align import AVAlignModule
    _test_module_basics(AVAlignModule, "av_align")


def test_av_align_image(image_sample):
    from ayase.modules.av_align import AVAlignModule
    image_sample.quality_metrics = QualityMetrics()
    m = AVAlignModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert image_sample.quality_metrics.av_align_score is None


def test_av_align_video_no_audio(video_sample):
    """A video without an audio stream must leave av_align_score unset."""
    from ayase.modules.av_align import AVAlignModule, _has_audio_stream

    assert _has_audio_stream(str(video_sample.path)) is False
    video_sample.quality_metrics = QualityMetrics()
    m = AVAlignModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert video_sample.quality_metrics.av_align_score is None


# ─────────────────────────────────────────────────────────────────────────────
# Faithful IoU port — exact values
# ─────────────────────────────────────────────────────────────────────────────

def test_calc_iou_faithful():
    from ayase.modules.av_align import _calc_intersection_over_union as iou

    fps = 25  # window = +/- 0.04 s
    # Perfect overlap -> 1.0
    assert iou([1.0, 2.0], [1.0, 2.0], fps) == pytest.approx(1.0)
    # Partial: 2 of 3 audio peaks match -> 2 / (3 + 3 - 2) = 0.5
    assert iou([1.0, 2.0, 3.0], [1.0, 2.0, 5.0], fps) == pytest.approx(0.5)
    # No overlap -> 0.0
    assert iou([1.0], [5.0], fps) == pytest.approx(0.0)
    # A single video peak matches at most one audio peak: 1 / (2 + 1 - 1) = 0.5
    assert iou([1.0, 1.01], [1.0], fps) == pytest.approx(0.5)
    # Both empty -> None (guarded division by zero)
    assert iou([], [], fps) is None


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic matched vs mismatched
# ─────────────────────────────────────────────────────────────────────────────

def test_av_align_matched_vs_mismatched(tmp_dir):
    pytest.importorskip("librosa")
    from ayase.modules.av_align import (
        _detect_audio_peaks,
        _detect_video_peaks,
        _calc_intersection_over_union,
    )

    frames = _make_burst_frames()
    _, video_peaks = _detect_video_peaks(frames, _FPS)
    # Sanity: the large-block jumps produced one flow peak per event.
    assert len(video_peaks) == len(_EVENTS)

    matched_wav = tmp_dir / "matched.wav"
    mismatched_wav = tmp_dir / "mismatched.wav"
    _write_click_wav(matched_wav, [e / _FPS for e in _EVENTS])
    # Offset clicks by 0.2 s, far outside the +/-1/fps (0.04 s) tolerance window.
    _write_click_wav(mismatched_wav, [e / _FPS + 0.2 for e in _EVENTS])

    matched_audio = _detect_audio_peaks(str(matched_wav))
    mismatched_audio = _detect_audio_peaks(str(mismatched_wav))

    matched = _calc_intersection_over_union(list(matched_audio), video_peaks, _FPS)
    mismatched = _calc_intersection_over_union(list(mismatched_audio), video_peaks, _FPS)

    assert matched is not None and mismatched is not None
    assert matched >= 0.5, f"matched IoU unexpectedly low: {matched}"
    assert matched > mismatched, f"matched {matched} not > mismatched {mismatched}"
    assert mismatched <= 0.5


def test_av_align_end_to_end_muxed(tmp_dir):
    """Full process() on a muxed A/V clip: exercises audio extraction + flow."""
    pytest.importorskip("librosa")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg/ffprobe not available")

    from ayase.modules.av_align import AVAlignModule, _has_audio_stream

    frames = _make_burst_frames()
    silent = tmp_dir / "silent.mp4"
    wav = tmp_dir / "clicks.wav"
    muxed = tmp_dir / "av.mp4"
    _write_silent_video(silent, frames)
    _write_click_wav(wav, [e / _FPS for e in _EVENTS])
    if not _mux(silent, wav, muxed):
        pytest.skip("ffmpeg muxing failed in this environment")
    assert _has_audio_stream(str(muxed)) is True

    sample = Sample(path=muxed, is_video=True)
    sample.quality_metrics = QualityMetrics()
    m = AVAlignModule()
    m.setup()  # global light-mode test_mode skips on_mount->setup; init the real port
    assert m._backend == "port"
    result = m.process(sample)
    assert result is sample
    score = sample.quality_metrics.av_align_score
    assert score is not None
    assert 0.0 <= score <= 1.0

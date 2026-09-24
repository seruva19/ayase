"""Tests for lip_sync module."""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ..conftest import _test_module_basics
from ayase.models import QualityMetrics


def test_lip_sync_basics():
    from ayase.modules.lip_sync import LipSyncModule
    _test_module_basics(LipSyncModule, "lip_sync")

def test_lip_sync_video(video_sample):
    from ayase.modules.lip_sync import LipSyncModule
    video_sample.quality_metrics = QualityMetrics()
    m = LipSyncModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample


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
    """A path with spaces and non-ASCII characters used to break the shell call silently."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found")
    inferencer = _syncnet_inferencer()
    clip = tmp_path / "Дональд Трамп" / "clip 1.mp4"
    clip.parent.mkdir()
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=1",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=1", "-shortest", str(clip)], check=True)
    work = tmp_path / "work"
    inferencer.video_to_frames_audio(clip, str(work))
    assert len(list((work / "images").glob("frame-*.jpg"))) >= 20
    assert (work / "audio.wav").stat().st_size > 0


def test_syncnet_extraction_failure_is_raised(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not found")
    inferencer = _syncnet_inferencer()
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        inferencer.video_to_frames_audio(tmp_path / "нет такого файла.mp4", str(tmp_path / "work"))

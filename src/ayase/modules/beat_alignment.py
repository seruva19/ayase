"""Beat Alignment Score (BAS) — Bailando / CVPR 2022 (Eq. 15).

Measures synchronisation between music beats and kinematic dance beats.
Music beats are detected with librosa's beat tracker; kinematic beats are
local minima of the smoothed joint-velocity curve extracted from a per-frame
2D pose sequence (MediaPipe). The score is the published Bailando kernel:

    BAS = mean over music beats t_m of exp(-min_d (t_d - t_m)^2 / (2*sigma^2))

with distances in frames and sigma = 3.

bas_score — higher = better alignment (0-1)
Returns None when no audio stream is present or no pose is detected.

Basis: https://github.com/lisiyao21/Bailando (Li et al., CVPR 2022)
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _extract_audio_to_wav(video_path: str, wav_path: str) -> bool:
    """Use ffmpeg to extract audio track to a mono 22050 Hz WAV file."""
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vn", "-ac", "1", "-ar", "22050", "-f", "wav", wav_path,
            ],
            capture_output=True, timeout=60,
        )
        return result.returncode == 0 and Path(wav_path).stat().st_size > 44
    except Exception:
        return False


def _has_audio_stream(video_path: str) -> bool:
    """Check whether the video file contains an audio stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in result.stdout.lower()
    except Exception:
        return False


def _detect_music_beats(wav_path: str, sr: int = 22050) -> np.ndarray:
    """Music beat times (seconds) via librosa's onset+tempo beat tracker."""
    import librosa

    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    _tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    return librosa.frames_to_time(beat_frames, sr=sr)


def _kinematic_beats(video_path: str, pose) -> np.ndarray:
    """Kinematic beat frame indices — local minima of smoothed joint speed.

    Pose landmarks are extracted per frame; the mean per-joint displacement
    norm is smoothed and its local minima are the kinematic beats.
    """
    cap = cv2.VideoCapture(video_path)
    positions = []
    frame_idx = []
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            if results is None or results.pose_landmarks is None:
                continue
            lm = results.pose_landmarks.landmark
            positions.append([(p.x, p.y) for p in lm])
            frame_idx.append(idx)
    finally:
        cap.release()

    if len(positions) < 4:
        return np.array([])

    joints = np.asarray(positions, dtype=np.float32)  # [T, J, 2]
    speed = np.linalg.norm(np.diff(joints, axis=0), axis=2).mean(axis=1)  # [T-1]
    speed_idx = np.asarray(frame_idx[1:], dtype=np.float64)

    # Smooth (box filter ~ the published Savitzky-Golay role)
    win = max(3, len(speed) // 50 * 2 + 1)
    if win > 3:
        kernel = np.ones(win) / win
        sm = np.convolve(speed, kernel, mode="same")
    else:
        sm = speed

    beats = []
    for i in range(1, len(sm) - 1):
        if sm[i] < sm[i - 1] and sm[i] < sm[i + 1]:
            beats.append(speed_idx[i])
    return np.asarray(beats)


class BeatAlignmentModule(PipelineModule):
    name = "beat_alignment"
    provenance = "adapted"
    sources = {
        "bas_score": "BAS (Li et al., Bailando CVPR 2022 Eq.15; EDGE) — https://github.com/lisiyao21/Bailando",
    }
    deviations = {
        "bas_score": "the source uses AIST++/OpenPose 3D pose; here MediaPipe 2D pose and joint speed in normalized coordinates — the BAS formula is the same",
    }
    description = "BAS beat alignment score — audio-motion sync (Bailando/CVPR 2022)"
    default_config = {
        "sigma": 3.0,  # published normalisation (frames)
    }
    metric_groups = {
        "bas_score": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._librosa_available = False
        self._pose = None

    def setup(self) -> None:
        try:
            import librosa  # noqa: F401
            self._librosa_available = True
        except ImportError:
            logger.warning("BeatAlignment unavailable: librosa not installed")
            return
        try:
            import mediapipe as mp

            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                enable_segmentation=False, min_detection_confidence=0.5,
            )
            self._ml_available = True
            logger.info("BeatAlignment initialised (librosa + mediapipe)")
        except ImportError:
            logger.warning("BeatAlignment unavailable: mediapipe not installed")
        except Exception as e:
            logger.warning("BeatAlignment pose init failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        if not _has_audio_stream(str(sample.path)):
            logger.debug("BeatAlignment: no audio in %s, skipping", sample.path.name)
            return sample

        try:
            score = self._compute_bas(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.bas_score = score
        except Exception as e:
            logger.warning("BeatAlignment failed for %s: %s", sample.path, e)

        return sample

    def _compute_bas(self, sample: Sample) -> Optional[float]:
        """BAS: mean over music beats of exp(-min kinematic-beat dist^2/18)."""
        sigma = float(self.config.get("sigma", 3.0))

        cap = cv2.VideoCapture(str(sample.path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        cap.release()
        if fps <= 0:
            fps = 30.0

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name

        try:
            if not _extract_audio_to_wav(str(sample.path), wav_path):
                return None

            music_beats = _detect_music_beats(wav_path)  # seconds
            if len(music_beats) == 0:
                return None
            music_frames = music_beats * fps

            kinematic_beats = _kinematic_beats(str(sample.path), self._pose)  # frames
            if len(kinematic_beats) == 0:
                return None

            terms = []
            for tm in music_frames:
                d = float(np.min(np.abs(kinematic_beats - tm)))
                terms.append(np.exp(-(d * d) / (2.0 * sigma * sigma)))
            return float(np.mean(terms))
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

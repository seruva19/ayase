"""AV-Align — audio/video peak Intersection-over-Union (Yariv et al., 2024).

Faithful port of the AV-Align metric from TempoTokens ("Diverse and Aligned
Audio-to-Video Generation via Text-to-Video Model Adaptation", Yariv et al.,
AAAI 2024; reference:
https://github.com/guyyariv/TempoTokens/blob/master/av_align.py).

Algorithm (identical to the reference implementation):
  1. Audio peaks — ``librosa`` onset detection: onset-strength envelope followed
     by ``librosa.onset.onset_detect``; peak frames converted to seconds.
  2. Video peaks — dense Farneback optical flow between consecutive frames; the
     per-frame mean flow magnitude forms a trajectory whose local maxima (with
     magnitude ≥ 0.1) are the video peaks, in seconds.
  3. Score — Intersection over Union of the two event sets: an audio peak and a
     video peak intersect when they fall within ±1/fps seconds of each other;
     each video peak matches at most one audio peak. IoU = ``|∩| / (|A| + |V| − |∩|)``.

``av_align_score`` — higher = better audio-visual alignment (0-1). This is a
published deterministic algorithm, so a faithful port *is* the real backend
(``_backend = "port"``). Optical flow uses Farneback (cv2), exactly as the
reference (dependency-free; RAFT is deliberately not substituted because the
0.1 magnitude threshold is calibrated for Farneback average magnitudes).

Returns ``None`` when no audio stream is present or when ``librosa`` is missing.
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_audio_stream(video_path: str) -> bool:
    """Return True iff *video_path* contains an audio stream (via ffprobe)."""
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


def _extract_audio_to_wav(video_path: str, wav_path: str) -> bool:
    """Extract the audio track to a mono 22050 Hz WAV via ffmpeg.

    22050 Hz matches ``librosa.load``'s default sample rate, so the subsequent
    onset detection sees the audio exactly as the reference does.
    """
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


def _detect_audio_peaks(audio_file: str) -> np.ndarray:
    """Detect audio peaks via librosa onset detection (reference method).

    Mirrors TempoTokens ``detect_audio_peaks``: onset-strength envelope followed
    by ``onset_detect`` with default parameters, converted to seconds.
    """
    import librosa

    y, sr = librosa.load(audio_file)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


# ─────────────────────────────────────────────────────────────────────────────
# Optical-flow (video peak) helpers — faithful to the reference
# ─────────────────────────────────────────────────────────────────────────────

def _compute_of(img1: np.ndarray, img2: np.ndarray) -> float:
    """Average Farneback optical-flow magnitude between two frames.

    Identical to TempoTokens ``compute_of``: BGR→gray, Farneback with parameters
    ``(0.5, 3, 15, 3, 5, 1.2, 0)``, mean of the flow-vector magnitude.
    """
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    avg_magnitude = cv2.mean(magnitude)[0]
    return avg_magnitude


def _find_local_max_indexes(arr: List[float], fps: float) -> List[float]:
    """Local maxima (in seconds) of *arr*, ignoring magnitudes below 0.1.

    Faithful to the reference ``find_local_max_indexes``: the ``>= 0.1`` guard
    prevents static scenes with tiny optical flow from registering as peaks.
    """
    local_extrema_indexes: List[float] = []
    n = len(arr)
    for i in range(1, n - 1):
        if arr[i - 1] < arr[i] > arr[i + 1] and arr[i] >= 0.1:
            local_extrema_indexes.append(i / fps)
    return local_extrema_indexes


def _detect_video_peaks(frames: List[np.ndarray], fps: float) -> Tuple[List[float], List[float]]:
    """Return (flow_trajectory, video_peak_times) for a list of frames.

    Reproduces TempoTokens ``detect_video_peaks`` exactly, including the
    duplicated first trajectory element ``compute_of(frames[0], frames[1])``.
    """
    flow_trajectory = [_compute_of(frames[0], frames[1])] + [
        _compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))
    ]
    video_peaks = _find_local_max_indexes(flow_trajectory, fps)
    return flow_trajectory, video_peaks


def _calc_intersection_over_union(
    audio_peaks: List[float], video_peaks: List[float], fps: float
) -> Optional[float]:
    """IoU of audio and video peak sets within a ±1/fps window (reference).

    Faithful to ``calc_intersection_over_union``: each video peak matches at most
    one audio peak. Returns ``None`` when both event sets are empty (the
    reference would divide by zero; this is the only added guard).
    """
    intersection_length = 0
    used_video_peaks = [False] * len(video_peaks)
    for audio_peak in audio_peaks:
        for j, video_peak in enumerate(video_peaks):
            if not used_video_peaks[j] and video_peak - 1 / fps < audio_peak < video_peak + 1 / fps:
                intersection_length += 1
                used_video_peaks[j] = True
                break
    denominator = len(audio_peaks) + len(video_peaks) - intersection_length
    if denominator <= 0:
        return None
    return intersection_length / denominator


def _extract_frames(video_path: str, max_frames: Optional[int] = None) -> Tuple[List[np.ndarray], float]:
    """Consecutively decode frames (needed for optical flow) with cleanup.

    Optical flow requires *consecutive* frames, so this reads them in order
    rather than uniformly sampling. ``max_frames`` bounds memory for pathological
    inputs; AV-Align targets short generated clips, so the default is generous.
    """
    frames: List[np.ndarray] = []
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return frames, fps


# ─────────────────────────────────────────────────────────────────────────────
# Module
# ─────────────────────────────────────────────────────────────────────────────

class AVAlignModule(PipelineModule):
    name = "av_align"
    description = (
        "AV-Align — IoU of audio onsets and optical-flow motion peaks "
        "(TempoTokens / Yariv et al. 2024; higher=better)"
    )
    default_config = {
        "max_frames": 3000,  # memory bound for consecutive-frame decode
    }
    metric_info = {
        "av_align_score": "AV-Align onset/flow-peak IoU within ±1/fps (0-1, higher=better)",
    }
    metric_groups = {
        "av_align_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "unavailable"
        self._librosa_available = False

    def setup(self) -> None:
        try:
            import librosa  # noqa: F401

            self._librosa_available = True
            self._backend = "port"
            logger.info("av_align initialised (faithful TempoTokens port; librosa onsets)")
        except ImportError:
            self._librosa_available = False
            self._backend = "unavailable"
            logger.warning(
                "av_align: librosa not installed (pip install librosa); "
                "av_align_score left unset."
            )

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        if not _has_audio_stream(str(sample.path)):
            logger.debug("av_align: no audio stream in %s, skipping", sample.path.name)
            return sample

        if self._backend != "port" or not self._librosa_available:
            return sample

        try:
            score = self._compute_av_align(sample.path)
            if score is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.av_align_score = float(np.clip(score, 0.0, 1.0))
            logger.debug("av_align for %s: IoU=%.4f", sample.path.name, score)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("av_align failed for %s: %s", sample.path, e)
        return sample

    def _compute_av_align(self, video_path: Path) -> Optional[float]:
        max_frames = self.config.get("max_frames", 3000)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            if not _extract_audio_to_wav(str(video_path), wav_path):
                return None

            audio_peaks = _detect_audio_peaks(wav_path)

            frames, fps = _extract_frames(str(video_path), max_frames=max_frames)
            if len(frames) < 2 or fps is None or fps <= 0:
                return None

            _, video_peaks = _detect_video_peaks(frames, fps)
            return _calc_intersection_over_union(list(audio_peaks), video_peaks, fps)
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

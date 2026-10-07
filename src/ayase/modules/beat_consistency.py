"""Beat Consistency — audio-beat to body-kinematic-beat synchronisation.

BEAT (Liu et al., ECCV 2022) and EMAGE (Liu et al., CVPR 2024) evaluate
co-speech gesture rhythm with the kernel

    BC = mean over audio beats t_a of exp(-min_k (t_k - t_a)^2 / (2*sigma^2))

where t_k are local minima of the smoothed joint-velocity curve and distances
are measured in seconds with sigma = 0.1. Audio beats come from librosa's
beat tracker; joint velocity is the mean per-joint displacement of the
MediaPipe body-pose sequence.

beat_consistency -- higher = better alignment (0-1).
Returns None when the video has no audio stream or no pose is detected.
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from .beat_alignment import (
    _detect_music_beats,
    _extract_audio_to_wav,
    _has_audio_stream,
)

logger = logging.getLogger(__name__)


class BeatConsistencyModule(PipelineModule):
    name = "beat_consistency"
    description = "Beat Consistency — audio/gesture beat kernel (BEAT, EMAGE)"
    provenance = "adapted"
    sources = {
        "beat_consistency": "BC, BEAT (Liu et al., ECCV 2022, arXiv:2203.05297); EMAGE (arXiv:2401.00374) — https://github.com/PantoMatrix/BEAT",
    }
    deviations = {
        "beat_consistency": "the kernel and sigma are the published ones, but joint velocity comes from MediaPipe 33-joint 2D pose instead of the source's SMPL-X skeleton, so absolute values are not comparable",
    }
    default_config = {
        "sigma": 0.1,  # seconds — published BEAT normalisation
        "fps": 0.0,    # 0 = read every frame
        "max_frames": 1200,
    }
    metric_info = {
        "beat_consistency": "BEAT beat-consistency kernel on body-pose velocity (0-1, higher=better)",
    }
    metric_groups = {"beat_consistency": "motion"}

    def __init__(self, config=None):
        super().__init__(config)
        self._pose = None
        self._backend = None
        self._librosa_available = False

    def setup(self) -> None:
        try:
            import librosa  # noqa: F401

            self._librosa_available = True
        except ImportError:
            logger.warning("beat_consistency: librosa not installed, disabled")
            return
        if self.config.get("test_mode"):
            self._backend = "unavailable"
            return
        try:
            import mediapipe as mp

            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                enable_segmentation=False, min_detection_confidence=0.5,
            )
            self._backend = "mediapipe"
        except ImportError:
            logger.warning("beat_consistency: mediapipe not installed, disabled")
            self._backend = "unavailable"
        except Exception as e:
            logger.warning("beat_consistency: Pose init failed: %s", e)
            self._backend = "unavailable"

    def on_dispose(self) -> None:
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
            self._pose = None

    @staticmethod
    def _kinematic_beats_sec(seq: np.ndarray, inds: np.ndarray,
                             fps_src: float) -> np.ndarray:
        if len(seq) < 4:
            return np.array([])
        speed = np.linalg.norm(np.diff(seq[:, :, :2], axis=0), axis=2).mean(axis=1)
        speed_t = inds[1:] / fps_src  # seconds
        win = max(3, len(speed) // 50 * 2 + 1)
        sm = np.convolve(speed, np.ones(win) / win, mode="same") if win > 3 else speed
        beats = []
        for i in range(1, len(sm) - 1):
            if sm[i] < sm[i - 1] and sm[i] < sm[i + 1]:
                beats.append(speed_t[i])
        return np.asarray(beats)

    def process(self, sample: Sample) -> Sample:
        if self._backend != "mediapipe":
            return sample
        if not sample.is_video:
            return sample
        if not _has_audio_stream(str(sample.path)):
            return sample
        try:
            score = self._compute(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.beat_consistency = score
        except Exception as e:
            logger.warning("beat_consistency: failed on %s: %s",
                           Path(sample.path).name, e)
        return sample

    def _compute(self, sample: Sample) -> Optional[float]:
        import cv2

        from ._mp_seq import body_pose_seq

        cap = cv2.VideoCapture(str(sample.path))
        fps_src = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()

        res = body_pose_seq(Path(sample.path),
                            fps=float(self.config.get("fps", 0.0)),
                            max_frames=int(self.config.get("max_frames", 1200)),
                            detector=self._pose)
        if res is None:
            return None
        kin = self._kinematic_beats_sec(res[0], res[1], fps_src)
        if len(kin) == 0:
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            if not _extract_audio_to_wav(str(sample.path), wav_path):
                return None
            music_beats = _detect_music_beats(wav_path)  # seconds
            if len(music_beats) == 0:
                return None
            sigma = float(self.config.get("sigma", 0.1))
            terms = []
            for tm in music_beats:
                d = float(np.min(np.abs(kin - tm)))
                terms.append(np.exp(-(d * d) / (2.0 * sigma * sigma)))
            return float(np.mean(terms))
        finally:
            try:
                Path(wav_path).unlink(missing_ok=True)
            except Exception:
                pass

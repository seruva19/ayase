"""Head Beat Align — audio-beat to head-motion-beat synchronisation.

Talking-head evals (SadTalker, Hallo, AniPortrait, EchoMimic) report "Beat
Align": the published Bailando kernel between audio beats detected with
librosa and kinematic beats of the head — the local minima of the smoothed
head-motion velocity:

    score = mean over audio beats t_m of exp(-min_d (t_d - t_m)^2 / (2*sigma^2))

with distances in frames and sigma = 3. Head motion is the per-frame 3DMM
pose sequence extracted with TDDFA; velocity is the L2 norm of successive
pose-coefficient differences.

head_beat_align -- higher = better alignment (0-1). Returns None when the
video has no audio stream or no face is detected.
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
from ._tddfa_coeffs import POSE_SLICE, ensure_tddfa_weights

logger = logging.getLogger(__name__)


def _head_kinematic_beats(params: np.ndarray, inds: np.ndarray) -> np.ndarray:
    """Local minima of the smoothed head-pose velocity curve, in resampled
    frame indices."""
    if len(params) < 4:
        return np.array([])
    speed = np.linalg.norm(np.diff(params, axis=0), axis=1)  # [T-1]
    speed_idx = inds[1:].astype(np.float64)

    win = max(3, len(speed) // 50 * 2 + 1)
    sm = np.convolve(speed, np.ones(win) / win, mode="same") if win > 3 else speed

    beats = []
    for i in range(1, len(sm) - 1):
        if sm[i] < sm[i - 1] and sm[i] < sm[i + 1]:
            beats.append(speed_idx[i])
    return np.asarray(beats)


class HeadBeatAlignModule(PipelineModule):
    name = "head_beat_align"
    description = "Beat Align — audio/head-motion sync via the Bailando kernel"
    provenance = "adapted"
    sources = {
        "head_beat_align": "BAS kernel (Li et al., Bailando CVPR 2022 Eq.15) applied to head pose; reported as Beat Align by SadTalker (arXiv:2211.12194), Hallo, AniPortrait, EchoMimic — https://github.com/lisiyao21/Bailando",
    }
    deviations = {
        "head_beat_align": "the BAS kernel is verbatim but kinematic beats come from the smoothed velocity of TDDFA/3DDFA_V2 pose coefficients rather than the pose track used by the source papers, so absolute values are not comparable",
    }
    default_config = {
        "sigma": 3.0,
        "fps": 25,
        "read_stride": 96,
        "rec_stride": 32,
        "det_size_threshold": 75,
        "det_score_threshold": 0.7,
        "det_target_size": 1280,
        "device": "auto",
    }
    models = [
        {
            "id": "akhaliq/RetinaFace-R50",
            "type": "huggingface",
            "url": "https://huggingface.co/akhaliq/RetinaFace-R50/resolve/main/RetinaFace-R50.pth",
            "task": "RetinaFace ResNet50 face detector (shared)",
        },
        {
            "id": "Stable-Human/3ddfa_v2",
            "type": "huggingface",
            "url": "https://huggingface.co/Stable-Human/3ddfa_v2/resolve/main/mb1_120x120.pth",
            "task": "TDDFA/3DDFA_V2 MobileNet-1 3DMM regressor (shared)",
        },
    ]
    metric_info = {
        "head_beat_align": "Bailando beat-alignment kernel on head-pose velocity (0-1, higher=better)",
    }
    metric_groups = {"head_beat_align": "face"}

    def __init__(self, config=None):
        super().__init__(config)
        self._extractor = None
        self._backend = None
        self._librosa_available = False

    def setup(self) -> None:
        try:
            import librosa  # noqa: F401

            self._librosa_available = True
        except ImportError:
            logger.warning("head_beat_align: librosa not installed, disabled")
            return
        if self.config.get("test_mode"):
            self._backend = "unavailable"
            return
        resources = ensure_tddfa_weights(self.config.get("models_dir", "models"),
                                         ["Resnet50_Final.pth", "mb1_120x120.pth"])
        if resources is None:
            self._backend = "unavailable"
            return
        device = self.config.get("device", "auto")
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        try:
            from ._tddfa_coeffs import TDDFACoeffExtractor

            self._extractor = TDDFACoeffExtractor(
                resources,
                device=device,
                fps=int(self.config.get("fps", 25)),
                read_stride=int(self.config.get("read_stride", 96)),
                rec_stride=int(self.config.get("rec_stride", 32)),
                det_size_threshold=int(self.config.get("det_size_threshold", 75)),
                det_score_threshold=float(self.config.get("det_score_threshold", 0.7)),
                det_target_size=int(self.config.get("det_target_size", 1280)),
            )
            self._backend = "tddfa"
        except Exception as e:
            logger.warning("head_beat_align: TDDFA init failed: %s", e)
            self._backend = "unavailable"

    def process(self, sample: Sample) -> Sample:
        if self._backend != "tddfa" or self._extractor is None:
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
                sample.quality_metrics.head_beat_align = score
        except Exception as e:
            logger.warning("head_beat_align: failed on %s: %s",
                           Path(sample.path).name, e)
        return sample

    def _compute(self, sample: Sample) -> Optional[float]:
        sigma = float(self.config.get("sigma", 3.0))
        res = self._extractor.extract(Path(sample.path))
        if res is None:
            return None
        params, inds = res
        kinematic_beats = _head_kinematic_beats(params[:, POSE_SLICE], inds)
        if len(kinematic_beats) == 0:
            return None

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            if not _extract_audio_to_wav(str(sample.path), wav_path):
                return None
            music_beats = _detect_music_beats(wav_path)  # seconds
            if len(music_beats) == 0:
                return None
            music_frames = music_beats * float(self.config.get("fps", 25))

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

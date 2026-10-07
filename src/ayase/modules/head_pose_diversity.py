"""Head-pose diversity — spread of head pose over a video.

SadTalker's talking-head evaluation reports "Diversity": the spread of the
head-pose sequence across the video, i.e. the standard deviation of the head
pose embedding over time (higher = more varied head motion). Per-frame 3DMM
pose coefficients are extracted with TDDFA and the score is the mean per-dim
standard deviation of the 12 pose parameters.

head_pose_diversity -- higher = more diverse head motion (0+).
"""

import logging
from pathlib import Path

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ._tddfa_coeffs import POSE_SLICE, ensure_tddfa_weights

logger = logging.getLogger(__name__)


class HeadPoseDiversityModule(PipelineModule):
    name = "head_pose_diversity"
    description = "Head pose diversity — temporal std of 3DMM pose coefficients"
    provenance = "adapted"
    sources = {
        "head_pose_diversity": "Diversity, SadTalker (Zhang et al., arXiv:2211.12194) — https://github.com/OpenTalker/SadTalker",
    }
    deviations = {
        "head_pose_diversity": "the source reports the std of head-pose embeddings extracted with Hopenet angles; here TDDFA/3DDFA_V2 12-dim pose coefficients are used, so absolute values are not comparable to the paper",
    }
    default_config = {
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
        "head_pose_diversity": "Temporal std of head-pose coefficients (higher=more diverse)",
    }
    metric_groups = {"head_pose_diversity": "face"}

    def __init__(self, config=None):
        super().__init__(config)
        self._extractor = None
        self._backend = None

    def setup(self) -> None:
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
            logger.warning("head_pose_diversity: TDDFA init failed: %s", e)
            self._backend = "unavailable"

    def process(self, sample: Sample) -> Sample:
        if self._backend != "tddfa" or self._extractor is None:
            return sample
        if not sample.is_video:
            return sample
        try:
            res = self._extractor.extract(Path(sample.path))
            if res is None:
                return sample
            pose = res[0][:, POSE_SLICE]
            if len(pose) < 2:
                return sample
            score = float(np.std(pose, axis=0).mean())
            if not np.isfinite(score):
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.head_pose_diversity = score
        except Exception as e:
            logger.warning("head_pose_diversity: failed on %s: %s",
                           Path(sample.path).name, e)
        return sample

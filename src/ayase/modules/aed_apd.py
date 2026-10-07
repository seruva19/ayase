"""Average Expression/Pose Distance (AED/APD) against a driver video.

PIRenderer's talking-head evaluation: the generated video and the driver video
are decoded to 3DMM coefficients per frame; AED is the mean absolute distance
of the expression coefficients and APD the mean absolute distance of the pose
coefficients, averaged over aligned frames. Lower = closer to the driver.
Requires ``sample.reference_path`` to point at the driver video.
"""

import logging
from pathlib import Path

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ._tddfa_coeffs import EXPR_SLICE, POSE_SLICE, ensure_tddfa_weights

logger = logging.getLogger(__name__)


class AedApdModule(PipelineModule):
    name = "aed_apd"
    description = "AED/APD: 3DMM expression and pose distance to a driver video"
    provenance = "adapted"
    sources = {
        "aed": "AED, PIRenderer (Ren et al., arXiv:2109.08379) — https://github.com/RenYurui/PIRender",
        "apd": "APD, PIRenderer (Ren et al., arXiv:2109.08379) — https://github.com/RenYurui/PIRender",
    }
    deviations = {
        "aed": "the published protocol extracts expression coefficients with Deep3DFaceRecon (BFM09); here TDDFA/3DDFA_V2 62-dim coefficients are used, so absolute values are not comparable to the paper",
        "apd": "the published protocol extracts pose coefficients with Deep3DFaceRecon (BFM09); here TDDFA/3DDFA_V2 62-dim coefficients are used, so absolute values are not comparable to the paper",
    }
    requires_reference = True
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
        "aed": "Mean absolute distance of 3DMM expression coefficients vs driver (lower=closer)",
        "apd": "Mean absolute distance of 3DMM pose coefficients vs driver (lower=closer)",
    }
    metric_groups = {"aed": "face", "apd": "face"}

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
            logger.warning("aed_apd: TDDFA init failed: %s", e)
            self._backend = "unavailable"

    def process(self, sample: Sample) -> Sample:
        if self._backend != "tddfa" or self._extractor is None:
            return sample
        if not sample.is_video:
            return sample
        ref = sample.reference_path
        if ref is None or not Path(ref).is_file():
            return sample
        try:
            cand = self._extractor.extract(Path(sample.path))
            ref_coefs = self._extractor.extract(Path(ref))
            if cand is None or ref_coefs is None:
                return sample
            n = min(len(cand[0]), len(ref_coefs[0]))
            if n == 0:
                return sample
            aed = float(
                np.abs(cand[0][:n, EXPR_SLICE] - ref_coefs[0][:n, EXPR_SLICE]).mean()
            )
            apd = float(
                np.abs(cand[0][:n, POSE_SLICE] - ref_coefs[0][:n, POSE_SLICE]).mean()
            )
            if not np.isfinite(aed) or not np.isfinite(apd):
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.aed = aed
            sample.quality_metrics.apd = apd
        except Exception as e:
            logger.warning("aed_apd: failed on %s: %s", Path(sample.path).name, e)
        return sample

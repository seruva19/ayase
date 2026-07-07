"""RAFT optical flow motion scoring module.

From Data-Juicer's video_motion_score_raft_filter.
Uses torchvision's RAFT model for accurate optical flow estimation.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class RAFTMotionModule(PipelineModule):
    name = "raft_motion"
    description = "RAFT optical flow motion scoring (torchvision)"
    default_config = {"subsample": 8}
    metric_groups = {
        "raft_motion_score": "motion",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._transforms = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import torch  # noqa: F401
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            from ayase.runtime import resolve_torch_device, shared_runtime_resource

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            weights = Raft_Large_Weights.DEFAULT

            def load_raft():
                model = raft_large(weights=weights).to(self._device).eval()
                return model, weights.transforms()

            self._model, self._transforms = shared_runtime_resource(
                self,
                ("raft", "raft_large", self._device),
                load_raft,
            )
            self._ml_available = True
            logger.info("RAFT model loaded on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("RAFT unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        if not sample.is_video:
            return sample

        try:
            import torch
            from ayase.image import sample_frames

            subsample = self.config.get("subsample", 8)
            frames = sample_frames(sample.path, max_frames=subsample, color="rgb")

            if len(frames) < 2:
                return sample

            motion_scores = []
            for i in range(len(frames) - 1):
                arr1 = np.ascontiguousarray(frames[i], dtype=np.float32)
                arr2 = np.ascontiguousarray(frames[i + 1], dtype=np.float32)
                frame1 = torch.from_numpy(arr1).permute(2, 0, 1).unsqueeze(0)
                frame2 = torch.from_numpy(arr2).permute(2, 0, 1).unsqueeze(0)

                img1, img2 = self._transforms(frame1, frame2)
                img1 = img1.to(self._device)
                img2 = img2.to(self._device)

                with torch.no_grad():
                    flow_list = self._model(img1, img2)
                    flow = flow_list[-1]  # last iteration

                magnitude = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2)
                motion_scores.append(float(magnitude.mean().cpu()))

            if motion_scores:
                sample.quality_metrics.raft_motion_score = float(np.mean(motion_scores))
        except Exception as e:
            logger.warning("RAFT motion scoring failed: %s", e)
        return sample

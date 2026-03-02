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

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._transforms = None
        self._device = None

    def setup(self) -> None:
        try:
            import torch
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

            weights = Raft_Large_Weights.DEFAULT
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = raft_large(weights=weights).to(device).eval()
            self._transforms = weights.transforms()
            self._device = device
            self._ml_available = True
            logger.info("RAFT model loaded on %s", device)
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
            import cv2
            import torch

            subsample = self.config.get("subsample", 8)
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb)
            cap.release()

            if len(frames) < 2:
                return sample

            motion_scores = []
            for i in range(len(frames) - 1):
                frame1 = torch.from_numpy(frames[i]).permute(2, 0, 1).float().unsqueeze(0)
                frame2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float().unsqueeze(0)

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

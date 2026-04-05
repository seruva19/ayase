"""RAFT optical flow magnitude analysis across all consecutive frame pairs.

Uses RAFT-Large or RAFT-Small from torchvision to compute mean flow score.
Higher flow_score indicates more motion. Low scores flag static content."""

import logging
import numpy as np
import cv2
from typing import List

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AdvancedFlowModule(PipelineModule):
    name = "advanced_flow"
    description = "RAFT optical flow: flow_score (all consecutive pairs)"

    default_config = {
        "use_large_model": True,
        "max_frames": 150,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.use_large_model = self.config.get("use_large_model", True)
        self.max_frames = self.config.get("max_frames", 150)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._transforms = None

    def setup(self) -> None:
        try:
            import os
            import torch

            # Redirect torch hub cache to models_dir so RAFT weights respect config
            models_dir = self.config.get("models_dir")
            if models_dir:
                os.environ["TORCH_HOME"] = str(models_dir)

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            if self.use_large_model:
                from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
                weights = Raft_Large_Weights.DEFAULT
                self._model = raft_large(weights=weights, progress=False).to(self._device)
                logger.info(f"Setting up RAFT-Large on {self._device}...")
            else:
                from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
                weights = Raft_Small_Weights.DEFAULT
                self._model = raft_small(weights=weights, progress=False).to(self._device)
                logger.info(f"Setting up RAFT-Small on {self._device}...")

            self._model.eval()
            self._transforms = weights.transforms()
            self._ml_available = True

        except ImportError:
            logger.warning("torchvision >= 0.13 required for RAFT.")
        except Exception as e:
            logger.warning(f"Failed to setup RAFT: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            import torch

            frames = self._load_all_frames(sample)
            if len(frames) < 2:
                return sample

            # Compute flow for ALL consecutive frame pairs (matches EvalCrafter)
            optical_flows = []

            with torch.no_grad():
                for i in range(len(frames) - 1):
                    img1 = torch.from_numpy(frames[i]).permute(2, 0, 1).unsqueeze(0).to(self._device)
                    img2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).unsqueeze(0).to(self._device)

                    if self._transforms:
                        img1, img2 = self._transforms(img1, img2)

                    list_of_flows = self._model(img1, img2)
                    predicted_flow = list_of_flows[-1]

                    flow_magnitude = torch.norm(predicted_flow.squeeze(0), dim=0)
                    mean_flow = flow_magnitude.mean().item()
                    optical_flows.append(mean_flow)

            if not optical_flows:
                return sample

            flow_score = float(np.mean(optical_flows))

            if sample.quality_metrics is None:
                from ayase.models import QualityMetrics
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.flow_score = flow_score

            if flow_score < 0.5:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low Dynamic Degree (Static): {flow_score:.2f}",
                        details={"flow_score": flow_score},
                    )
                )

        except Exception as e:
            logger.warning(f"Flow analysis failed: {e}")

        return sample

    def _load_all_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames > 0 and total_frames > self.max_frames:
                # Subsample uniformly to stay within max_frames
                indices = set(np.linspace(0, total_frames - 1, self.max_frames, dtype=int))
                frame_idx = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx in indices:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame)
                    frame_idx += 1
            else:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
                    if len(frames) >= self.max_frames:
                        break
            cap.release()
        except Exception as e:
            logger.debug(f"Failed to load frames for advanced flow: {e}")
        return frames

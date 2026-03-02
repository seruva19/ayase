"""Inception Score (IS) — EvalCrafter metric #3.

Computes Inception Score by passing video frames through InceptionV3 and
measuring the KL divergence between the conditional and marginal class
distributions.  Higher IS = better visual quality and diversity.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class InceptionScoreModule(PipelineModule):
    name = "inception_score"
    description = "Inception Score (IS) using InceptionV3 — EvalCrafter quality metric"
    default_config = {
        "num_frames": 16,
        "splits": 1,  # Per-sample we use 1 split; dataset-level uses 10
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.num_frames = self.config.get("num_frames", 16)
        self.splits = self.config.get("splits", 1)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._transform = None

    def setup(self):
        try:
            import torch
            from torchvision import models, transforms
            from torchvision.models import Inception_V3_Weights

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading InceptionV3 for IS on {self._device}...")

            self._model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
            self._model.to(self._device).eval()

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self._ml_available = True
        except Exception as e:
            logger.warning(f"Failed to load InceptionV3: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            import torch

            frames = self._load_frames(sample)
            if len(frames) < 2:
                return sample

            # Get softmax probabilities from InceptionV3
            probs_list = []
            with torch.no_grad():
                for frame in frames:
                    tensor = self._transform(frame).unsqueeze(0).to(self._device)
                    logits = self._model(tensor)
                    # inception_v3 returns InceptionOutputs during eval — extract .logits
                    if hasattr(logits, "logits"):
                        logits = logits.logits
                    probs = torch.softmax(logits, dim=1)
                    probs_list.append(probs)

            probs_all = torch.cat(probs_list, dim=0)  # [N, 1000]

            # IS = exp(E[KL(p(y|x) || p(y))])
            # p(y) = mean over all frames
            marginal = probs_all.mean(dim=0, keepdim=True)  # [1, 1000]
            kl_divs = probs_all * (torch.log(probs_all + 1e-10) - torch.log(marginal + 1e-10))
            kl_per_frame = kl_divs.sum(dim=1)  # [N]
            is_score = torch.exp(kl_per_frame.mean()).item()

            from ayase.models import QualityMetrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.is_score = float(is_score)

            if is_score < 2.0:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low Inception Score ({is_score:.2f})",
                        details={"is_score": float(is_score)},
                        recommendation="Low IS indicates poor visual quality or low diversity across frames.",
                    )
                )

        except Exception as e:
            logger.warning(f"Inception Score failed for {sample.path}: {e}")

        return sample

    def _load_frames(self, sample: Sample):
        frames = []
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    return frames
                indices = np.linspace(0, total - 1, min(self.num_frames, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            else:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.debug(f"Frame loading failed for IS: {e}")
        return frames

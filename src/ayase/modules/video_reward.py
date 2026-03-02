"""VideoAlign reward model module.

NeurIPS 2025. Human preference alignment scoring for video generation.
Based on QWen2-VL reward model.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VideoRewardModule(PipelineModule):
    name = "video_reward"
    description = "VideoAlign human preference reward model (NeurIPS 2025)"
    default_config = {
        "model_name": "KlingTeam/VideoAlign-Reward",
        "subsample": 8,
        "trust_remote_code": True,
        "model_revision": None,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = None

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoProcessor

            model_name = self.config.get("model_name", "KlingTeam/VideoAlign-Reward")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=trc, revision=rev
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_name, trust_remote_code=trc, revision=rev
            ).to(device)
            self._model.eval()
            self._device = device
            self._ml_available = True
            logger.info("VideoAlign reward model loaded on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("VideoAlign unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            import cv2
            import torch
            from PIL import Image

            subsample = self.config.get("subsample", 8)
            frames = []

            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = list(range(0, total, max(1, total // subsample)))[:subsample]
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(rgb))
                cap.release()
            else:
                frames.append(Image.open(str(sample.path)).convert("RGB"))

            if not frames:
                return sample

            caption = sample.caption.text if sample.caption else "a video"
            prompt = f"Rate the quality of this video: {caption}"

            inputs = self._processor(
                text=prompt, images=frames, return_tensors="pt"
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                score = outputs.logits.squeeze().item()

            sample.quality_metrics.video_reward_score = float(score)
        except Exception as e:
            logger.warning("VideoAlign processing failed: %s", e)
        return sample

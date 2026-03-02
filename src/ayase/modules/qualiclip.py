"""QualiCLIP opinion-unaware quality assessment module."""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class QualiCLIPModule(PipelineModule):
    name = "qualiclip"
    description = "QualiCLIP opinion-unaware CLIP-based no-reference IQA"
    default_config = {"subsample": 8}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None

    def setup(self) -> None:
        try:
            import pyiqa
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = pyiqa.create_metric("qualiclip", device=device)
            self._ml_available = True
            logger.info("QualiCLIP model loaded on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("QualiCLIP unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        try:
            import cv2
            import torch

            frames = self._load_frames(sample)
            if not frames:
                return sample

            scores = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = (
                    torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                )
                tensor = tensor.to(next(self._model.parameters()).device)
                with torch.no_grad():
                    score = self._model(tensor).item()
                scores.append(score)

            sample.quality_metrics.qualiclip_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("QualiCLIP processing failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> list:
        import cv2

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
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames

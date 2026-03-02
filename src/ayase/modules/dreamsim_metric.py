"""DreamSim foundation model perceptual similarity module."""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DreamSimModule(PipelineModule):
    name = "dreamsim"
    description = "DreamSim foundation model perceptual similarity (CLIP+DINO ensemble)"
    default_config = {"subsample": 8, "model_type": "ensemble"}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._preprocess = None

    def setup(self) -> None:
        try:
            from dreamsim import dreamsim

            model, preprocess = dreamsim(pretrained=True)
            self._model = model
            self._preprocess = preprocess
            self._ml_available = True
            logger.info("DreamSim model loaded")
        except (ImportError, Exception) as e:
            logger.warning("DreamSim unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        reference_path = getattr(sample, "reference_path", None)
        if reference_path is None:
            # For videos without reference, compute inter-frame similarity
            if sample.is_video:
                return self._process_video_self(sample)
            return sample

        try:
            from PIL import Image

            ref_img = Image.open(str(reference_path)).convert("RGB")
            dist_img = Image.open(str(sample.path)).convert("RGB")

            ref_t = self._preprocess(ref_img).unsqueeze(0)
            dist_t = self._preprocess(dist_img).unsqueeze(0)

            import torch
            with torch.no_grad():
                distance = self._model(ref_t, dist_t).item()

            sample.quality_metrics.dreamsim = float(distance)
        except Exception as e:
            logger.warning("DreamSim processing failed: %s", e)
        return sample

    def _process_video_self(self, sample: Sample) -> Sample:
        """For videos: compute average DreamSim distance between consecutive frames."""
        try:
            import cv2
            from PIL import Image
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
                    pil_img = Image.fromarray(rgb)
                    frames.append(self._preprocess(pil_img).unsqueeze(0))
            cap.release()

            if len(frames) < 2:
                return sample

            distances = []
            with torch.no_grad():
                for i in range(len(frames) - 1):
                    dist = self._model(frames[i], frames[i + 1]).item()
                    distances.append(dist)

            sample.quality_metrics.dreamsim = float(np.mean(distances))
        except Exception as e:
            logger.warning("DreamSim video processing failed: %s", e)
        return sample


class DreamSimCompatModule(DreamSimModule):
    """Compatibility alias matching filename-based discovery."""

    name = "dreamsim_metric"

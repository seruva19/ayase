"""Recognize Anything Model (RAM) tagging module.

From Data-Juicer's video_tagging_from_frames_filter.
RAM auto-tags images with 6,449 tag categories.
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class RAMTaggingModule(PipelineModule):
    name = "ram_tagging"
    description = "RAM (Recognize Anything Model) auto-tagging for video frames"
    default_config = {
        "model_name": "xinyu1205/recognize-anything-plus-model",
        "subsample": 4,
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
            from transformers import AutoModelForImageClassification, AutoProcessor

            model_name = self.config.get(
                "model_name", "xinyu1205/recognize-anything-plus-model"
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=trc, revision=rev
            )
            self._model = AutoModelForImageClassification.from_pretrained(
                model_name, trust_remote_code=trc, revision=rev
            ).to(device)
            self._model.eval()
            self._device = device
            self._ml_available = True
            logger.info("RAM model loaded on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("RAM tagging unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            import cv2
            import torch
            from PIL import Image

            frames = self._load_frames(sample)
            if not frames:
                return sample

            all_tags = set()
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                inputs = self._processor(images=pil_img, return_tensors="pt").to(
                    self._device
                )
                with torch.no_grad():
                    outputs = self._model(**inputs)

                logits = outputs.logits
                probs = torch.sigmoid(logits)
                threshold = 0.5
                predicted = (probs > threshold).squeeze()
                if hasattr(self._model.config, "id2label"):
                    for idx in predicted.nonzero(as_tuple=True)[0]:
                        tag = self._model.config.id2label.get(idx.item(), "")
                        if tag:
                            all_tags.add(tag)

            if all_tags:
                sample.quality_metrics.ram_tags = ", ".join(sorted(all_tags))
        except Exception as e:
            logger.warning("RAM tagging failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 4)
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

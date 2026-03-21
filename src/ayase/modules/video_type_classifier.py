"""Video type classifier module.

From NVIDIA Curator's video type classification concept.
Classifies content as real-world, animated, game, abstract, etc.
Uses CLIP zero-shot classification.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

VIDEO_TYPE_LABELS = [
    "real-world photography",
    "animated cartoon or anime",
    "video game footage",
    "abstract visual pattern",
    "screen recording or presentation",
    "text-heavy or document",
]

VIDEO_TYPE_SHORT = [
    "real",
    "animated",
    "game",
    "abstract",
    "screen_recording",
    "text_heavy",
]


class VideoTypeClassifierModule(PipelineModule):
    name = "video_type_classifier"
    description = "CLIP zero-shot video content type classification"
    default_config = {"subsample": 4}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = None

    def setup(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = "openai/clip-vit-base-patch32"
            self._model = CLIPModel.from_pretrained(model_name).to(device)
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model.eval()
            self._device = device
            self._ml_available = True
            logger.info("CLIP model loaded for video type classification on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("Video type classifier unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            import cv2
            import torch
            from PIL import Image

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
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(rgb))
                cap.release()
            else:
                img = Image.open(str(sample.path))
                try:
                    frames.append(img.convert("RGB"))
                except Exception:
                    img.close()
                    raise

            if not frames:
                return sample

            # Classify each frame and aggregate
            votes = np.zeros(len(VIDEO_TYPE_LABELS))
            for img in frames:
                inputs = self._processor(
                    text=VIDEO_TYPE_LABELS,
                    images=img,
                    return_tensors="pt",
                    padding=True,
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    logits = outputs.logits_per_image.squeeze()
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    votes += probs

            votes /= len(frames)
            best_idx = int(np.argmax(votes))
            sample.quality_metrics.video_type = VIDEO_TYPE_SHORT[best_idx]
            sample.quality_metrics.video_type_confidence = float(votes[best_idx])
        except Exception as e:
            logger.warning("Video type classification failed: %s", e)
        return sample

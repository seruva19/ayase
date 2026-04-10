"""MaxVQA — Explainable VQA via Language-Prompted CLIP.

ACM MM 2023 Oral — language-prompted VQA using modified CLIP
for explainable quality scoring with MaxWell dataset.

GitHub: https://github.com/VQAssessment/ExplainableVQA

maxvqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MaxVQAModule(PipelineModule):
    name = "maxvqa"
    description = "MaxVQA explainable language-prompted VQA (ACM MM 2023)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._clip_model = None
        self._clip_processor = None
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: native maxvqa package
        try:
            import maxvqa
            self._model = maxvqa
            self._backend = "native"
            self._ml_available = True
            logger.info("MaxVQA (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: CLIP-based quality scoring
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir="models"
            ).to(device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir="models"
            )
            self._device = device
            self._backend = "clip"
            self._ml_available = True
            logger.info(f"MaxVQA (CLIP) initialised on {device}")
            return
        except (ImportError, Exception):
            pass

        logger.warning("MaxVQA: no backend available (install maxvqa or transformers)")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if self._backend == "native":
                score = self._process_native(sample)
            elif self._backend == "clip":
                score = self._process_clip(sample)
            else:
                return sample

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.maxvqa_score = score

        except Exception as e:
            logger.warning(f"MaxVQA failed for {sample.path}: {e}")

        return sample

    def _process_native(self, sample: Sample) -> Optional[float]:
        return float(self._model.predict(str(sample.path)))

    def _process_clip(self, sample: Sample) -> Optional[float]:
        """CLIP-based: cosine similarity with quality anchor texts."""
        import torch
        from PIL import Image

        quality_texts = [
            "a high quality, sharp, well-lit video frame",
            "a low quality, blurry, poorly-lit video frame",
        ]

        frames = self._extract_frames(sample)
        if not frames:
            return None

        scores = []
        with torch.no_grad():
            for frame in frames:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self._clip_processor(
                    text=quality_texts, images=pil_img,
                    return_tensors="pt", padding=True,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image.softmax(dim=-1)
                # P(high quality)
                scores.append(float(logits[0, 0].cpu()))

        return float(np.mean(scores))

    def _extract_frames(self, sample: Sample):
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

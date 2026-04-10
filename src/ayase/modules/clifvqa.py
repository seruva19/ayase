"""CLiF-VQA --- Human Feelings VQA via CLIP (2024).

Extracts human-feelings features from CLIP to simulate the HVS
perceptual process for quality assessment.  Quality is the weighted
similarity between video frames and quality-feeling text prompts
(comfort, clarity, warmth, vibrancy, harmony) vs negative feeling prompts.

clifvqa_score --- higher = better quality (0-1 range)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Human-feeling text prompts modelling HVS perceptual dimensions
_POSITIVE_FEELINGS = [
    "a comfortable and pleasant image",
    "a clear and sharp image",
    "a warm and inviting image",
    "a vibrant image with rich colors",
    "a harmonious and balanced image",
    "a natural and realistic image",
    "a relaxing and soothing image",
]
_NEGATIVE_FEELINGS = [
    "an uncomfortable and unpleasant image",
    "a blurry and unclear image",
    "a cold and uninviting image",
    "a dull image with washed out colors",
    "a chaotic and unbalanced image",
    "an unnatural and artificial image",
    "an irritating and disturbing image",
]


class CLiFVQAModule(PipelineModule):
    name = "clifvqa"
    description = "CLiF-VQA human feelings VQA via CLIP (2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = None
        self._pos_embeds = None
        self._neg_embeds = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = CLIPModel.from_pretrained(self._clip_model_name)
            self._processor = CLIPProcessor.from_pretrained(self._clip_model_name)
            self._model.to(self._device).eval()

            with torch.no_grad():
                pos_inputs = self._processor(
                    text=_POSITIVE_FEELINGS, return_tensors="pt", padding=True
                ).to(self._device)
                self._pos_embeds = self._model.get_text_features(**pos_inputs)
                self._pos_embeds = self._pos_embeds / self._pos_embeds.norm(
                    dim=-1, keepdim=True
                )

                neg_inputs = self._processor(
                    text=_NEGATIVE_FEELINGS, return_tensors="pt", padding=True
                ).to(self._device)
                self._neg_embeds = self._model.get_text_features(**neg_inputs)
                self._neg_embeds = self._neg_embeds / self._neg_embeds.norm(
                    dim=-1, keepdim=True
                )

            self._ml_available = True
            logger.info("CLiF-VQA (CLIP feelings) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("CLiF-VQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            score = self._compute_feeling_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.clifvqa_score = score
                logger.debug("CLiF-VQA for %s: %.4f", sample.path.name, score)
        except Exception as e:
            logger.warning("CLiF-VQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_feeling_score(self, sample: Sample) -> Optional[float]:
        """Score frames via CLIP similarity with feeling prompts."""
        import torch
        from PIL import Image

        frames = self._extract_frames(sample)
        if not frames:
            return None

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

        frame_scores = []
        with torch.no_grad():
            for pil_img in pil_frames:
                inputs = self._processor(
                    images=pil_img, return_tensors="pt"
                ).to(self._device)
                img_emb = self._model.get_image_features(**inputs)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

                # Per-dimension feeling similarity
                pos_sims = (img_emb @ self._pos_embeds.T).squeeze(0)  # (7,)
                neg_sims = (img_emb @ self._neg_embeds.T).squeeze(0)  # (7,)

                # Per-dimension score: sigmoid of difference
                dim_logits = (pos_sims - neg_sims) * 10.0
                dim_scores = torch.sigmoid(dim_logits)
                # Average across all feeling dimensions
                frame_scores.append(float(dim_scores.mean().item()))

        if not frame_scores:
            return None
        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(
                0, total - 1, min(self.subsample, total), dtype=int
            )
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

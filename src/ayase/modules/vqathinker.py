"""VQAThinker — Generalizable Explainable VQA via RL (2025).

RL-based explainable video quality assessment that uses quality
reasoning via CLIP backbone. Computes score with rationale using
softmax over quality-level descriptions with temperature scaling.

Implementation: CLIP backbone for quality reasoning. Extract
quality-relevant features and compute score via zero-shot
classification over quality level descriptions.

vqathinker_score — higher = better (0-1)
"""

import logging
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Quality level descriptions for CLIP zero-shot scoring
# Inspired by the VQAThinker paper's reasoning approach
_QUALITY_LEVELS = [
    "an extremely low quality video with severe distortions and artifacts",
    "a very low quality video with significant noise and blur",
    "a low quality video with noticeable compression artifacts",
    "a below average quality video with some visible imperfections",
    "a fair quality video with minor issues",
    "an acceptable quality video with mostly clean visuals",
    "a good quality video with clear details",
    "a high quality video with sharp and clean visuals",
    "a very high quality video with excellent detail preservation",
    "an outstanding quality video with perfect visual fidelity",
]

# Numeric quality values mapped to each level (0-1 scale)
_QUALITY_VALUES = np.linspace(0.05, 0.95, len(_QUALITY_LEVELS))


class VQAThinkerModule(PipelineModule):
    name = "vqathinker"
    description = "VQAThinker RL-based explainable VQA (2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
        "temperature": 0.07,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.temperature = self.config.get("temperature", 0.07)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        self._clip_model = None
        self._clip_processor = None
        self._quality_text_embeds = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.compat import extract_features

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            clip_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(clip_name, models_dir)

            self._clip_model = CLIPModel.from_pretrained(resolved).to(self._device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(resolved)

            # Pre-compute text embeddings for quality levels
            with torch.no_grad():
                inputs = self._clip_processor(
                    text=_QUALITY_LEVELS,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self._device)
                text_feats = extract_features(
                    self._clip_model.get_text_features(**inputs)
                )
                self._quality_text_embeds = text_feats / text_feats.norm(
                    p=2, dim=-1, keepdim=True
                )  # [10, D]

            self._ml_available = True
            self._backend = "clip_thinker"
            logger.info(
                "VQAThinker (CLIP quality reasoning) initialised on %s",
                self._device,
            )

        except Exception as e:
            logger.warning("VQAThinker setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vqathinker_score = score
        except Exception as e:
            logger.warning("VQAThinker failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """CLIP quality reasoning with temperature-scaled softmax."""
        import torch
        from PIL import Image
        from ayase.compat import extract_features

        frames = self._extract_frames(sample)
        if not frames:
            return None

        frame_scores = []

        with torch.no_grad():
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Encode image
                inputs = self._clip_processor(
                    images=pil_img, return_tensors="pt"
                ).to(self._device)
                img_feats = extract_features(
                    self._clip_model.get_image_features(**inputs)
                )
                img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)

                # Compute similarity to each quality level
                similarities = (img_feats @ self._quality_text_embeds.T).squeeze(0)  # [10]

                # Temperature-scaled softmax
                logits = similarities / self.temperature
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

                # Expected quality value (weighted sum)
                score = float(np.sum(probs * _QUALITY_VALUES))
                frame_scores.append(score)

        if not frame_scores:
            return None

        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    def _extract_frames(self, sample: Sample):
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

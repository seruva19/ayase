"""UGVQ — Unified Generated Video Quality (ACM TOMM 2024).

Multi-dimensional quality assessment for generated video content.
Evaluates visual fidelity, temporal consistency, and content
naturalness using CLIP embeddings.

Implementation: CLIP for multi-dimensional quality scoring via
quality-aware text prompts, frame embedding stability for temporal
consistency, and real-vs-generated discrimination.

ugvq_score — higher = better (0-1)
"""

import logging
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Quality assessment prompt sets for CLIP zero-shot scoring
_FIDELITY_PROMPTS = {
    "high": [
        "a high quality photo with sharp details",
        "a clear and well-defined image",
        "a photo with excellent visual quality",
        "a crisp and detailed image",
    ],
    "low": [
        "a blurry low quality photo",
        "a noisy and distorted image",
        "a photo with poor visual quality",
        "an image with visible artifacts",
    ],
}

_NATURALNESS_PROMPTS = {
    "natural": [
        "a natural realistic photograph",
        "a photo of a real scene",
        "a natural looking image",
    ],
    "generated": [
        "an AI generated image",
        "a synthetic computer generated image",
        "an artificial image with unnatural features",
    ],
}


class UGVQModule(PipelineModule):
    name = "ugvq"
    description = "UGVQ unified generated video quality (TOMM 2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        self._clip_model = None
        self._clip_processor = None
        self._fidelity_high_embeds = None
        self._fidelity_low_embeds = None
        self._natural_embeds = None
        self._generated_embeds = None

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

            # Pre-compute text embeddings for quality prompts
            with torch.no_grad():
                self._fidelity_high_embeds = self._encode_texts(
                    _FIDELITY_PROMPTS["high"]
                )
                self._fidelity_low_embeds = self._encode_texts(
                    _FIDELITY_PROMPTS["low"]
                )
                self._natural_embeds = self._encode_texts(
                    _NATURALNESS_PROMPTS["natural"]
                )
                self._generated_embeds = self._encode_texts(
                    _NATURALNESS_PROMPTS["generated"]
                )

            self._ml_available = True
            self._backend = "clip_ugvq"
            logger.info("UGVQ (CLIP multi-dim quality) initialised on %s", self._device)

        except Exception as e:
            logger.warning("UGVQ setup failed: %s", e)

    def _encode_texts(self, texts: List[str]):
        """Encode text prompts into normalized CLIP embeddings."""
        import torch
        from ayase.compat import extract_features

        inputs = self._clip_processor(text=texts, return_tensors="pt", padding=True).to(self._device)
        feats = extract_features(self._clip_model.get_text_features(**inputs))
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats  # [N, D]

    def _encode_image(self, pil_img):
        """Encode a PIL image into a normalized CLIP embedding."""
        import torch
        from ayase.compat import extract_features

        inputs = self._clip_processor(images=pil_img, return_tensors="pt").to(self._device)
        feats = extract_features(self._clip_model.get_image_features(**inputs))
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats  # [1, D]

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.ugvq_score = score
        except Exception as e:
            logger.warning("UGVQ failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """Multi-dimensional CLIP-based quality assessment."""
        import torch
        from PIL import Image

        frames = self._extract_frames(sample)
        if not frames:
            return None

        frame_embeddings = []
        fidelity_scores = []
        naturalness_scores = []

        with torch.no_grad():
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                img_embed = self._encode_image(pil_img)  # [1, D]
                frame_embeddings.append(img_embed)

                # Dimension 1: Visual Fidelity
                # Softmax over quality-positive vs quality-negative prompts
                high_sim = (img_embed @ self._fidelity_high_embeds.T).mean().item()
                low_sim = (img_embed @ self._fidelity_low_embeds.T).mean().item()
                fidelity = 1.0 / (1.0 + np.exp(-(high_sim - low_sim) * 10.0))
                fidelity_scores.append(fidelity)

                # Dimension 2: Content Naturalness
                nat_sim = (img_embed @ self._natural_embeds.T).mean().item()
                gen_sim = (img_embed @ self._generated_embeds.T).mean().item()
                naturalness = 1.0 / (1.0 + np.exp(-(nat_sim - gen_sim) * 10.0))
                naturalness_scores.append(naturalness)

        # Dimension 3: Temporal Consistency (frame embedding stability)
        temporal_score = 1.0
        if len(frame_embeddings) > 1:
            embeddings = torch.cat(frame_embeddings, dim=0)  # [T, D]
            consec_sims = []
            for i in range(embeddings.size(0) - 1):
                sim = (embeddings[i] @ embeddings[i + 1]).item()
                consec_sims.append(sim)
            # Map cosine similarity to quality score
            mean_sim = np.mean(consec_sims)
            temporal_score = float(np.clip((mean_sim - 0.5) * 2.0, 0.0, 1.0))

        # Unified score: weighted combination of dimensions
        avg_fidelity = float(np.mean(fidelity_scores))
        avg_naturalness = float(np.mean(naturalness_scores))

        score = (
            0.40 * avg_fidelity
            + 0.30 * temporal_score
            + 0.30 * avg_naturalness
        )

        return float(np.clip(score, 0.0, 1.0))

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

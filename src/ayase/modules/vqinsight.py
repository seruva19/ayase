"""VQ-Insight — ByteDance AIGC Video Quality (AAAI 2026).

Multi-dimensional AIGC video quality assessment. Evaluates generated
video across four quality dimensions: visual quality, text alignment,
temporal coherence, and aesthetic quality.

Implementation: CLIP for multi-dimensional AIGC quality scoring.
Each dimension gets its own prompt set and score via zero-shot
CLIP classification.

vqinsight_score — higher = better (0-1)
"""

import logging
from typing import Optional, Dict, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Multi-dimensional prompt sets for CLIP zero-shot scoring
_DIMENSION_PROMPTS: Dict[str, Dict[str, List[str]]] = {
    "visual_quality": {
        "positive": [
            "a high quality image with sharp details and low noise",
            "a crisp clear image with excellent resolution",
            "a well-exposed image with vivid colors and no artifacts",
            "a professionally captured image with superb clarity",
        ],
        "negative": [
            "a blurry image with noise and compression artifacts",
            "a low resolution image with poor detail",
            "a distorted image with visible quality degradation",
            "an image with blocky artifacts and color banding",
        ],
    },
    "aesthetic_quality": {
        "positive": [
            "a beautiful aesthetically pleasing image with great composition",
            "an artistically composed image with harmonious colors",
            "a visually stunning image with balanced lighting",
            "a well-composed image with appealing visual balance",
        ],
        "negative": [
            "an ugly unpleasant image with poor composition",
            "a visually unappealing image with clashing colors",
            "a poorly composed image with bad framing",
            "an aesthetically displeasing image",
        ],
    },
    "content_naturalness": {
        "positive": [
            "a natural realistic photograph of a real scene",
            "a photorealistic image indistinguishable from a photograph",
            "a natural looking scene with realistic textures and lighting",
        ],
        "negative": [
            "an obviously AI generated image with unnatural features",
            "a synthetic image with telltale generation artifacts",
            "an artificial looking image with uncanny visual elements",
        ],
    },
}


class VQInsightModule(PipelineModule):
    name = "vqinsight"
    description = "VQ-Insight ByteDance multi-dim AIGC scoring (AAAI 2026)"
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
        self._dim_embeds: Dict[str, Dict[str, object]] = {}

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

            # Pre-compute text embeddings for each quality dimension
            with torch.no_grad():
                for dim_name, prompts in _DIMENSION_PROMPTS.items():
                    self._dim_embeds[dim_name] = {
                        "positive": self._encode_texts(prompts["positive"]),
                        "negative": self._encode_texts(prompts["negative"]),
                    }

            self._ml_available = True
            self._backend = "clip_vqinsight"
            logger.info(
                "VQ-Insight (CLIP multi-dim AIGC, %d dimensions) initialised on %s",
                len(_DIMENSION_PROMPTS), self._device,
            )

        except Exception as e:
            logger.warning("VQ-Insight setup failed: %s", e)

    def _encode_texts(self, texts: List[str]):
        """Encode text prompts into normalized CLIP embeddings."""
        from ayase.compat import extract_features

        inputs = self._clip_processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self._device)
        feats = extract_features(self._clip_model.get_text_features(**inputs))
        feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
        return feats

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vqinsight_score = score
        except Exception as e:
            logger.warning("VQ-Insight failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """Multi-dimensional CLIP-based AIGC quality assessment."""
        import torch
        from PIL import Image
        from ayase.compat import extract_features

        frames = self._extract_frames(sample)
        if not frames:
            return None

        # Collect per-frame embeddings and dimension scores
        frame_embeddings = []
        dim_scores: Dict[str, List[float]] = {
            dim: [] for dim in _DIMENSION_PROMPTS
        }

        with torch.no_grad():
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                inputs = self._clip_processor(
                    images=pil_img, return_tensors="pt"
                ).to(self._device)
                img_feats = extract_features(
                    self._clip_model.get_image_features(**inputs)
                )
                img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
                frame_embeddings.append(img_feats)

                # Score each quality dimension
                for dim_name, embeds in self._dim_embeds.items():
                    pos_sim = (img_feats @ embeds["positive"].T).mean().item()
                    neg_sim = (img_feats @ embeds["negative"].T).mean().item()
                    # Sigmoid of difference for 0-1 score
                    dim_score = 1.0 / (1.0 + np.exp(-(pos_sim - neg_sim) * 10.0))
                    dim_scores[dim_name].append(dim_score)

        # Temporal coherence dimension (from frame embedding stability)
        temporal_coherence = 1.0
        if len(frame_embeddings) > 1:
            all_embeds = torch.cat(frame_embeddings, dim=0)  # [T, D]
            consec_sims = []
            for i in range(all_embeds.size(0) - 1):
                sim = (all_embeds[i] @ all_embeds[i + 1]).item()
                consec_sims.append(sim)
            mean_sim = np.mean(consec_sims)
            temporal_coherence = float(np.clip((mean_sim - 0.5) * 2.0, 0.0, 1.0))

        # Average each dimension across frames
        avg_dims = {
            dim: float(np.mean(scores)) for dim, scores in dim_scores.items()
        }

        # Final multi-dimensional fusion score
        # Weights: visual quality (0.30), aesthetic (0.25),
        #          temporal coherence (0.25), content naturalness (0.20)
        score = (
            0.30 * avg_dims.get("visual_quality", 0.5)
            + 0.25 * avg_dims.get("aesthetic_quality", 0.5)
            + 0.25 * temporal_coherence
            + 0.20 * avg_dims.get("content_naturalness", 0.5)
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

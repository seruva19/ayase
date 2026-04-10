"""ChronoMagic-Bench module — NeurIPS 2024 (arXiv:2406.18522).

Two metrics for time-lapse video quality:
  - **MTScore** (Metamorphic Temporal): smoothness of temporal progression
  - **CHScore** (Chrono-Hallucination): detection of temporal hallucinations

Backend tier:
  1. **CLIP** — CLIP ViT-B/32 temporal embedding analysis

Video-only: returns None for images.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)


class ChronoMagicModule(PipelineModule):
    name = "chronomagic"
    description = "ChronoMagic-Bench MTScore + CHScore (CLIP)"
    default_config = {
        "subsample": 16,
        "hallucination_threshold": 2.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._ml_available = False
        self._clip_model = None
        self._clip_processor = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: CLIP
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            models_dir = self.config.get("models_dir", "models")

            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=models_dir,
            ).to(self._device)
            self._clip_model.eval()
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir=models_dir,
            )
            self._ml_available = True
            logger.info("ChronoMagic loaded CLIP on %s", self._device)
            return
        except Exception as e:
            logger.info("CLIP unavailable for ChronoMagic: %s", e)

        logger.warning("ChronoMagic unavailable: CLIP not installed")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        if not sample.is_video:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            mt_score, ch_score = self._compute_clip(sample)

            if mt_score is not None:
                sample.quality_metrics.chronomagic_mt_score = mt_score
            if ch_score is not None:
                sample.quality_metrics.chronomagic_ch_score = ch_score

        except Exception as e:
            logger.warning("ChronoMagic processing failed: %s", e)

        return sample

    # ------------------------------------------------------------------ #
    # Shared frame extraction                                              #
    # ------------------------------------------------------------------ #

    def _extract_frames(self, sample: Sample) -> list:
        num_frames = self.config.get("subsample", 16)
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 4:
            cap.release()
            return []

        indices = list(range(0, total, max(1, total // num_frames)))[:num_frames]
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()
        return frames

    # ------------------------------------------------------------------ #
    # CLIP temporal analysis                                               #
    # ------------------------------------------------------------------ #

    def _compute_clip(self, sample: Sample) -> Tuple[Optional[float], Optional[float]]:
        import torch
        from PIL import Image

        frames = self._extract_frames(sample)
        if len(frames) < 4:
            return None, None

        # Get CLIP embeddings for each frame
        embeddings = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            inputs = self._clip_processor(images=pil_img, return_tensors="pt").to(self._device)
            with torch.no_grad():
                features = extract_features(self._clip_model.get_image_features(**inputs))
                features = features / features.norm(dim=-1, keepdim=True)
                embeddings.append(features.cpu().numpy().flatten())

        embeddings = np.array(embeddings)  # [T, D]

        # MTScore: temporal gradient smoothness
        gradients = np.diff(embeddings, axis=0)  # [T-1, D]
        gradient_norms = np.linalg.norm(gradients, axis=1)  # [T-1]

        # Smoothness = low variance of gradient norms (consistent rate of change)
        if gradient_norms.std() > 1e-8:
            gradient_cv = gradient_norms.std() / (gradient_norms.mean() + 1e-8)
            gradient_smoothness = 1.0 / (1.0 + gradient_cv)
        else:
            gradient_smoothness = 1.0

        # Progression quality: monotonic progression in embedding space
        cosine_sims = []
        for i in range(len(embeddings) - 1):
            sim = float(np.dot(embeddings[i], embeddings[i + 1]))
            cosine_sims.append(sim)
        cosine_sims = np.array(cosine_sims)

        # Good progression = consistently high but not perfect similarity
        progression_quality = float(np.mean(cosine_sims))
        progression_quality = min(max(progression_quality, 0.0), 1.0)

        mt_score = 0.5 * gradient_smoothness + 0.5 * progression_quality

        # CHScore: detect abrupt embedding spikes (hallucination frames)
        threshold = self.config.get("hallucination_threshold", 2.0)
        mean_norm = gradient_norms.mean()
        std_norm = gradient_norms.std() + 1e-8
        hallucination_mask = gradient_norms > (mean_norm + threshold * std_norm)
        hallucination_frac = float(np.mean(hallucination_mask))
        ch_score = 1.0 - hallucination_frac  # Higher = fewer hallucinations

        return (
            float(np.clip(mt_score, 0.0, 1.0)),
            float(np.clip(ch_score, 0.0, 1.0)),
        )

"""ChronoMagic-Bench module — NeurIPS 2024 (arXiv:2406.18522).

Two metrics for time-lapse video quality:
  - **MTScore** (Metamorphic Temporal): smoothness of temporal progression
  - **CHScore** (Chrono-Hallucination): detection of temporal hallucinations

Backend tiers:
  1. **CLIP** — CLIP ViT-B/32 temporal embedding analysis
  2. **Heuristic** — Pixel-level temporal gradient + histogram correlation

Video-only: returns None for images.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ChronoMagicModule(PipelineModule):
    name = "chronomagic"
    description = "ChronoMagic-Bench MTScore + CHScore (CLIP / heuristic)"
    default_config = {
        "subsample": 16,
        "hallucination_threshold": 2.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "heuristic"
        self._clip_model = None
        self._clip_processor = None
        self._device = "cpu"

    def setup(self) -> None:
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
            self._backend = "clip"
            logger.info("ChronoMagic loaded CLIP on %s", self._device)
            return
        except Exception as e:
            logger.info("CLIP unavailable for ChronoMagic: %s", e)

        logger.info("ChronoMagic using heuristic backend")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            if self._backend == "clip":
                mt_score, ch_score = self._compute_clip(sample)
            else:
                mt_score, ch_score = self._compute_heuristic(sample)

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
    # Tier 1: CLIP temporal analysis                                       #
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
                features = self._clip_model.get_image_features(**inputs)
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

    # ------------------------------------------------------------------ #
    # Tier 2: Heuristic (pixel-level)                                      #
    # ------------------------------------------------------------------ #

    def _compute_heuristic(self, sample: Sample) -> Tuple[Optional[float], Optional[float]]:
        frames = self._extract_frames(sample)
        if len(frames) < 4:
            return None, None

        # Convert to grayscale
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]

        # Temporal pixel differences
        diffs = []
        for i in range(len(grays) - 1):
            diff = np.abs(grays[i + 1] - grays[i]).mean()
            diffs.append(diff)
        diffs = np.array(diffs)

        # MTScore: smoothness of temporal gradients
        if diffs.std() > 1e-6:
            cv = diffs.std() / (diffs.mean() + 1e-6)
            gradient_smoothness = 1.0 / (1.0 + cv)
        else:
            gradient_smoothness = 1.0

        # Histogram correlation for progression quality
        hist_sims = []
        for i in range(len(grays) - 1):
            h1 = cv2.calcHist([grays[i].astype(np.uint8)], [0], None, [64], [0, 256])
            h2 = cv2.calcHist([grays[i + 1].astype(np.uint8)], [0], None, [64], [0, 256])
            corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            hist_sims.append(max(corr, 0.0))
        progression_quality = float(np.mean(hist_sims)) if hist_sims else 0.5

        mt_score = 0.5 * gradient_smoothness + 0.5 * progression_quality

        # CHScore: detect abrupt changes (hallucination)
        threshold = self.config.get("hallucination_threshold", 2.0)
        mean_diff = diffs.mean()
        std_diff = diffs.std() + 1e-6
        hallucination_mask = diffs > (mean_diff + threshold * std_diff)
        ch_score = 1.0 - float(np.mean(hallucination_mask))

        return (
            float(np.clip(mt_score, 0.0, 1.0)),
            float(np.clip(ch_score, 0.0, 1.0)),
        )

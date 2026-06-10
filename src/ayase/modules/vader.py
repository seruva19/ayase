"""VADER --- Video Diffusion Alignment via Reward Gradients (ICLR 2025).

GitHub: https://github.com/mihirp1998/VADER

VADER is a method for fine-tuning video diffusion models via reward
gradients.  It uses reward models internally --- most notably HPS v2
(Human Preference Score) and LAION aesthetics.

This module extracts the HPS v2 reward signal that VADER relies on.
Backend priority:
  1. HPS v2 via ``hpsv2`` package (ImageReward/HPS-v2)
  2. CLIP aesthetic scoring as lightweight fallback (cosine similarity
     of frame embeddings with aesthetic quality prompts)

vader_score --- higher = better (0-1 range, normalised)
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    cached_clip_text_features,
    media_state_key,
)

logger = logging.getLogger(__name__)

# Fallback: aesthetic quality prompts for CLIP scoring
_AESTHETIC_POS = [
    "a beautiful high quality image with great aesthetics",
    "an aesthetically pleasing image with harmonious composition",
    "a visually stunning image with excellent lighting and color",
    "a professional photograph with superb visual quality",
]
_AESTHETIC_NEG = [
    "an ugly low quality image with poor aesthetics",
    "a visually unpleasant image with bad composition",
    "a dull image with poor lighting and washed out colors",
    "an amateur photograph with terrible visual quality",
]


class VADERModule(PipelineModule):
    name = "vader"
    description = "VADER HPS v2 reward signal (ICLR 2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-large-patch14",
    }
    metric_groups = {
        "vader_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-large-patch14"
        )
        self._ml_available = False
        self._backend = None
        self._model = None
        self._processor = None
        self._device = "cpu"
        # CLIP fallback pre-encoded prompts
        self._aes_pos_embeds = None
        self._aes_neg_embeds = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: HPS v2 via hpsv2 package
        if self._try_hpsv2_setup():
            return
        # Tier 2: CLIP aesthetic scoring fallback
        self._try_clip_aesthetic_setup()

    def _try_hpsv2_setup(self) -> bool:
        """Try HPS v2 via the ``hpsv2`` package."""
        try:
            import hpsv2
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            # hpsv2 provides a scoring function directly
            self._model = hpsv2
            self._ml_available = True
            self._backend = "hpsv2"
            logger.info("VADER (HPS v2 package) initialised")
            return True
        except (ImportError, Exception) as e:
            logger.debug("HPS v2 package not available: %s", e)
            return False

    def _try_clip_aesthetic_setup(self) -> bool:
        """Fallback: CLIP aesthetic prompt-based scoring."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self._clip_model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._model, self._processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )

            self._aes_pos_embeds = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _AESTHETIC_POS,
                model_key=self._clip_model_name,
                device=self._device,
            )
            self._aes_neg_embeds = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                _AESTHETIC_NEG,
                model_key=self._clip_model_name,
                device=self._device,
            )

            self._ml_available = True
            self._backend = "clip_aesthetic"
            logger.info(
                "VADER (CLIP aesthetic fallback) initialised on %s", self._device
            )
            return True
        except (ImportError, Exception) as e:
            logger.warning("VADER setup failed (no backend available): %s", e)
            return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            if self._backend == "hpsv2":
                score = self._process_hpsv2(sample)
            else:
                score = self._process_clip_aesthetic(sample)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.vader_score = score
            logger.debug("VADER for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("VADER failed for %s: %s", sample.path, e)
        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        if self._backend != "clip_aesthetic":
            return super().process_batch(samples)

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            for sample in samples:
                frames = self._extract_frames(sample)
                if not frames:
                    continue
                prepared.append(sample)
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self.subsample, media_state_key(sample.path)))

            if not prepared:
                return samples

            feature_groups = cached_clip_image_feature_groups(
                self,
                self._model,
                self._processor,
                image_groups,
                model_key=self._clip_model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for sample, image_features in zip(prepared, feature_groups):
                score = self._score_clip_features(image_features)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vader_score = score
        except Exception as e:
            logger.warning("VADER batch failed: %s", e)

        return samples

    # ------------------------------------------------------------------
    # Tier 1: HPS v2 scoring
    # ------------------------------------------------------------------

    def _process_hpsv2(self, sample: Sample) -> Optional[float]:
        """Score using HPS v2 package.  Returns normalised 0-1 score."""

        frames = self._extract_frames(sample)
        if not frames:
            return None

        pil_frames = arrays_to_pil(frames)

        # hpsv2.score expects (images, prompt) -> list of floats
        # Use a generic quality prompt as the conditioning text
        prompt = "a high quality image"
        scores = []
        for pil_img in pil_frames:
            try:
                result = self._model.score(pil_img, prompt)
                # hpsv2.score returns a list; take first element
                if isinstance(result, (list, tuple)):
                    raw = float(result[0])
                else:
                    raw = float(result)
                # HPS v2 raw scores are typically in ~0.20-0.32 range
                # Normalise to 0-1 using observed range
                normalised = float(np.clip((raw - 0.18) / 0.16, 0.0, 1.0))
                scores.append(normalised)
            except Exception as e:
                logger.debug("HPS v2 frame scoring failed: %s", e)

        if not scores:
            return None
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Tier 2: CLIP aesthetic fallback
    # ------------------------------------------------------------------

    def _process_clip_aesthetic(self, sample: Sample) -> Optional[float]:
        """CLIP aesthetic scoring: positive vs negative prompt similarity."""

        frames = self._extract_frames(sample)
        if not frames:
            return None

        image_features = cached_clip_image_features(
            self,
            self._model,
            self._processor,
            arrays_to_pil(frames),
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        return self._score_clip_features(image_features)

    def _score_clip_features(self, image_features) -> Optional[float]:
        if image_features is None or image_features.size(0) == 0:
            return None

        frame_scores = []
        for img_embed in image_features:
            img_embed = img_embed.unsqueeze(0)
            # Cosine similarity with positive and negative aesthetic prompts
            pos_sim = (img_embed @ self._aes_pos_embeds.T).mean().item()
            neg_sim = (img_embed @ self._aes_neg_embeds.T).mean().item()

            # Reward = difference, then sigmoid to [0, 1]
            reward_logit = (pos_sim - neg_sim) * 10.0
            reward = 1.0 / (1.0 + np.exp(-reward_logit))
            frame_scores.append(reward)

        if not frame_scores:
            return None
        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("VADER frame loading failed for %s: %s", sample.path, e)
            return []

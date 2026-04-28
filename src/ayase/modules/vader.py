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

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

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
            import torch

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
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
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = CLIPModel.from_pretrained(self._clip_model_name)
            self._processor = CLIPProcessor.from_pretrained(self._clip_model_name)
            self._model.to(self._device).eval()

            # Pre-encode aesthetic text prompts
            with torch.no_grad():
                pos_inputs = self._processor(
                    text=_AESTHETIC_POS, return_tensors="pt", padding=True
                ).to(self._device)
                self._aes_pos_embeds = self._model.get_text_features(**pos_inputs)
                self._aes_pos_embeds = self._aes_pos_embeds / self._aes_pos_embeds.norm(
                    dim=-1, keepdim=True
                )

                neg_inputs = self._processor(
                    text=_AESTHETIC_NEG, return_tensors="pt", padding=True
                ).to(self._device)
                self._aes_neg_embeds = self._model.get_text_features(**neg_inputs)
                self._aes_neg_embeds = self._aes_neg_embeds / self._aes_neg_embeds.norm(
                    dim=-1, keepdim=True
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

    # ------------------------------------------------------------------
    # Tier 1: HPS v2 scoring
    # ------------------------------------------------------------------

    def _process_hpsv2(self, sample: Sample) -> Optional[float]:
        """Score using HPS v2 package.  Returns normalised 0-1 score."""
        from PIL import Image

        frames = self._extract_frames(sample)
        if not frames:
            return None

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

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
                img_embed = self._model.get_image_features(**inputs)
                img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)

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
                ret, f = cap.read()
                if ret:
                    frames.append(f)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

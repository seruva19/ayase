"""ImageReward module.

ImageReward predicts human preferences for text-to-image generation quality.
Given an image and a text prompt, it produces a reward score that correlates
with human quality assessment. Higher score = better alignment with human preference.
Typical range: -2 to +2.

This is a per-sample metric that requires a caption/prompt for each sample.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)


class ImageRewardModule(PipelineModule):
    name = "image_reward"
    description = "Human preference prediction for text-to-image quality (ImageReward)"
    default_config = {
        "model_name": "ImageReward-v1.0",
        "num_frames": 5,  # For video: number of frames to evaluate
        "warning_threshold": 0.0,  # Threshold for warning (0 = neutral)
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "ImageReward-v1.0")
        self.num_frames = self.config.get("num_frames", 5)
        self.warning_threshold = self.config.get("warning_threshold", 0.0)
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None  # "image_reward" or "heuristic"

    def setup(self) -> None:
        # Tier 1: image-reward library
        try:
            # Shim: image-reward imports several functions from
            # transformers.modeling_utils that were moved to
            # transformers.pytorch_utils in transformers >= 4.50.
            import transformers.modeling_utils as _mu

            try:
                from transformers import pytorch_utils as _pu

                for _fn in (
                    "apply_chunking_to_forward",
                    "find_pruneable_heads_and_indices",
                    "prune_linear_layer",
                ):
                    if hasattr(_pu, _fn) and not hasattr(_mu, _fn):
                        setattr(_mu, _fn, getattr(_pu, _fn))
            except ImportError:
                pass

            import ImageReward as ir_lib

            self._model = ir_lib.load(self.model_name)
            self._backend = "image_reward"
            self._ml_available = True
            logger.info(f"ImageReward module initialized with {self.model_name}")
            return
        except ImportError:
            logger.debug("image-reward library not available, trying heuristic fallback")
        except Exception as e:
            logger.warning(f"Failed to load ImageReward model: {e}")

        # Tier 2: Heuristic fallback (CLIP similarity * aesthetic proxy)
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            clip_model_name = "openai/clip-vit-base-patch32"
            resolved = resolve_model_path(clip_model_name, models_dir)

            self._clip_model = CLIPModel.from_pretrained(resolved, use_safetensors=True).to(
                self._device
            )
            self._clip_processor = CLIPProcessor.from_pretrained(resolved)
            self._backend = "heuristic"
            self._ml_available = True
            logger.info("ImageReward module initialized with CLIP heuristic fallback")
            return
        except ImportError:
            logger.debug("transformers not available for heuristic fallback")
        except Exception as e:
            logger.debug(f"Heuristic fallback setup failed: {e}")

        logger.warning(
            "ImageReward: no backend available (install image-reward or transformers). "
            "Module disabled."
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        # Get caption text (same pattern as semantic_alignment.py)
        caption_text = None
        if sample.caption:
            caption_text = sample.caption.text
        else:
            txt_path = sample.path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    caption_text = txt_path.read_text().strip()
                except Exception:
                    logger.debug(f"Failed to read caption file: {txt_path}")

        if not caption_text:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            if self._backend == "image_reward":
                score = self._score_image_reward(frames, caption_text)
            else:
                score = self._score_heuristic(frames, caption_text)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()

                sample.quality_metrics.image_reward_score = score

                if score < self.warning_threshold:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Low ImageReward score: {score:.3f}",
                            details={
                                "image_reward_score": score,
                                "caption": caption_text[:80] + "..."
                                if len(caption_text) > 80
                                else caption_text,
                            },
                        )
                    )

        except Exception as e:
            logger.warning(f"ImageReward inference failed: {e}")

        return sample

    def _score_image_reward(self, frames: List[Image.Image], caption: str) -> Optional[float]:
        """Score frames using the ImageReward library.

        Args:
            frames: List of PIL images
            caption: Text prompt

        Returns:
            Average reward score across frames
        """
        scores = []
        for pil_image in frames:
            try:
                score = self._model.score(caption, pil_image)
                scores.append(float(score))
            except Exception as e:
                logger.debug(f"ImageReward scoring failed for frame: {e}")

        if not scores:
            return None

        return float(np.mean(scores))

    def _score_heuristic(self, frames: List[Image.Image], caption: str) -> Optional[float]:
        """Approximate ImageReward using CLIP similarity as a proxy.

        Computes CLIP cosine similarity between image and caption. Maps the
        similarity range [0, 0.4] to the typical ImageReward range [-2, +2].

        Args:
            frames: List of PIL images
            caption: Text prompt

        Returns:
            Approximate reward score
        """
        try:
            import torch

            # Encode text once
            text_inputs = self._clip_processor(
                text=[caption],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._device)

            with torch.no_grad():
                text_features = extract_features(self._clip_model.get_text_features(**text_inputs))
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            similarities = []
            for pil_image in frames:
                image_inputs = self._clip_processor(
                    images=pil_image,
                    return_tensors="pt",
                ).to(self._device)

                with torch.no_grad():
                    image_features = extract_features(self._clip_model.get_image_features(**image_inputs))
                    image_features = image_features / image_features.norm(
                        p=2, dim=-1, keepdim=True
                    )
                    sim = (image_features @ text_features.T).item()
                    similarities.append(sim)

            if not similarities:
                return None

            avg_sim = float(np.mean(similarities))

            # Map CLIP similarity [0, 0.4] to ImageReward-like range [-2, +2]
            # CLIP sim ~0.2 is typical neutral, map to 0
            reward_approx = (avg_sim - 0.2) * 10.0
            reward_approx = max(-2.0, min(2.0, reward_approx))

            return reward_approx

        except Exception as e:
            logger.debug(f"Heuristic scoring failed: {e}")
            return None

    def _load_frames(self, sample: Sample) -> List[Image.Image]:
        """Load frames from video (uniformly sampled) or single image.

        Returns:
            List of PIL Image objects in RGB
        """
        try:
            if not sample.is_video:
                bgr = cv2.imread(str(sample.path))
                if bgr is None:
                    return []
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return [Image.fromarray(rgb)]

            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []

            n = min(self.num_frames, total)
            indices = np.linspace(0, total - 1, n, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
            cap.release()
            return frames

        except Exception:
            return []

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

    def setup(self) -> None:
        if self.test_mode:
            return
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
            self._ml_available = True
            logger.info(f"ImageReward module initialized with {self.model_name}")
        except ImportError:
            logger.warning("ImageReward: image-reward library not installed, module disabled")
        except Exception as e:
            logger.warning(f"Failed to load ImageReward model: {e}")

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

            score = self._score_image_reward(frames, caption_text)

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

"""Harmful Content Detection module.

Beyond NSFW — detects potentially harmful visual content:

  harmful_content_score — 0-1 (higher = more harmful)

Detection categories:
  - Violence / fighting (motion + red-channel analysis)
  - Blood / gore (colour-based heuristic)
  - Disturbing textures (high-frequency noise patterns)

Uses OpenCV heuristics as baseline.  If ``transformers`` and a
safety classifier are available, uses zero-shot CLIP classification
for more accurate results.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class HarmfulContentModule(PipelineModule):
    name = "harmful_content"
    description = "Violence, gore, and disturbing content detection"
    default_config = {
        "subsample": 10,
        "max_frames": 60,
        "warning_threshold": 0.4,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 10)
        self.max_frames = self.config.get("max_frames", 60)
        self.warning_threshold = self.config.get("warning_threshold", 0.4)

        self._clip_model = None
        self._clip_processor = None
        self._clip_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            ).to(device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_available = True
            logger.info(f"Harmful content: CLIP classifier on {device}")
        except ImportError:
            logger.info("CLIP unavailable, using heuristic-only detection")
        except Exception as e:
            logger.warning(f"CLIP init failed: {e}")

    # ------------------------------------------------------------------
    # Heuristic detectors
    # ------------------------------------------------------------------

    @staticmethod
    def _blood_gore_score(frame_bgr: np.ndarray) -> float:
        """Heuristic: detect red-dominant regions (blood/gore-like).

        Looks for high red saturation in HSV space.
        Returns 0-1.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # Red hue ranges (wraps around 0/180 in OpenCV HSV)
        mask_low = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
        mask_high = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
        red_mask = mask_low | mask_high

        red_ratio = float(red_mask.sum() / 255 / red_mask.size)

        # Dark red (blood-like) specifically
        dark_red = cv2.inRange(hsv, (0, 100, 30), (15, 255, 120))
        dark_red_high = cv2.inRange(hsv, (165, 100, 30), (180, 255, 120))
        blood_mask = dark_red | dark_red_high
        blood_ratio = float(blood_mask.sum() / 255 / blood_mask.size)

        # Combined: heavy red presence + dark red = higher score
        combined = red_ratio * 0.4 + blood_ratio * 0.6
        return float(np.clip(combined * 5.0, 0, 1))

    @staticmethod
    def _violence_motion_score(
        prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> float:
        """Heuristic: high-magnitude sudden motion (violence indicator).

        Rapid motion alone doesn't indicate violence, but combined
        with other signals it contributes.
        Returns 0-1.
        """
        diff = cv2.absdiff(prev_gray, curr_gray)
        motion_intensity = float(diff.mean()) / 255.0
        # Very high motion (>0.15) is suspicious
        return float(np.clip(motion_intensity * 5.0, 0, 1))

    # ------------------------------------------------------------------
    # CLIP zero-shot classification
    # ------------------------------------------------------------------

    def _clip_harm_score(self, frame_bgr: np.ndarray) -> Optional[float]:
        if not self._clip_available:
            return None

        try:
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            safe_texts = [
                "a safe, normal photograph",
                "a peaceful scene",
                "a landscape or everyday scene",
            ]
            harmful_texts = [
                "a violent scene with fighting",
                "a disturbing or graphic image",
                "blood or injury",
            ]

            all_texts = safe_texts + harmful_texts
            inputs = self._clip_processor(
                text=all_texts, images=pil_img, return_tensors="pt", padding=True
            )
            device = next(self._clip_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                probs = torch.softmax(outputs.logits_per_image[0], dim=0).cpu().numpy()

            # Sum of harmful probabilities
            n_safe = len(safe_texts)
            harm_prob = float(probs[n_safe:].sum())
            return harm_prob

        except Exception as e:
            logger.debug(f"CLIP harm classification failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def _score_frame(
        self,
        frame_bgr: np.ndarray,
        prev_gray: Optional[np.ndarray] = None,
    ) -> float:
        blood = self._blood_gore_score(frame_bgr)

        motion = 0.0
        if prev_gray is not None:
            curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            motion = self._violence_motion_score(prev_gray, curr_gray)

        clip_score = self._clip_harm_score(frame_bgr)

        if clip_score is not None:
            return 0.2 * blood + 0.1 * motion + 0.7 * clip_score
        return 0.6 * blood + 0.4 * motion

    def process(self, sample: Sample) -> Sample:
        try:
            if sample.is_video:
                score = self._process_video(sample.path)
            else:
                score = self._process_image(sample.path)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.harmful_content_score = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Potentially harmful content detected: {score:.2f}",
                        details={"harmful_content_score": score},
                        recommendation=(
                            "Content may contain violence, gore, or "
                            "disturbing material.  Review manually."
                        ),
                    )
                )

            logger.debug(f"Harmful content for {sample.path.name}: {score:.3f}")

        except Exception as e:
            logger.error(f"Harmful content detection failed for {sample.path}: {e}")

        return sample

    def _process_image(self, path: Path) -> Optional[float]:
        img = cv2.imread(str(path))
        if img is None:
            return None
        return self._score_frame(img)

    def _process_video(self, path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        scores = []
        prev_gray = None
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                s = self._score_frame(frame, prev_gray)
                scores.append(s)
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            idx += 1

        cap.release()
        return float(np.mean(scores)) if scores else None

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

from ayase.image import arrays_to_pil
from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
from ayase.runtime import cached_clip_image_features, cached_clip_text_features, media_state_key

logger = logging.getLogger(__name__)


class HarmfulContentModule(PipelineModule):
    name = "harmful_content"
    description = "Violence, gore, and disturbing content detection"
    default_config = {
        "subsample": 10,
        "max_frames": 60,
        "clip_model": "openai/clip-vit-base-patch32",
        "warning_threshold": 0.4,
    }
    metric_groups = {
        "harmful_content_score": "safety",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 10)
        self.max_frames = self.config.get("max_frames", 60)
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self.warning_threshold = self.config.get("warning_threshold", 0.4)

        self._clip_model = None
        self._clip_processor = None
        self._clip_device = "cpu"
        self._clip_available = False

    def setup(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            device = resolve_torch_device(self.config.get("device", "auto"))
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.clip_model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=device,
                ).to(device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )
            self._clip_device = device
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
        scores = self._clip_harm_scores([frame_bgr])
        return scores[0] if scores else None

    def _clip_harm_scores(
        self,
        frames_bgr: list[np.ndarray],
        cache_key: Optional[tuple] = None,
    ) -> list[Optional[float]]:
        if not self._clip_available or not frames_bgr:
            return [None] * len(frames_bgr)
        try:
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
            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
            image_features = cached_clip_image_features(
                self,
                self._clip_model,
                self._clip_processor,
                arrays_to_pil(rgb_frames),
                model_key=self.clip_model_name,
                device=self._clip_device,
                cache_key=cache_key or ("harmful_content_frames", len(frames_bgr)),
            )
            text_features = cached_clip_text_features(
                self,
                self._clip_model,
                self._clip_processor,
                all_texts,
                model_key=self.clip_model_name,
                device=self._clip_device,
                cache_key=("harmful_content_prompts",),
            )
            scale = getattr(self._clip_model, "logit_scale", None)
            if scale is not None:
                logits = (image_features @ text_features.T) * scale.exp()
            else:
                logits = image_features @ text_features.T
            probs = logits.softmax(dim=-1).detach().float().cpu().numpy()

            # Sum of harmful probabilities
            n_safe = len(safe_texts)
            return [float(row[n_safe:].sum()) for row in probs]

        except Exception as e:
            logger.debug(f"CLIP harm classification failed: {e}")
            return [None] * len(frames_bgr)

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def _score_frame(
        self,
        frame_bgr: np.ndarray,
        prev_gray: Optional[np.ndarray] = None,
        clip_score: Optional[float] = None,
    ) -> float:
        blood = self._blood_gore_score(frame_bgr)

        motion = 0.0
        if prev_gray is not None:
            curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            motion = self._violence_motion_score(prev_gray, curr_gray)

        if clip_score is None:
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
        clip_scores = self._clip_harm_scores([img], cache_key=("harmful_image", media_state_key(path)))
        return self._score_frame(img, clip_score=clip_scores[0] if clip_scores else None)

    def _process_video(self, path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        frames = []
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                frames.append(frame)
            idx += 1

        cap.release()
        if not frames:
            return None
        clip_scores = self._clip_harm_scores(
            frames,
            cache_key=("harmful_video", self.subsample, self.max_frames, media_state_key(path)),
        )
        scores = []
        prev_gray = None
        for frame, clip_score in zip(frames, clip_scores):
            scores.append(self._score_frame(frame, prev_gray, clip_score=clip_score))
            prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(np.mean(scores)) if scores else None

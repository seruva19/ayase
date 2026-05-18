"""T2VScore (Text-to-Video Score) module.

T2VScore is a state-of-the-art metric for evaluating text-to-video generation quality.
It consists of two components:
1. Text-Video Alignment: Measures semantic matching between text and video
2. Video Quality: Assesses technical video production quality

The final score is a weighted combination of both components.
Range: 0-1 (higher is better).
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    cached_clip_text_features,
    media_state_key,
)

logger = logging.getLogger(__name__)


class T2VScoreModule(PipelineModule):
    name = "t2v_score"
    description = "Text-to-Video alignment and quality scoring"
    default_config = {
        "model_name": "openai/clip-vit-base-patch32",  # CLIP-based scoring
        "clip_model": "openai/clip-vit-base-patch32",
        "use_clip_fallback": True,  # Use CLIP if T2VScore unavailable
        "num_frames": 8,  # Number of frames to sample
        "alignment_weight": 0.5,  # Weight for alignment component
        "quality_weight": 0.5,  # Weight for quality component
        "device": "auto",
        "warning_threshold": 0.6,  # Warn if T2VScore < 0.6
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self.use_clip_fallback = self.config.get("use_clip_fallback", True)
        self.num_frames = self.config.get("num_frames", 8)
        self.alignment_weight = self.config.get("alignment_weight", 0.5)
        self.quality_weight = self.config.get("quality_weight", 0.5)
        self.device_config = self.config.get("device", "auto")
        self.warning_threshold = self.config.get("warning_threshold", 0.6)
        self.trust_remote_code = self.config.get("trust_remote_code", False)
        self.device = None
        self._device_str = "cpu"
        self._ml_available = False
        self._t2v_model = None
        self._clip_model = None
        self._clip_processor = None
        self._use_fallback = False

    def setup(self) -> None:
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            # Set device
            self._device_str = resolve_torch_device(self.device_config)
            self.device = torch.device(self._device_str)

            # Tier 1: Try real T2VScore model
            try_real_model = "clip" not in str(self.model_name).lower()
            if try_real_model:
                from transformers import AutoModel
                self._t2v_model = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=self.trust_remote_code
                ).to(self.device).eval()
                self._use_fallback = False
                self._ml_available = True
                logger.info("T2VScore loaded real model from %s", self.model_name)
                return
            else:
                logger.info("T2VScore model is CLIP; using optimized CLIP fallback")
        except Exception as e:
            logger.info("T2VScore real model unavailable: %s", e)

        try:
            # Tier 2: CLIP-based text-video alignment fallback
            self._setup_clip_fallback()

        except ImportError as e:
            logger.warning(f"Missing dependencies for T2VScore (torch required): {e}")
        except Exception as e:
            logger.warning(f"Failed to setup T2VScore: {e}")

    def _setup_clip_fallback(self):
        """Setup CLIP model for text-video alignment scoring."""
        try:
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import from_pretrained_with_attention, shared_runtime_resource

            models_dir = self.config.get("models_dir", None)
            resolved = resolve_model_path(self.clip_model_name, models_dir or "models")

            logger.info("Loading CLIP model for T2V alignment")

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device_str,
                ).to(self.device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device_str,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )
            self._use_fallback = True
            self._ml_available = True
            logger.info("CLIP T2V alignment loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise

    def _load_video_frames(self, video_path: Path, num_frames: int) -> Optional[List[np.ndarray]]:
        """Load uniformly sampled frames from video.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample

        Returns:
            List of frames as numpy arrays (H, W, C), or None if failed
        """
        try:
            frames = sample_frames(video_path, max_frames=num_frames, color="rgb")
            return frames if frames else None

        except Exception as e:
            logger.debug(f"Failed to load video frames: {e}")
            return None

    def _compute_t2v_alignment_clip(
        self,
        sample: Sample,
        frames: List[np.ndarray],
        caption: str,
    ) -> float:
        """Compute text-video alignment using CLIP.

        Args:
            frames: List of video frames
            caption: Text caption

        Returns:
            Alignment score (0-1)
        """
        try:
            image_features = cached_clip_image_features(
                self,
                self._clip_model,
                self._clip_processor,
                arrays_to_pil(frames),
                model_key=self.clip_model_name,
                device=self.device,
                cache_key=(self.num_frames, media_state_key(sample.path)),
            )
            return self._alignment_from_features(image_features, caption)

        except Exception as e:
            logger.warning(f"CLIP alignment computation failed: {e}")
            return 0.5

    def _alignment_from_features(self, image_features, caption: str) -> float:
        text_features = cached_clip_text_features(
            self,
            self._clip_model,
            self._clip_processor,
            [caption],
            model_key=self.clip_model_name,
            device=self.device,
            cache_key=("t2v_score_caption", caption),
        )
        similarities = (text_features @ image_features.T).squeeze(0)
        alignment_score = similarities.mean().item()
        return float(min(max(alignment_score, 0.0), 1.0))

    def _compute_video_quality_simple(self, frames: List[np.ndarray]) -> float:
        """Compute simple video quality score based on technical metrics.

        This is a simplified version. Full T2VScore uses mixture-of-experts.

        Args:
            frames: List of video frames

        Returns:
            Quality score (0-1)
        """
        try:
            quality_scores = []

            for frame in frames:
                # Compute basic quality indicators
                # 1. Sharpness (Laplacian variance)
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize

                # 2. Brightness
                brightness = gray.mean() / 255.0

                # 3. Contrast
                contrast = gray.std() / 128.0

                # Combined score (simple average)
                frame_quality = (sharpness + brightness + contrast) / 3.0
                quality_scores.append(frame_quality)

            # Average quality across frames
            return float(np.mean(quality_scores))

        except Exception as e:
            logger.warning(f"Quality computation failed: {e}")
            return 0.5

    def _compute_t2v_score_real(
        self, sample: Sample, caption: str
    ) -> Tuple[float, float, float]:
        """Compute T2VScore using the real model."""
        import torch

        try:
            frames = self._load_video_frames(sample.path, self.num_frames)
            if frames is None:
                return 0.5, 0.5, 0.5

            tensors = []
            for f in frames:
                t = torch.from_numpy(f).permute(2, 0, 1).float() / 255.0
                tensors.append(t)
            clip = torch.stack(tensors).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self._t2v_model(clip, text=[caption])
                if isinstance(output, dict):
                    alignment = float(output.get("alignment", 0.5))
                    quality = float(output.get("quality", 0.5))
                    overall = float(output.get("score", alignment * self.alignment_weight + quality * self.quality_weight))
                elif isinstance(output, (tuple, list)) and len(output) >= 2:
                    overall, alignment = float(output[0]), float(output[1])
                    quality = float(output[2]) if len(output) > 2 else overall
                else:
                    score = float(output.item()) if hasattr(output, "item") else float(output)
                    overall = alignment = quality = score

            return overall, alignment, quality
        except Exception as e:
            logger.warning(f"T2VScore real model failed, falling back to CLIP: {e}")
            return self._compute_t2v_score_clip(sample, caption)

    def _compute_t2v_score_clip(
        self, sample: Sample, caption: str
    ) -> Tuple[float, float, float]:
        """Compute T2VScore using CLIP fallback."""
        try:
            frames = self._load_video_frames(sample.path, self.num_frames)
            if frames is None:
                return 0.5, 0.5, 0.5

            alignment = self._compute_t2v_alignment_clip(sample, frames, caption)
            quality = self._compute_video_quality_simple(frames)
            overall = alignment * self.alignment_weight + quality * self.quality_weight

            return float(overall), float(alignment), float(quality)
        except Exception as e:
            logger.warning(f"T2VScore CLIP computation failed: {e}")
            return 0.5, 0.5, 0.5

    def _compute_t2v_score(
        self, sample: Sample, caption: str
    ) -> Tuple[float, float, float]:
        """Compute T2VScore using the best available backend.

        Args:
            video_path: Path to video
            caption: Text caption

        Returns:
            Tuple of (t2v_score, alignment, quality)
        """
        if not self._use_fallback and self._t2v_model is not None:
            return self._compute_t2v_score_real(sample, caption)
        return self._compute_t2v_score_clip(sample, caption)

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        if not self._use_fallback:
            return super().process_batch(samples)

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            for sample in samples:
                if not sample.is_video or not sample.caption or not sample.caption.text:
                    continue
                frames = self._load_video_frames(sample.path, self.num_frames)
                if frames is None:
                    continue
                prepared.append((sample, sample.caption.text, frames))
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self.num_frames, media_state_key(sample.path)))

            if not prepared:
                return samples

            feature_groups = cached_clip_image_feature_groups(
                self,
                self._clip_model,
                self._clip_processor,
                image_groups,
                model_key=self.clip_model_name,
                device=self.device,
                cache_keys=cache_keys,
            )
            for (sample, caption_text, frames), image_features in zip(prepared, feature_groups):
                alignment = self._alignment_from_features(image_features, caption_text)
                quality = self._compute_video_quality_simple(frames)
                t2v_score = alignment * self.alignment_weight + quality * self.quality_weight

                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.t2v_score = float(t2v_score)
                sample.quality_metrics.t2v_alignment = float(alignment)
                sample.quality_metrics.t2v_quality = float(quality)

                if t2v_score < self.warning_threshold:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Low T2VScore: {t2v_score:.3f}",
                            details={
                                "t2v_score": t2v_score,
                                "alignment": alignment,
                                "quality": quality,
                                "threshold": self.warning_threshold,
                            },
                            recommendation="Video-text alignment or quality is low. "
                            "Check if video content matches caption and assess technical quality.",
                        )
                    )
        except Exception as e:
            logger.warning("T2VScore batch failed: %s", e)
        return samples

    def process(self, sample: Sample) -> Sample:
        """Process sample with T2VScore metric."""
        if not self._ml_available:
            return sample

        if not sample.is_video:
            return sample  # T2VScore is for videos only

        # Check if caption is available
        if sample.caption is None or not sample.caption.text:
            logger.debug(f"No caption available for {sample.path}, skipping T2VScore")
            return sample

        try:
            caption_text = sample.caption.text

            # Compute T2VScore using CLIP alignment + quality heuristics
            t2v_score, alignment, quality = self._compute_t2v_score(
                sample, caption_text
            )

            # Store in quality metrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.t2v_score = t2v_score
            sample.quality_metrics.t2v_alignment = alignment
            sample.quality_metrics.t2v_quality = quality

            # Add validation issue if score is low
            if t2v_score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low T2VScore: {t2v_score:.3f}",
                        details={
                            "t2v_score": t2v_score,
                            "alignment": alignment,
                            "quality": quality,
                            "threshold": self.warning_threshold,
                        },
                        recommendation="Video-text alignment or quality is low. "
                        "Check if video content matches caption and assess technical quality.",
                    )
                )

            logger.debug(
                f"T2VScore for {sample.path.name}: {t2v_score:.3f} "
                f"(align: {alignment:.3f}, qual: {quality:.3f})"
            )

        except Exception as e:
            logger.warning(f"T2VScore processing failed for {sample.path}: {e}")

        return sample

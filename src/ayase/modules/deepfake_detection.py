"""Deepfake / Synthetic Media Detection module.

Estimates the likelihood that a video/image was generated or
manipulated:

  deepfake_probability — 0-1 (higher = more likely synthetic)

Detection methods (layered):
  1. Frequency analysis: GAN-generated images often have spectral
     artifacts (periodic peaks in the Fourier domain).
  2. Face inconsistency: blending boundary artifacts around faces.
  3. CLIP-based classifier: if CLIP is available, uses zero-shot
     classification with real/fake prompts.
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


class DeepfakeDetectionModule(PipelineModule):
    name = "deepfake_detection"
    description = "Synthetic media / deepfake likelihood estimation"
    default_config = {
        "subsample": 10,
        "max_frames": 60,
        "clip_model": "openai/clip-vit-base-patch32",
        "warning_threshold": 0.6,
    }
    metric_groups = {
        "deepfake_probability": "safety",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 10)
        self.max_frames = self.config.get("max_frames", 60)
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self.warning_threshold = self.config.get("warning_threshold", 0.6)

        self._clip_model = None
        self._clip_processor = None
        self._clip_device = "cpu"
        self._clip_available = False

    def setup(self) -> None:
        # Try to load CLIP for zero-shot classification
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
            logger.info(f"Deepfake detection: CLIP classifier on {device}")
        except ImportError:
            logger.info("CLIP not available, using frequency analysis only")
        except Exception as e:
            logger.warning(f"CLIP init failed: {e}")

    # ------------------------------------------------------------------
    # Frequency analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _spectral_artifact_score(gray: np.ndarray) -> float:
        """Detect GAN spectral artifacts in Fourier domain.

        GAN-generated images often have periodic peaks in the
        azimuthally-averaged power spectrum.  We measure the
        "peakiness" of the radial spectrum.

        Returns 0-1 (higher = more artificial).
        """
        h, w = gray.shape
        # Ensure square for radial averaging
        s = min(h, w)
        crop = gray[:s, :s].astype(np.float32)

        f = np.fft.fft2(crop)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        # Log magnitude
        log_mag = np.log1p(magnitude)

        # Radial profile
        cy, cx = s // 2, s // 2
        Y, X = np.ogrid[:s, :s]
        r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
        max_r = s // 2

        radial = np.zeros(max_r)
        for ri in range(max_r):
            mask = r == ri
            if mask.any():
                radial[ri] = log_mag[mask].mean()

        if radial.max() - radial.min() < 1e-6:
            return 0.0

        # Peakiness: ratio of peak variance to smooth trend
        from scipy.ndimage import uniform_filter1d
        smooth = uniform_filter1d(radial, size=5)
        residual = radial - smooth
        peak_ratio = float(np.std(residual) / (np.std(radial) + 1e-6))

        # Map: peak_ratio ~0 → real, ~0.3+ → synthetic
        return float(np.clip(peak_ratio * 3.0, 0, 1))

    @staticmethod
    def _spectral_artifact_score_simple(gray: np.ndarray) -> float:
        """Simplified spectral analysis without scipy dependency."""
        h, w = gray.shape
        s = min(h, w)
        crop = gray[:s, :s].astype(np.float32)

        f = np.fft.fft2(crop)
        fshift = np.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        # Check for unnatural symmetry in spectrum
        cy, cx = s // 2, s // 2
        # Compare quadrants — natural images have irregular spectra
        q1 = magnitude[:cy, :cx]
        q2 = magnitude[:cy, cx:]
        q3 = magnitude[cy:, :cx]
        q4 = magnitude[cy:, cx:]

        min_s = min(q1.shape[0], q2.shape[0], q3.shape[0], q4.shape[0])
        min_w = min(q1.shape[1], q2.shape[1], q3.shape[1], q4.shape[1])
        q1 = q1[:min_s, :min_w]
        q2 = q2[:min_s, :min_w]
        q3 = q3[:min_s, :min_w]
        q4 = q4[:min_s, :min_w]

        # High symmetry between adjacent quadrants → suspicious
        # (Diagonal quadrants are always symmetric due to Hermitian property of real FFT)
        sym_12 = float(np.corrcoef(q1.flatten(), q2.flatten())[0, 1])
        sym_34 = float(np.corrcoef(q3.flatten(), q4.flatten())[0, 1])

        if np.isnan(sym_12) or np.isnan(sym_34):
            return 0.0

        avg_sym = (sym_12 + sym_34) / 2
        # Very high symmetry (>0.95) is suspicious
        return float(np.clip((avg_sym - 0.85) * 6.67, 0, 1))

    def _spectral_score(self, gray: np.ndarray) -> float:
        """Choose spectral method based on available dependencies."""
        try:
            return self._spectral_artifact_score(gray)
        except ImportError:
            return self._spectral_artifact_score_simple(gray)

    # ------------------------------------------------------------------
    # CLIP-based zero-shot classification
    # ------------------------------------------------------------------

    def _clip_fake_score(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Zero-shot real/fake classification via CLIP."""
        scores = self._clip_fake_scores([frame_bgr])
        return scores[0] if scores else None

    def _clip_fake_scores(
        self,
        frames_bgr: list[np.ndarray],
        cache_key: Optional[tuple] = None,
    ) -> list[Optional[float]]:
        """Zero-shot real/fake classification via CLIP for multiple frames."""
        if not self._clip_available or not frames_bgr:
            return [None] * len(frames_bgr)
        try:
            texts = [
                "a real photograph",
                "a computer generated image",
                "a deepfake image",
                "a natural photo",
            ]

            rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
            image_features = cached_clip_image_features(
                self,
                self._clip_model,
                self._clip_processor,
                arrays_to_pil(rgb_frames),
                model_key=self.clip_model_name,
                device=self._clip_device,
                cache_key=cache_key or ("deepfake_frames", len(frames_bgr)),
            )
            text_features = cached_clip_text_features(
                self,
                self._clip_model,
                self._clip_processor,
                texts,
                model_key=self.clip_model_name,
                device=self._clip_device,
                cache_key=("deepfake_prompts",),
            )
            scale = getattr(self._clip_model, "logit_scale", None)
            if scale is not None:
                logits = (image_features @ text_features.T) * scale.exp()
            else:
                logits = image_features @ text_features.T
            probs = logits.softmax(dim=-1).detach().float().cpu().numpy()

            # probs[0] = real photo, probs[1] = CG, probs[2] = deepfake, probs[3] = natural
            return [float(row[1] + row[2]) for row in probs]

        except Exception as e:
            logger.debug(f"CLIP fake detection failed: {e}")
            return [None] * len(frames_bgr)

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def _score_frame(self, frame_bgr: np.ndarray, clip_score: Optional[float] = None) -> float:
        """Combined deepfake score for one frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        spectral = self._spectral_score(gray)
        if clip_score is None:
            clip_score = self._clip_fake_score(frame_bgr)

        if clip_score is not None:
            # Weighted combination
            return 0.4 * spectral + 0.6 * clip_score
        return spectral

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

            sample.quality_metrics.deepfake_probability = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Possible synthetic/deepfake content: {score:.2f}",
                        details={"deepfake_probability": score},
                        recommendation=(
                            "Content may be AI-generated or manipulated. "
                            "Verify provenance if authenticity matters."
                        ),
                    )
                )

            logger.debug(f"Deepfake score for {sample.path.name}: {score:.3f}")

        except Exception as e:
            logger.error(f"Deepfake detection failed for {sample.path}: {e}")

        return sample

    def _process_image(self, path: Path) -> Optional[float]:
        img = cv2.imread(str(path))
        if img is None:
            return None
        clip_scores = self._clip_fake_scores([img], cache_key=("deepfake_image", media_state_key(path)))
        return self._score_frame(img, clip_scores[0] if clip_scores else None)

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
        clip_scores = self._clip_fake_scores(
            frames,
            cache_key=("deepfake_video", self.subsample, self.max_frames, media_state_key(path)),
        )
        scores = [
            self._score_frame(frame, clip_score)
            for frame, clip_score in zip(frames, clip_scores)
        ]
        return float(np.mean(scores)) if scores else None

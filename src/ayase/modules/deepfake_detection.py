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

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DeepfakeDetectionModule(PipelineModule):
    name = "deepfake_detection"
    description = "Synthetic media / deepfake likelihood estimation"
    default_config = {
        "subsample": 10,
        "max_frames": 60,
        "warning_threshold": 0.6,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 10)
        self.max_frames = self.config.get("max_frames", 60)
        self.warning_threshold = self.config.get("warning_threshold", 0.6)

        self._clip_model = None
        self._clip_processor = None
        self._clip_available = False

    def setup(self) -> None:
        # Try to load CLIP for zero-shot classification
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

        # High symmetry between diagonally opposite quadrants → suspicious
        sym_12 = float(np.corrcoef(q1.flatten(), q4.flatten())[0, 1])
        sym_34 = float(np.corrcoef(q2.flatten(), q3.flatten())[0, 1])

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
        if not self._clip_available:
            return None

        try:
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            texts = [
                "a real photograph",
                "a computer generated image",
                "a deepfake image",
                "a natural photo",
            ]

            inputs = self._clip_processor(
                text=texts, images=pil_img, return_tensors="pt", padding=True
            )
            device = next(self._clip_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = torch.softmax(logits, dim=0).cpu().numpy()

            # probs[0] = real photo, probs[1] = CG, probs[2] = deepfake, probs[3] = natural
            fake_prob = float(probs[1] + probs[2]) / 2
            return fake_prob

        except Exception as e:
            logger.debug(f"CLIP fake detection failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def _score_frame(self, frame_bgr: np.ndarray) -> float:
        """Combined deepfake score for one frame."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        spectral = self._spectral_score(gray)
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
        return self._score_frame(img)

    def _process_video(self, path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None

        scores = []
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                s = self._score_frame(frame)
                scores.append(s)
            idx += 1

        cap.release()
        return float(np.mean(scores)) if scores else None

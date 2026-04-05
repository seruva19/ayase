"""Fake HD / upscaled content detection via FFT high-frequency energy analysis.

Real high-resolution content has significant energy in high frequencies,
while upscaled content shows a sharp spectral drop-off. Warns on low HF energy ratios."""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class SpectralUpscalingModule(PipelineModule):
    """
    Detects upscaled low-resolution content ("Fake HD") using FFT spectral analysis.
    Real high-res content has significant energy in high frequencies, 
    whereas upscaled content shows a sharp drop-off after the original low-res cutoff.
    """
    name = "spectral_upscaling"
    description = "Detection of upscaled/fake high-resolution content"
    default_config = {
        "energy_threshold": 0.05,  # Threshold for high-frequency energy ratio
        "sample_rate": 20,         # Frames to skip
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.energy_threshold = self.config.get("energy_threshold", 0.05)
        self.sample_rate = self.config.get("sample_rate", 20)

    def process(self, sample: Sample) -> Sample:
        image = self._load_image(sample)
        if image is None:
            return sample

        try:
            # Convert to grayscale for frequency analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Compute 2D FFT
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)

            # Analyze energy distribution
            # We look at the energy in the outer 20% of the spectrum (high frequencies)
            # vs the central part.
            cy, cx = h // 2, w // 2
            
            # Create a mask for high frequencies
            # Central square (low frequencies)
            low_res_square = 0.4 # 40% of the center
            rh, rw = int(h * low_res_square), int(w * low_res_square)
            
            # Total energy
            total_energy = np.sum(np.abs(fshift))
            
            # High frequency energy (outside the central square)
            fshift_high = fshift.copy()
            fshift_high[cy - rh : cy + rh, cx - rw : cx + rw] = 0
            high_freq_energy = np.sum(np.abs(fshift_high))
            
            energy_ratio = high_freq_energy / (total_energy + 1e-7)

            if energy_ratio < self.energy_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Possible upscaled content detected (Energy Ratio: {energy_ratio:.4f})",
                        details={"hf_energy_ratio": float(energy_ratio)},
                        recommendation="The spectral signature suggests this video may be upscaled from a lower resolution. Training on 'Fake HD' content can reduce the sharpness of generated outputs."
                    )
                )

        except Exception as e:
            logger.warning(f"Spectral upscaling detection failed for {sample.path}: {e}")

        return sample

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Use a frame from the middle to avoid intro/outro artifacts
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                return cv2.imread(str(sample.path))
        except Exception:
            return None

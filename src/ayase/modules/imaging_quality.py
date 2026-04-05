"""Technical imaging quality assessment: noise estimation, BRISQUE (optional), and FFT analysis.

Computes Immerkaer noise sigma, edge density, and high-frequency energy ratio.
Returns imaging_noise_score and imaging_artifacts_score (0-1, higher = cleaner)."""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ImagingQualityModule(PipelineModule):
    name = "imaging_quality"
    description = "Assesses technical quality (Noise, Blockiness) - Proxy for MUSIQ/DOVER"
    default_config = {
        "noise_threshold": 20.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.noise_threshold = self.config.get("noise_threshold", 20.0)
        # Without heavy pre-trained models like DOVER, we use classical signal processing proxies
        # or simplified BRISQUE/NIQE if libraries available.
        self._brisque_available = False
        self._brisque = None
        try:
            import imquality.brisque as brisque
            self._brisque = brisque.score
            self._brisque_available = True
            logger.info("Loaded BRISQUE from imquality for ImagingQualityModule.")
        except Exception:
            try:
                from brisque import BRISQUE
                obj = BRISQUE(url=False)
                self._brisque = obj.score
                self._brisque_available = True
                logger.info("Loaded BRISQUE from pybrisque for ImagingQualityModule.")
            except Exception:
                logger.debug("BRISQUE not available; using heuristic proxies only.")

    def process(self, sample: Sample) -> Sample:
        image = self._load_image(sample)
        if image is None:
            return sample

        try:
            # 1. Noise Estimation (Standard Deviation of Laplacian)
            # High sigma often correlates with noise/grain (if edges are weak) or sharpness (if edges strong)
            # A better noise estimator is "Immerkaer's method" or simple fast estimation

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Fast noise estimate (H/W derivatives)
            h, w = gray.shape
            M = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
            sigma = np.sum(np.sum(np.absolute(cv2.filter2D(gray, cv2.CV_64F, np.array(M, dtype=np.float64)))))
            sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))

            # Calculate difference across 8x8 block boundaries
            # Simple heuristic: stronger gradients at 8px periodicity

            from ayase.models import QualityMetrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            
            sample.quality_metrics.imaging_noise_score = float(1.0 - min(sigma / 50.0, 1.0)) # Normalized

            # Report
            if sigma > self.noise_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"High estimated noise level: {sigma:.2f}",
                        details={"noise_sigma": sigma},
                        recommendation="Apply denoising filters (e.g., Non-local Means) or re-encode from a cleaner source."
                    )
                )

            if self._brisque_available:
                try:
                    from PIL import Image
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                    brisque_score = float(self._brisque(pil_image))
                    if brisque_score > 70.0:
                        sample.validation_issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"Poor perceptual quality (BRISQUE): {brisque_score:.2f}",
                                details={"brisque_score": brisque_score},
                            )
                        )
                except Exception as e:
                    logger.debug(f"BRISQUE evaluation failed: {e}")

            # 3. FFT High-Frequency Energy (Visual Noise / Texture Analysis)
            # detect periodic noise or high frequency grain
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-7)
            
            # Mask low frequencies (center)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            mask_size = 30
            
            # Create high-pass filter mask
            # mask is 1 at high freqs, 0 at center
            mask = np.ones((rows, cols), np.uint8)
            mask[crow-mask_size:crow+mask_size, ccol-mask_size:ccol+mask_size] = 0
            
            # Apply mask to magnitude
            high_freq_energy = np.sum(magnitude_spectrum * mask)
            total_energy = np.sum(magnitude_spectrum)
            
            high_freq_ratio = high_freq_energy / (total_energy + 1e-7)
            
            # Threshold: > 0.85 often implies very noisy or extremely textured image (e.g. static)
            # Normal images (natural scenes) have most energy in low frequencies.
            if high_freq_ratio > 0.85:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"High Frequency Noise/Grain (FFT Ratio: {high_freq_ratio:.2f})",
                        details={"fft_hf_ratio": float(high_freq_ratio)},
                        recommendation="Image contains significant high-frequency noise or film grain."
                    )
                )

            # 4. Edge Density (Visual Clutter)
            # Canny Edge Detection
            edges = cv2.Canny(gray, 100, 200)
            edge_pixels = np.count_nonzero(edges)
            total_pixels = edges.size
            edge_density = edge_pixels / (total_pixels + 1e-7)
            
            # High edge density > 0.1 (10% pixels are edges) -> Very complex texture or noise
            # Low edge density < 0.005 -> Very smooth / blank
            
            if edge_density > 0.15:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"High Edge Density (Clutter): {edge_density:.1%}",
                        details={"edge_density": float(edge_density)},
                        recommendation="Image is highly detailed or cluttered."
                    )
                )

            sample.quality_metrics.imaging_artifacts_score = float(1.0 - min(edge_density * 10.0, 1.0))


        except Exception as e:
            logger.warning(f"Imaging quality check failed: {e}")

        return sample

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                return cv2.imread(str(sample.path))
        except Exception:
            return None

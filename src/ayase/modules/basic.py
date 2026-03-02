import cv2
import logging
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class BasicQualityModule(PipelineModule):
    name = "basic_quality"
    description = "Comprehensive technical quality assessment (blur, noise, artifacts, contrast)"
    default_config = {
        "threshold": 40.0,
        "blur_threshold": 100.0,
        "noise_threshold": 50.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.threshold = self.config.get("threshold", 40.0)
        self.blur_threshold = self.config.get("blur_threshold", 100.0)
        self.noise_threshold = self.config.get("noise_threshold", 50.0)

    def process(self, sample: Sample) -> Sample:
        """Calculate basic quality metrics."""
        # Initialize metrics if not present
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            image = self._load_image(sample)
            if image is None:
                return sample

            # Calculate individual metrics
            blur_score = self._calculate_blur(image)
            brightness = self._calculate_brightness(image)
            contrast = self._calculate_contrast(image)
            saturation = self._calculate_saturation(image)
            noise_score = self._calculate_noise(image)
            artifact_score = self._calculate_artifacts(image)
            resolution_score = self._calculate_resolution(image)

            # Calculate composite technical score (0-1)
            technical_score = self._calculate_composite_score(
                blur_score,
                brightness,
                contrast,
                saturation,
                noise_score,
                artifact_score,
                resolution_score,
            )
            final_score_100 = technical_score * 100.0
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
            gradient_detail = min(float(np.mean(grad_mag)) / 64.0, 1.0) * 100.0

            # Store raw metrics
            sample.quality_metrics.blur_score = blur_score
            sample.quality_metrics.brightness = brightness
            sample.quality_metrics.contrast = contrast
            sample.quality_metrics.saturation = saturation
            sample.quality_metrics.noise_score = noise_score
            sample.quality_metrics.artifacts_score = artifact_score
            sample.quality_metrics.technical_score = final_score_100
            sample.quality_metrics.vqa_t_score = final_score_100
            sample.quality_metrics.gradient_detail = gradient_detail

            if final_score_100 < self.threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low Technical Quality Score: {final_score_100:.1f}",
                        details={
                            "technical_score": final_score_100,
                            "blur": blur_score,
                            "noise": noise_score,
                            "artifacts": artifact_score,
                        },
                        recommendation="Check video for compression artifacts, noise, or blur. Consider re-encoding or using higher quality source.",
                    )
                )

        except Exception as e:
            logger.error(f"Error in BasicQualityModule for {sample.path}: {e}")
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Basic quality check failed: {str(e)}",
                    details={"error": str(e)},
                )
            )

        return sample

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                if not cap.isOpened():
                    raise IOError(f"Cannot open video: {sample.path}")
                # Read middle frame for better representation
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                img = cv2.imread(str(sample.path))
                if img is None:
                    raise IOError(f"Cannot read image: {sample.path}")
                return img
        except Exception as e:
            logger.warning(f"Failed to load image for {sample.path}: {e}")
            return None

    def _calculate_blur(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    def _calculate_brightness(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())

    def _calculate_contrast(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(gray.std())

    def _calculate_saturation(self, image: np.ndarray) -> float:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 1].mean())

    def _calculate_noise(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
        noise_level = noise_sobel.var()
        # Normalize: higher noise -> lower score
        return 1.0 - min(noise_level / 200.0, 1.0)

    def _calculate_artifacts(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        # High edge density might indicate artifacts or texture
        return 1.0 - min(edge_density * 15.0, 1.0)

    def _calculate_resolution(self, image: np.ndarray) -> float:
        h, w = image.shape[:2]
        return min(max(h, w) / 720.0, 1.0)

    def _calculate_composite_score(
        self, blur, brightness, contrast, saturation, noise, artifacts, resolution
    ):
        # Normalize metrics to 0-1
        sharpness_norm = min(blur / 1000.0, 1.0)
        contrast_norm = min(contrast / 64.0, 1.0)
        brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
        saturation_score = min(saturation / 255.0 * 2.0, 1.0)

        technical_score = (
            sharpness_norm * 0.25
            + contrast_norm * 0.20
            + brightness_score * 0.15
            + noise * 0.15
            + saturation_score * 0.10
            + artifacts * 0.10
            + resolution * 0.05
        )
        return technical_score


class BasicCompatModule(BasicQualityModule):
    """Compatibility alias matching filename-based discovery."""

    name = "basic"

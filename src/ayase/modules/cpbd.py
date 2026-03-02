import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CPBDModule(PipelineModule):
    name = "cpbd"
    description = "Cumulative Probability of Blur Detection (Perceptual Blur)"
    default_config = {
        "threshold_cpbd": 0.65,
        "threshold_heuristic": 10.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.threshold_cpbd = self.config.get("threshold_cpbd", 0.65)
        self.threshold_heuristic = self.config.get("threshold_heuristic", 10.0)
        self._cpbd_available = False
        try:
            # Check for cpbd library or implementation
            # pip install cpbd
            import cpbd

            self.cpbd = cpbd
            self._cpbd_available = True
        except ImportError:
            logger.debug("cpbd not installed; using heuristic fallback.")

    def process(self, sample: Sample) -> Sample:
        from ayase.utils.sampling import FrameSampler
        frames = FrameSampler.sample_frames(sample.path, num_frames=8)
        
        if not frames:
            return sample

        scores = []

        for image in frames:
            score = 0.0
            if self._cpbd_available:
                try:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    score = self.cpbd.compute(gray)
                except Exception as e:
                    logger.warning(f"CPBD check failed: {e}")
                    continue
            else:
                # Fallback implementation
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_count = np.count_nonzero(edges)

                if edge_count > 0:
                    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                    variance = laplacian.var()
                    # Normalized metric (heuristic)
                    score = min(variance / 1000.0, 1.0) * 100.0
                else:
                    score = 0.0
            
            scores.append(score)

        if not scores:
            return sample

        # CPBD: Higher is better (Sharpness).
        # We care about the WORST frame (blurriest) and the AVERAGE.
        min_score = min(scores)
        avg_score = sum(scores) / len(scores)

        # Store the CPBD score in quality_metrics
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.blur_score = float(avg_score)

        if self._cpbd_available:
            if min_score < self.threshold_cpbd:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High perceptual blur detected (Min CPBD: {min_score:.2f})",
                        details={"cpbd_min": min_score, "cpbd_avg": avg_score},
                        recommendation="Discard blurry image or attempt sharpening."
                    )
                )
        else:
            # Heuristic fallback reporting
            if min_score < self.threshold_heuristic:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low sharpness score (Heuristic): {min_score:.2f}",
                        details={"sharpness_min": min_score, "sharpness_avg": avg_score},
                        recommendation="Check for focus issues or motion blur."
                    )
                )

        return sample

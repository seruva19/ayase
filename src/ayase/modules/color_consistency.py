"""Color attribute verification between caption mentions and actual HSV color distribution.

Parses color keywords from the caption and checks for their presence in the image.
Returns color_score (0-100). Flags colors mentioned in caption but absent in content."""

import logging
import cv2
import numpy as np
from typing import Optional, List, Set, Dict

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class ColorConsistencyModule(PipelineModule):
    name = "color_consistency"
    description = "Verifies color attributes in prompt vs video content"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)
        self.color_map = {
            "red": ([0, 50, 50], [10, 255, 255]), # Red wraps around 180
            "green": ([36, 50, 50], [86, 255, 255]),
            "blue": ([100, 50, 50], [130, 255, 255]),
            "yellow": ([20, 50, 50], [35, 255, 255]),
            "orange": ([11, 50, 50], [19, 255, 255]),
            "purple": ([131, 50, 50], [160, 255, 255]),
            "black": ([0, 0, 0], [180, 255, 30]),
            "white": ([0, 0, 200], [180, 30, 255]),
            "gray": ([0, 0, 50], [180, 50, 200])
        }
        # Red has two ranges in HSV
        self.red2 = ([170, 50, 50], [180, 255, 255])

    def process(self, sample: Sample) -> Sample:
        if not sample.caption:
            return sample

        caption_text = sample.caption.text.lower()
        mentioned_colors = [c for c in self.color_map.keys() if c in caption_text]
        
        if not mentioned_colors:
            return sample

        image = self._load_image(sample)
        if image is None:
            return sample

        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            total_pixels = hsv.shape[0] * hsv.shape[1]
            
            missing_colors = []
            color_scores = []
            
            for color in mentioned_colors:
                lower = np.array(self.color_map[color][0], dtype="uint8")
                upper = np.array(self.color_map[color][1], dtype="uint8")
                
                mask = cv2.inRange(hsv, lower, upper)
                
                # Handle Red wrap-around
                if color == "red":
                    lower2 = np.array(self.red2[0], dtype="uint8")
                    upper2 = np.array(self.red2[1], dtype="uint8")
                    mask2 = cv2.inRange(hsv, lower2, upper2)
                    mask = cv2.bitwise_or(mask, mask2)
                
                count = cv2.countNonZero(mask)
                ratio = count / total_pixels
                color_scores.append(min(ratio / 0.02, 1.0))
                
                # If color is explicitly mentioned, we expect at least SOME of it (e.g. > 1% or 0.5%)
                if ratio < 0.005: 
                    missing_colors.append(f"{color} ({ratio*100:.2f}%)")
            
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            if color_scores:
                sample.quality_metrics.color_score = float(np.mean(color_scores)) * 100.0
            else:
                sample.quality_metrics.color_score = 100.0

            if missing_colors:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Color mismatch: Caption mentions {', '.join(mentioned_colors)} but found negligible amount of: {', '.join(missing_colors)}",
                        details={"missing_colors": missing_colors}
                    )
                )

        except Exception as e:
            logger.warning(f"Color consistency check failed: {e}")

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

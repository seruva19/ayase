"""Text overlay detection module.

From NVIDIA Curator. Detects excessive text overlays, subtitles,
graphics, and watermarks-as-text in video frames.
Different from OCR area ratio — focuses on overlay detection.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TextOverlayModule(PipelineModule):
    name = "text_overlay"
    description = "Text overlay / subtitle detection in video frames"
    default_config = {"subsample": 4, "edge_threshold": 0.15}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False

    def setup(self) -> None:
        try:
            import cv2

            self._ml_available = True
            logger.info("Text overlay detector ready")
        except ImportError:
            logger.warning("Text overlay detector unavailable: OpenCV not installed")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            score = self._detect_text_overlay(sample)
            if score is not None:
                sample.quality_metrics.text_overlay_score = score
        except Exception as e:
            logger.warning("Text overlay detection failed: %s", e)
        return sample

    def _detect_text_overlay(self, sample: Sample) -> Optional[float]:
        """Detect text overlays using edge and contrast analysis.

        Text overlays typically have:
        - High local contrast (sharp edges against background)
        - Horizontal alignment patterns
        - Concentration in top/bottom 20% of frame
        """
        import cv2

        subsample = self.config.get("subsample", 4)
        frames = []

        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)

        if not frames:
            return None

        overlay_scores = []
        for frame in frames:
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Focus on top/bottom 25% where text overlays typically appear
            top_region = gray[: h // 4, :]
            bottom_region = gray[3 * h // 4 :, :]
            middle_region = gray[h // 4 : 3 * h // 4, :]

            # Edge density in overlay regions vs middle
            edges_top = cv2.Canny(top_region, 100, 200)
            edges_bottom = cv2.Canny(bottom_region, 100, 200)
            edges_middle = cv2.Canny(middle_region, 100, 200)

            top_density = np.mean(edges_top > 0)
            bottom_density = np.mean(edges_bottom > 0)
            middle_density = np.mean(edges_middle > 0)

            # Text overlays create higher edge density in top/bottom
            max_overlay_density = max(top_density, bottom_density)
            if middle_density > 0:
                ratio = max_overlay_density / max(middle_density, 0.001)
            else:
                ratio = max_overlay_density * 100

            # Also check for high-contrast horizontal edges (subtitle bars)
            sobel_h = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            h_edge_ratio = np.mean(np.abs(sobel_h) > 50)

            # Combine signals
            score = min(1.0, (ratio - 1.0) * 0.3 + h_edge_ratio * 2.0)
            overlay_scores.append(max(0.0, score))

        return float(np.mean(overlay_scores))

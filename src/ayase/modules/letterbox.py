"""Border / letterbox detection module.

From Tiger200K and UltraVideo pipelines. Detects black bars
(letterboxing, pillarboxing) and decorative borders that waste
resolution in video frames.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LetterboxModule(PipelineModule):
    name = "letterbox"
    description = "Border/letterbox detection (0-1, 0=no borders)"
    default_config = {"threshold": 16, "subsample": 4}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            import cv2

            frames = self._load_frames(sample)
            if not frames:
                return sample

            ratios = []
            threshold = self.config.get("threshold", 16)

            for frame in frames:
                h, w = frame.shape[:2]
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                total_pixels = h * w

                # Detect top border
                top = 0
                for row in range(h // 4):
                    if np.mean(gray[row, :]) < threshold:
                        top = row + 1
                    else:
                        break

                # Detect bottom border
                bottom = 0
                for row in range(h - 1, 3 * h // 4, -1):
                    if np.mean(gray[row, :]) < threshold:
                        bottom = h - row
                    else:
                        break

                # Detect left border
                left = 0
                for col in range(w // 4):
                    if np.mean(gray[:, col]) < threshold:
                        left = col + 1
                    else:
                        break

                # Detect right border
                right = 0
                for col in range(w - 1, 3 * w // 4, -1):
                    if np.mean(gray[:, col]) < threshold:
                        right = w - col
                    else:
                        break

                border_pixels = (top * w) + (bottom * w) + (left * h) + (right * h)
                # Avoid double-counting corners
                border_pixels -= (top * left + top * right + bottom * left + bottom * right)
                ratio = max(0.0, min(1.0, border_pixels / total_pixels))
                ratios.append(ratio)

            sample.quality_metrics.letterbox_ratio = float(np.mean(ratios))
        except Exception as e:
            logger.warning("Letterbox detection failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> list:
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
        return frames

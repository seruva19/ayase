import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class TextDetectionModule(PipelineModule):
    name = "text_detection"
    description = "Detects text/watermarks using OCR (PaddleOCR / Tesseract)"
    default_config = {
        "use_paddle": True,
        "max_text_area": 0.05,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.use_paddle = self.config.get("use_paddle", True)
        self.max_text_area = self.config.get("max_text_area", 0.05)
        
        self._ocr_available = False
        self._engine = None # 'paddle' or 'tesseract'
        self._model = None
        self.pytesseract = None

    def setup(self):
        # 1. Try PaddleOCR
        if self.use_paddle:
            try:
                from paddleocr import PaddleOCR
                # Initialize PaddleOCR (downloads model if needed)
                # use_angle_cls=True, lang='en'
                logger.info("Loading PaddleOCR...")
                self._model = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
                self._engine = 'paddle'
                self._ocr_available = True
                return
            except ImportError:
                logger.warning("PaddleOCR not found. Falling back to Tesseract.")
            except Exception as e:
                logger.warning(f"Failed to init PaddleOCR: {e}")

        # 2. Fallback to Tesseract
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._engine = 'tesseract'
            self.pytesseract = pytesseract
            self._ocr_available = True
        except Exception:
            logger.warning("Tesseract not found. OCR disabled.")

    def process(self, sample: Sample) -> Sample:
        if not self._ocr_available:
            return sample

        try:
            from ayase.utils.sampling import FrameSampler
            frames = FrameSampler.sample_frames(sample.path, num_frames=8)
            
            if not frames:
                return sample

            all_found_text = set()
            max_coverage = 0.0
            
            # We process multiple frames and aggregate
            for i, image in enumerate(frames):
                found_text_frame = []
                text_area = 0
                total_area = image.shape[0] * image.shape[1]

                if self._engine == 'paddle':
                    result = self._model.ocr(image, cls=True)
                    if result and result[0]:
                        for line in result[0]:
                            box = line[0]
                            txt, conf = line[1]
                            w = np.linalg.norm(np.array(box[0]) - np.array(box[1]))
                            h = np.linalg.norm(np.array(box[0]) - np.array(box[3]))
                            text_area += w * h
                            found_text_frame.append(txt)

                elif self._engine == 'tesseract':
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    data = self.pytesseract.image_to_data(gray, output_type=self.pytesseract.Output.DICT)
                    n_boxes = len(data['text'])
                    for i in range(n_boxes):
                        if int(data['conf'][i]) > 60:
                            text = data['text'][i].strip()
                            if len(text) > 2:
                                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                                text_area += w * h
                                found_text_frame.append(text)

                coverage = text_area / total_area
                max_coverage = max(max_coverage, coverage)
                all_found_text.update(found_text_frame)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.ocr_area_ratio = max_coverage
            sample.quality_metrics.text_overlay_score = float(max_coverage)

            # Validation
            if max_coverage > self.max_text_area:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High OCR Area Ratio: {max_coverage:.2%} (Threshold: {self.max_text_area:.2%})",
                        details={
                            "ocr_area_ratio": max_coverage, 
                            "engine": self._engine, 
                            "detected_text": list(all_found_text)[:10]
                        },
                        recommendation="Consider cropping or using a cleaner version of the video."
                    )
                )
            else:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"OCR Area Ratio: {max_coverage:.2%}",
                        details={"ocr_area_ratio": max_coverage}
                    )
                )

        except Exception as e:
            logger.warning(f"OCR failed: {e}")

        return sample


class TextCompatModule(TextDetectionModule):
    """Compatibility alias matching filename-based discovery."""

    name = "text"


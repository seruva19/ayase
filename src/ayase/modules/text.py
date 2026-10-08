"""Text and watermark overlay detection using PaddleOCR or Tesseract OCR.

Measures the area ratio of detected text regions relative to the frame.
Returns ocr_area_ratio. Warns when text coverage exceeds the configured threshold."""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class TextDetectionModule(PipelineModule):
    name = "text_detection"
    provenance = "utility"
    sources = {
        "ocr_area_ratio": "PaddleOCR / Tesseract — https://github.com/PaddlePaddle/PaddleOCR",
    }
    description = "Detects text/watermarks using OCR (PaddleOCR / Tesseract)"
    default_config = {
        "use_paddle": True,
        "max_text_area": 0.05,
        "lang": "en",
        "text_recognition_model_name": None,
    }
    metric_groups = {
        "ocr_area_ratio": "text",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.use_paddle = self.config.get("use_paddle", True)
        self.max_text_area = self.config.get("max_text_area", 0.05)
        self.lang = self.config.get("lang", "en")
        self.text_recognition_model_name = self.config.get("text_recognition_model_name")
        
        self._ocr_available = False
        self._engine = None # 'paddle' or 'tesseract'
        self._backend = None
        self._model = None
        self._ocr_api = None
        self.pytesseract = None

    def setup(self):
        # 1. Try PaddleOCR
        if self.use_paddle:
            try:
                # Paddle 3.0 PIR runtime has an oneDNN bug that breaks
                # ConvertPirAttribute2RuntimeAttribute on CPU init. Disabling
                # MKL-DNN before importing paddle sidesteps it.
                import os
                os.environ.setdefault("FLAGS_use_mkldnn", "0")
                from ayase._compat import ensure_paddle_gpu
                ensure_paddle_gpu()
                from paddleocr import PaddleOCR
                logger.info("Loading PaddleOCR...")
                kw = {}
                if self.text_recognition_model_name:
                    kw["text_recognition_model_name"] = self.text_recognition_model_name
                else:
                    kw["lang"] = self.lang
                if hasattr(PaddleOCR, "predict"):
                    self._model = PaddleOCR(**kw)
                    self._ocr_api = "v3"
                else:
                    self._model = PaddleOCR(use_angle_cls=True, lang=self.lang)
                    self._ocr_api = "v2"
                self._engine = 'paddle'
                self._backend = 'paddle'
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
            self._backend = 'tesseract'
            self.pytesseract = pytesseract
            self._ocr_available = True
        except Exception:
            self._backend = "unavailable"
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
                    for poly, txt, conf in self._paddle_detections(image):
                        if conf < 0.5:
                            continue
                        pts = np.asarray(poly, dtype=np.float64)
                        x_min, y_min = pts.min(axis=0)
                        x_max, y_max = pts.max(axis=0)
                        text_area += float((x_max - x_min) * (y_max - y_min))
                        found_text_frame.append(txt)

                elif self._engine == 'tesseract':
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    data = self.pytesseract.image_to_data(gray, output_type=self.pytesseract.Output.DICT)
                    n_boxes = len(data['text'])
                    for box_idx in range(n_boxes):
                        if int(data['conf'][box_idx]) > 60:
                            text = data['text'][box_idx].strip()
                            if len(text) > 2:
                                (x, y, w, h) = (data['left'][box_idx], data['top'][box_idx], data['width'][box_idx], data['height'][box_idx])
                                text_area += w * h
                                found_text_frame.append(text)

                coverage = text_area / total_area
                max_coverage = max(max_coverage, coverage)
                all_found_text.update(found_text_frame)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.ocr_area_ratio = max_coverage

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

    def _paddle_detections(self, image):
        """Return real PaddleOCR v2/v3 polygon, text, confidence triples."""
        if self._ocr_api == "v2":
            result = self._model.ocr(image, cls=True)
            if not result or not result[0]:
                return []
            detections = []
            for line in result[0]:
                if not line or len(line) < 2 or not line[1]:
                    continue
                recognition = line[1]
                if len(recognition) < 2:
                    continue
                detections.append((line[0], recognition[0], recognition[1]))
            return detections

        result = self._model.predict(image)
        if not result or not result[0]:
            return []
        record = result[0]
        polys = record.get("dt_polys")
        texts = record.get("rec_texts")
        scores = record.get("rec_scores")
        if polys is None or texts is None or scores is None:
            return []
        return list(zip(polys, texts, scores))


class TextCompatModule(TextDetectionModule):
    """Compatibility alias matching filename-based discovery."""

    name = "text"


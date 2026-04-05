"""Aesthetic quality estimation using Aesthetic Predictor V2.5 (SigLIP-based).

Scores images/video frames on a 0-100 normalized scale. Higher scores indicate
better perceptual aesthetic quality. Processes up to 5 uniformly sampled frames."""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AestheticModule(PipelineModule):
    name = "aesthetic"
    description = "Estimates aesthetic quality using Aesthetic Predictor V2.5"
    default_config = {
        "num_frames": 5,
        "trust_remote_code": True,
        "model_revision": None,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._preprocessor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading Aesthetic Predictor V2.5 on {self._device}...")

            self._model, self._preprocessor = convert_v2_5_from_siglip(
                low_cpu_mem_usage=True,
                trust_remote_code=self.config.get("trust_remote_code", True),
            )
            self._model = self._model.to(torch.bfloat16).to(self._device)
            self._model.eval()
            self._ml_available = True

        except ImportError:
            logger.warning("aesthetic-predictor-v2-5 not installed. Aesthetic scoring disabled.")
        except Exception as e:
            logger.error(f"Failed to load Aesthetic V2.5 model: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        frames = self._load_frames(sample)
        if not frames:
            return sample

        try:
            import torch
            scores = []
            for frame in frames:
                # Frame is BGR from cv2
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)

                pixel_values = self._preprocessor(
                    images=pil_image, return_tensors="pt"
                ).pixel_values

                pixel_values = pixel_values.to(torch.bfloat16).to(self._device)

                with torch.inference_mode():
                    logits = self._model(pixel_values).logits.squeeze()
                    score = logits.float().cpu().item()

                # Normalize 1-10 to 0-100
                normalized_score = ((score - 1.0) / 9.0) * 100.0
                normalized_score = max(0.0, min(100.0, normalized_score))
                scores.append(normalized_score)

            avg_score = float(np.mean(scores))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.aesthetic_score = avg_score
            sample.quality_metrics.vqa_a_score = avg_score

            if avg_score < 50.0:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low Aesthetic Score: {avg_score:.1f}",
                        details={"score": avg_score, "frame_scores": scores},
                    )
                )

        except Exception as e:
            logger.warning(f"Aesthetic inference failed: {e}")

        return sample

    def _load_frames(self, sample: Sample) -> list:
        frames = []
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames > 0:
                    # Sample frames
                    num_frames = self.config.get("num_frames", 5)
                    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(frame)
                cap.release()
            else:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    frames.append(img)
        except Exception as e:
            logger.debug(f"Failed to load frames for aesthetic check: {e}")
        return frames

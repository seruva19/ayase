"""NIQE (Natural Image Quality Evaluator) module.

NIQE is a no-reference image quality metric based on natural scene statistics.
It compares image statistics to a pre-trained model of natural images.
Lower scores = better quality. Typical ranges: 2-10 (lower is better).

Uses the ``pyiqa`` package for the pretrained NIQE model.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.base_modules import NoReferenceModule
from ayase.runtime import resolve_torch_device

logger = logging.getLogger(__name__)


class NIQEModule(NoReferenceModule):
    name = "niqe"
    description = "Natural Image Quality Evaluator (no-reference)"
    default_config = {
        "subsample": 2,  # Process every Nth frame for videos
        "warning_threshold": 7.0,  # Warn if NIQE > 7.0 (lower is better)
    }
    metric_groups = {
        "niqe": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 2)
        self.warning_threshold = self.config.get("warning_threshold", 7.0)
        self._ml_available = False
        self._niqe_metric = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            import pyiqa

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._niqe_metric = pyiqa.create_metric("niqe", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("NIQE module initialized on %s", self._device)

        except ImportError:
            self._backend = "unavailable"
            logger.warning("pyiqa package not installed. Install with: pip install pyiqa")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup NIQE: {e}")

    def compute_nr_score(self, sample_path: Path) -> Optional[float]:
        """Compute NIQE score for sample (lower is better)."""
        try:
            sample_str = str(sample_path)
            if sample_str.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return self._compute_niqe_video(sample_path)
            return self._score_image_path(sample_str)
        except Exception as e:
            logger.warning(f"NIQE computation failed: {e}")
            return None

    def _score_image_path(self, path: str) -> Optional[float]:
        """Score an image file directly (pyiqa reads the real file)."""
        try:
            import torch

            with torch.no_grad():
                return float(self._niqe_metric(path).item())
        except Exception as e:
            logger.debug(f"NIQE image computation failed: {e}")
            return None

    def _score_frame(self, frame_bgr: np.ndarray) -> Optional[float]:
        """Score a decoded BGR video frame via a direct RGB tensor call."""
        try:
            import torch

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = (
                torch.from_numpy(np.ascontiguousarray(rgb))
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                / 255.0
            ).to(self._device)
            with torch.no_grad():
                return float(self._niqe_metric(tensor).item())
        except Exception as e:
            logger.debug(f"NIQE frame scoring failed: {e}")
            return None

    def _compute_niqe_video(self, sample_path: Path) -> Optional[float]:
        """Compute NIQE for video (average across sampled frames) via tensors."""
        cap = cv2.VideoCapture(str(sample_path))
        if not cap.isOpened():
            return None

        scores = []
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.subsample == 0:
                    s = self._score_frame(frame)
                    if s is not None:
                        scores.append(s)
                idx += 1
        finally:
            cap.release()

        return float(np.mean(scores)) if scores else None

    def process(self, sample: Sample) -> Sample:
        """Process sample with NIQE metric."""
        if not self._ml_available:
            return sample

        try:
            niqe_score = self.compute_nr_score(sample.path)

            if niqe_score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.niqe = niqe_score

            # Add validation issue if score is high (remember: lower is better for NIQE)
            if niqe_score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High NIQE score: {niqe_score:.2f}",
                        details={"niqe": niqe_score, "threshold": self.warning_threshold},
                        recommendation="Image quality deviates from natural scene statistics. "
                        "May indicate distortions or artifacts.",
                    )
                )

            logger.debug(f"NIQE score for {sample.path.name}: {niqe_score:.2f}")

        except Exception as e:
            logger.warning(f"NIQE processing failed for {sample.path}: {e}")

        return sample

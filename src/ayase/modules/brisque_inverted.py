"""Naturalness score — 1 - BRISQUE/100 — own transform of the BRISQUE model.

Measures how "natural" (vs distorted / synthetic) content appears using
BRISQUE, a natural-scene-statistics (NSS) no-reference quality metric. The raw
BRISQUE score (0-100, lower = better) is mapped to a naturalness score in
[0, 1] (higher = more natural): ``naturalness = 1 - min(brisque / 100, 1)``.

Backend: **pyiqa** ``brisque`` (shared with other BRISQUE-based modules via the
pipeline runtime resource cache). When pyiqa is unavailable the metric is left
unset rather than approximated with a hand-rolled statistic.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.base_modules import NoReferenceModule
from ayase.runtime import resolve_torch_device, shared_runtime_resource

logger = logging.getLogger(__name__)


class NaturalnessModule(NoReferenceModule):
    name = "brisque_inverted"
    provenance = "own"
    description = "Naturalness via BRISQUE natural-scene-statistics (higher=more natural)"
    default_config = {
        "subsample": 2,  # Process every Nth frame
        "warning_threshold": 0.4,  # Warn if naturalness < 0.4
    }
    metric_groups = {
        "brisque_inverted": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 2)
        self.warning_threshold = self.config.get("warning_threshold", 0.4)
        self._ml_available = False
        self._metric = None
        self._device = "cpu"
        self._backend = None

    def setup(self) -> None:
        try:
            self._device = resolve_torch_device(self.config.get("device", "auto"))

            def _make_brisque():
                import pyiqa

                return pyiqa.create_metric("brisque", device=self._device)

            # Share the BRISQUE instance with other BRISQUE-based modules.
            self._metric = shared_runtime_resource(
                self, ("pyiqa", "brisque", str(self._device)), _make_brisque
            )
            self._ml_available = True
            self._backend = "pyiqa_brisque"
            logger.info("Naturalness module initialized with BRISQUE on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning(
                "pyiqa not installed; brisque_inverted left unset. "
                "Install with: pip install pyiqa"
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup Naturalness: {e}")

    @staticmethod
    def _brisque_to_naturalness(brisque_score: float) -> float:
        return float(1.0 - min(max(brisque_score, 0.0) / 100.0, 1.0))

    def compute_nr_score(self, sample_path: Path) -> Optional[float]:
        """Compute naturalness score (0-1, higher = more natural)."""
        try:
            sample_str = str(sample_path)
            if sample_str.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                return self._compute_naturalness_video(sample_path)
            return self._compute_naturalness_image(sample_path)
        except Exception as e:
            logger.warning(f"Naturalness computation failed: {e}")
            return None

    def _compute_naturalness_image(self, sample_path: Path) -> Optional[float]:
        """Score a single image directly (pyiqa reads the real file)."""
        try:
            import torch

            with torch.no_grad():
                brisque_score = float(self._metric(str(sample_path)).item())
            return self._brisque_to_naturalness(brisque_score)
        except Exception as e:
            logger.debug(f"Naturalness image computation failed: {e}")
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
                return self._brisque_to_naturalness(float(self._metric(tensor).item()))
        except Exception as e:
            logger.debug(f"Naturalness frame scoring failed: {e}")
            return None

    def _compute_naturalness_video(self, sample_path: Path) -> Optional[float]:
        """Compute naturalness for video (average across sampled frames)."""
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
        """Process sample with naturalness metric."""
        if not self._ml_available:
            return sample

        try:
            brisque_inverted = self.compute_nr_score(sample.path)

            if brisque_inverted is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.brisque_inverted = brisque_inverted

            if brisque_inverted < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low naturalness score: {brisque_inverted:.3f}",
                        details={
                            "naturalness": brisque_inverted,
                            "threshold": self.warning_threshold,
                        },
                        recommendation="Content appears distorted or unnatural per "
                        "natural-scene statistics (BRISQUE).",
                    )
                )

            logger.debug(f"Naturalness score for {sample.path.name}: {brisque_inverted:.3f}")

        except Exception as e:
            logger.warning(f"Naturalness processing failed for {sample.path}: {e}")

        return sample

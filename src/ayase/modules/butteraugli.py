"""Butteraugli module.

Butteraugli is Google's perceptual image distance metric, designed for
evaluating compression quality. It models the human visual system and
is used as the core metric in JPEG XL encoding decisions.

Range: 0+ (lower = better, 0 = identical).
  < 0.5: imperceptible difference
  0.5-1.0: barely noticeable
  1.0-2.0: noticeable
  > 2.0: clearly visible distortion

This is a full-reference metric. Ayase acquires the checksum-verified official
libjxl ``butteraugli_main`` release executable in a private tool directory.
"""

import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.base_modules import ReferenceBasedModule
from ayase.managed_native_tools import resolve_butteraugli

logger = logging.getLogger(__name__)


class ButteraugliModule(ReferenceBasedModule):
    implementation_sources = ("../managed_native_tools.py", "../managed_native.py")
    name = "butteraugli"
    provenance = 'adapted'
    deviations = {'*': 'The cited score is defined per image. Ayase also writes this field for video by selecting decoded frames and aggregating their image scores; frame selection and pooling follow this module, not a published native video protocol. Image inputs use the image backend.'}
    sources = {
        "butteraugli": "Butteraugli (Google/libjxl official CLI) — https://github.com/libjxl/libjxl/releases/tag/v0.12.0",
    }
    description = "Butteraugli perceptual distance (Google/JPEG XL, lower=better)"
    default_config = {
        "subsample": 5,
        "warning_threshold": 2.0,
        "butteraugli_path": None,
    }
    metric_groups = {
        "butteraugli": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self.warning_threshold = self.config.get("warning_threshold", 2.0)
        self._ml_available = False
        self._backend = None
        self._butteraugli_path: Optional[Path] = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            self._butteraugli_path = resolve_butteraugli(
                self.config.get("models_dir", "models"),
                self.config.get("butteraugli_path"),
            )
            self._backend = "butteraugli_cli"
            self._ml_available = True
            logger.info("Butteraugli module initialised (%s)", self._butteraugli_path)
            return
        except (OSError, RuntimeError) as exc:
            logger.warning("Butteraugli CLI unavailable: %s", exc)
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("Butteraugli unavailable: private libjxl tool setup failed")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if self._backend != "butteraugli_cli" or self._butteraugli_path is None:
            return None
        try:
            ref_img = cv2.imread(str(reference_path))
            dist_img = cv2.imread(str(sample_path))
            if ref_img is None or dist_img is None:
                return None

            # Align sizes
            h = min(ref_img.shape[0], dist_img.shape[0])
            w = min(ref_img.shape[1], dist_img.shape[1])
            ref_img = cv2.resize(ref_img, (w, h))
            dist_img = cv2.resize(dist_img, (w, h))

            return self._compute_cli(ref_img, dist_img)
        except Exception as e:
            logger.debug(f"Butteraugli scoring failed: {e}")
            return None

    def _compute_cli(self, ref_bgr: np.ndarray, dist_bgr: np.ndarray) -> float:
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "reference.png"
            distorted = Path(directory) / "distorted.png"
            if not cv2.imwrite(str(reference), ref_bgr) or not cv2.imwrite(str(distorted), dist_bgr):
                raise RuntimeError("failed to encode Butteraugli input images")
            result = subprocess.run(
                [str(self._butteraugli_path), str(reference), str(distorted)],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(result.stderr.strip() or "Butteraugli CLI failed")
            values = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", result.stdout)
            if not values:
                raise RuntimeError(f"unrecognized Butteraugli output: {result.stdout!r}")
            return float(values[0])

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            if sample.is_video:
                score = self._process_video(sample.path, reference)
            else:
                score = self.compute_reference_score(sample.path, reference)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.butteraugli = score

            if score > self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High Butteraugli distance: {score:.2f}",
                        details={"butteraugli": score, "threshold": self.warning_threshold},
                        recommendation="Visible perceptual differences from reference.",
                    )
                )
            logger.debug(f"Butteraugli for {sample.path.name}: {score:.2f}")
        except Exception as e:
            logger.error(f"Butteraugli failed for {sample.path}: {e}")
        return sample

    def _process_video(self, path: Path, ref_path: Path) -> Optional[float]:
        if self._backend != "butteraugli_cli":
            return None
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        scores = []
        idx = 0

        try:
            while True:
                r1, ref_f = ref_cap.read()
                r2, dist_f = dist_cap.read()
                if not r1 or not r2:
                    break
                if idx % self.subsample == 0:
                    h = min(ref_f.shape[0], dist_f.shape[0])
                    w = min(ref_f.shape[1], dist_f.shape[1])
                    ref_r = cv2.resize(ref_f, (w, h))
                    dist_r = cv2.resize(dist_f, (w, h))
                    try:
                        s = self._compute_cli(ref_r, dist_r)
                        scores.append(s)
                    except Exception:
                        pass
                idx += 1
        finally:
            ref_cap.release()
            dist_cap.release()
        return float(np.mean(scores)) if scores else None

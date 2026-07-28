"""AVQT -- Apple Advanced Video Quality Tool.

Apple's perceptual video quality metric for content delivery.
Full-reference metric using deep perceptual features with multi-scale
comparison modelling the human visual system.

Implementation:
    The only real backend is the upstream AVQT CLI tool (macOS). When it is
    not installed the metric is left ``None`` (no proxy is substituted).

avqt_score -- higher = better quality (0-1)

REVIVAL NOTES (requires_external_backend -- no turnkey backend)
Metric: AVQT (Apple Advanced Video Quality Tool).
Category: IMPOSSIBLE.
Why requires_external_backend: Closed-source macOS/Metal CLI binary; no published architecture or weights,
  Windows-unsupported.
To revive: Not reproducible -- no public architecture/weights to reimplement. The only real path is the
  upstream Apple AVQT CLI on macOS. Permanent.
Source: Apple AVQT (closed-source, macOS only).
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class AVQTModule(ReferenceBasedModule):
    name = "avqt"
    requires_external_backend = True  # no turnkey real backend in a standard install
    description = "Apple AVQT perceptual video quality (full-reference)"
    metric_field = "avqt_score"
    default_config = {
        "subsample": 8,
        "hysteresis_weight": 0.1,  # Weight for temporal hysteresis
    }
    metric_groups = {
        "avqt_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.hysteresis_weight = self.config.get("hysteresis_weight", 0.1)
        self._cli_available = False
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # The upstream AVQT CLI tool (macOS only) is the sole real backend.
        try:
            result = subprocess.run(
                ["avqt", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._cli_available = True
                self._ml_available = True
                self._backend = "avqt_cli"
                logger.info("AVQT (CLI) initialised")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("AVQT unavailable: the Apple `avqt` CLI tool was not found")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if not self._cli_available:
            return None
        return self._compute_cli(sample_path, reference_path)

    def _compute_cli(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Run AVQT CLI tool."""
        try:
            result = subprocess.run(
                ["avqt", "--ref", str(reference_path), "--dis", str(sample_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if "score" in line.lower() or "avqt" in line.lower():
                        parts = line.split()
                        for part in reversed(parts):
                            try:
                                return float(np.clip(float(part), 0.0, 1.0))
                            except ValueError:
                                continue
            return None
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("AVQT CLI failed: %s", e)
            return None

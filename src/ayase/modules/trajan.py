"""TRAJAN — point-trajectory autoencoder for assessing generated videos (ICLR 2025).

"Direct Motion Models for Assessing Generated Videos" (DeepMind, arXiv:2505.00209).
TRAJAN estimates point tracks with BootsTAPIR, then (auto)encodes the trajectory
set with a Perceiver-style transformer that reconstructs held-out query-point
trajectories; the realism/consistency score is the Average Jaccard (AJ) of the
autoencoder's track reconstruction.

A CoTracker point-track + jerk/velocity-smoothness aggregation is a different
computation, not TRAJAN's autoencoder-reconstruction score, so it is not emitted
under the TRAJAN name. This module reports itself unavailable until the real
TRAJAN (BootsTAPIR + trajectory autoencoder) backend is wired in.

Output field: ``trajan_score`` (populated only with a real backend).
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TRAJANModule(PipelineModule):
    name = "trajan"
    description = "TRAJAN point-track autoencoder motion realism (ICLR 2025)"
    default_config = {
        "num_frames": 16,
        "num_points": 256,
    }
    metric_groups = {
        "trajan_score": "motion",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import trajan  # type: ignore  # official TRAJAN (BootsTAPIR + autoencoder)

            self._model = trajan
            self._ml_available = True
            self._backend = "trajan"
            logger.info("TRAJAN initialised (trajan backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "TRAJAN unavailable: the point-track autoencoder (BootsTAPIR + TRAJAN) "
            "backend is not installed; trajan_score will not be populated by this "
            "module."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "trajan":
            return sample
        if not sample.is_video:
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.trajan_score = float(score)
        except Exception as e:
            logger.warning("TRAJAN processing failed: %s", e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        return None

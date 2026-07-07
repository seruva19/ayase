"""WorldScore — Unified Evaluation for World Generation (ICCV 2025).

GitHub: https://github.com/haoyi-duan/WorldScore
Distribution-level: worldscore

WorldScore is a multi-model evaluation suite (controllability / quality /
dynamics via dedicated backbones). This module requires that released
WorldScore backend; when it is not installed the metric is left ``None`` —
Laplacian/frame-difference heuristics are not substituted for the published
evaluation.

REVIVAL NOTES (provisional — no turnkey backend)
Metric: WorldScore (ICCV 2025).
Category: EXTERNAL.
Why provisional: Code-complete suite, but needs a heavy external model zoo installed.
To revive: `pip install -e` the repo (pulls GroundingDINO / SAM2 / VFIMamba / DROID-SLAM), then wire
  the aggregate score. Heavy; batch/dataset-level, owns no per-sample QualityMetrics field.
Source: https://github.com/haoyi-duan/WorldScore
"""
import logging
from typing import List, Optional

from ayase.models import Sample
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class WorldScoreModule(BatchMetricModule):
    name = "worldscore"
    provisional = True  # no turnkey real backend in a standard install
    description = "WorldScore world generation evaluation (ICCV 2025)"
    default_config = {"subsample": 8}

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._ml_available = False
        self._backend = None
        self._model = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import worldscore  # type: ignore  # official WorldScore backend

            self._model = worldscore
            self._ml_available = True
            self._backend = "worldscore"
            logger.info("WorldScore initialised (worldscore backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "WorldScore unavailable: the released WorldScore evaluation backend "
            "is not installed"
        )

    def extract_features(self, sample: Sample) -> Optional[object]:
        """Delegate to the real WorldScore backend when available."""
        if not self._ml_available or self._backend != "worldscore":
            return None
        try:
            extract = getattr(self._model, "extract_features", None)
            if extract is None:
                return None
            return extract(str(sample.path))
        except Exception as e:
            logger.warning("WorldScore feature extraction failed: %s", e)
            return None

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> Optional[float]:
        """Compute WorldScore from accumulated features via the real backend."""
        if not self._ml_available or self._backend != "worldscore" or not features:
            return None
        try:
            compute = getattr(self._model, "compute_score", None)
            if compute is None:
                return None
            score = float(compute(features))
        except Exception as e:
            logger.warning("WorldScore computation failed: %s", e)
            return None

        if hasattr(self, "pipeline") and self.pipeline and hasattr(self.pipeline, "stats"):
            self.pipeline.stats["worldscore"] = score
        return score

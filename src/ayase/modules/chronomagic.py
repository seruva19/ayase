"""ChronoMagic-Bench module — NeurIPS 2024 (arXiv:2406.18522).

Two metrics for time-lapse video quality:
  - **MTScore** (Metamorphic Temporal): magnitude of metamorphic change
  - **CHScore** (Chrono-Hallucination): detection of temporal hallucinations

Both are computed by the official ChronoMagic-Bench scorers. When that backend
is unavailable the metrics are left ``None`` — no CLIP-based approximation is
substituted for the published scores.

Video-only: returns None for images.
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ChronoMagicModule(PipelineModule):
    name = "chronomagic"
    description = "ChronoMagic-Bench MTScore + CHScore"
    default_config = {
        "subsample": 16,
    }
    metric_groups = {
        "chronomagic_ch_score": "temporal",
        "chronomagic_mt_score": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 16)
        self._ml_available = False
        self._model = None
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import chronomagic_bench  # type: ignore  # official ChronoMagic-Bench scorers

            self._model = chronomagic_bench
            self._ml_available = True
            self._backend = "chronomagic_bench"
            logger.info("ChronoMagic initialised (chronomagic_bench backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "ChronoMagic unavailable: the official ChronoMagic-Bench backend is not installed"
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "chronomagic_bench":
            return sample
        if not sample.is_video:
            return sample

        try:
            mt_score, ch_score = self._score(sample.path)
            if mt_score is not None:
                sample.quality_metrics.chronomagic_mt_score = float(mt_score)
            if ch_score is not None:
                sample.quality_metrics.chronomagic_ch_score = float(ch_score)
        except Exception as e:
            logger.warning("ChronoMagic processing failed: %s", e)

        return sample

    def _score(self, path) -> tuple[Optional[float], Optional[float]]:
        """Delegate scoring to the real ChronoMagic-Bench backend when available."""
        scorer = getattr(self._model, "score", None)
        if scorer is None:
            return None, None
        result = scorer(str(path))
        mt = result.get("mt_score") if isinstance(result, dict) else None
        ch = result.get("ch_score") if isinstance(result, dict) else None
        return mt, ch

"""Full-reference quality scoring through the external FUNQUE package.

The package compares ``sample.path`` with ``sample.reference_path`` and Ayase
stores its scalar result clipped to [0, 1], where higher represents greater
reference fidelity under FUNQUE. Input decoding, alignment, supported media,
and feature computation are delegated entirely to the installed package. A
missing reference, unavailable package, or backend failure leaves the metric
unset; no proxy score is produced.

Implementation basis: https://github.com/abhinaukumar/funque
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FUNQUEModule(PipelineModule):
    name = "funque"
    description = "Fused quality evaluator via the real FUNQUE package (full-reference)"
    default_config = {"subsample": 8}
    metric_groups = {
        "funque_score": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._funque_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import funque as _funque  # noqa: F401

            self._funque_available = True
            self._backend = "funque"
            logger.info("FUNQUE loaded real funque package")
        except (ImportError, Exception) as e:
            logger.warning("FUNQUE unavailable: install the 'funque' package (%s)", e)

    def process(self, sample: Sample) -> Sample:
        if not self._funque_available:
            return sample

        reference_path = getattr(sample, "reference_path", None)
        if reference_path is None or not Path(str(reference_path)).exists():
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        self._process_funque(sample, Path(str(reference_path)))
        return sample

    def _process_funque(self, sample: Sample, reference_path: Path) -> None:
        """Process using the real FUNQUE package."""
        import funque

        try:
            # FUNQUE API: compute quality score between reference and distorted.
            result = funque.compute(
                str(reference_path),
                str(sample.path),
            )
            if isinstance(result, dict) and "funque" in result:
                score = float(result["funque"])
            else:
                score = float(result)

            sample.quality_metrics.funque_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            # Real backend failed for this sample: leave the metric unset rather
            # than fabricate a value.
            logger.warning("FUNQUE package failed for %s: %s", sample.path, e)

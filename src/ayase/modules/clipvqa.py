"""CLIPVQA — Video Quality Assessment via CLIP.

IEEE TIP 2024 — CLIP-based frame encoder with self-attention
for spatiotemporal quality. 37% better generalizability than
existing VQA methods across 8 datasets.

GitHub: https://github.com/GZHU-DVL/CLIPVQA

Backend:
  1. **clipvqa** — the native CLIPVQA package (trained weights).

``clipvqa_score`` is only produced by the real CLIPVQA model. When the
package is not installed the metric is left unset (no heuristic proxy).

clipvqa_score — higher = better quality
"""

import logging
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CLIPVQAModule(PipelineModule):
    name = "clipvqa"
    description = "CLIPVQA CLIP-based spatiotemporal VQA (TIP 2024)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "clipvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import clipvqa
            self._model = clipvqa
            self._backend = "clipvqa"
            self._ml_available = True
            logger.info("CLIPVQA (native) initialised")
            return
        except ImportError:
            logger.warning(
                "CLIPVQA unavailable: the native CLIPVQA package is not installed "
                "(github.com/GZHU-DVL/CLIPVQA); clipvqa_score skipped."
            )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = float(self._model.predict(str(sample.path)))
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.clipvqa_score = score

        except Exception as e:
            logger.warning(f"CLIPVQA failed for {sample.path}: {e}")

        return sample

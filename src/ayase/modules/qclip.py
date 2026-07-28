"""Q-CLIP --- VLM-Based VQA via Cross-Modal Quality Adaptation (2025).

Q-CLIP is a trained cross-modal quality-adaptation model. A CLIP-IQA-style
zero-shot comparison against quality-level text prompts is a proxy in the
CLIP-IQA family — not the trained Q-CLIP weights — so it is not emitted under
the Q-CLIP name. This module reports itself unavailable until a real Q-CLIP
backend is wired in.

Output field: ``qclip_score`` (populated only with a real backend).

REVIVAL NOTES (requires_external_backend — no turnkey backend)
Metric: Q-CLIP (2025).
Category: TRAINING-ONLY.
Why requires_external_backend: No released adapter weights; only the Shared Cross-Modal Adapter trains (VLM frozen).
To revive: Reimplement the Shared Cross-Modal Adapter on a frozen VLM; train on LSVQ
  (cross-test KoNViD / LIVE-VQC); validate you reproduce the paper's SRCC/PLCC before flipping
  requires_external_backend=False. Effort S — cheapest revival of the whole set.
Source: Q-CLIP, 2025 (no released adapter weights).
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class QCLIPModule(PipelineModule):
    name = "qclip"
    requires_external_backend = True  # no turnkey real backend in a standard install
    description = "Q-CLIP VLM-based VQA (2025)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "qclip_score": "nr_quality",
    }

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
            import qclip  # type: ignore  # upstream Q-CLIP backend (trained weights)

            self._model = qclip
            self._ml_available = True
            self._backend = "qclip"
            logger.info("Q-CLIP initialised (qclip backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "Q-CLIP unavailable: no trained Q-CLIP model is installable; "
            "qclip_score will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "qclip":
            return sample

        try:
            predict = getattr(self._model, "predict", None)
            if predict is None:
                return sample
            score = predict(str(sample.path))
            if score is not None:
                sample.quality_metrics.qclip_score = float(score)
        except Exception as e:
            logger.warning("Q-CLIP failed for %s: %s", sample.path, e)
        return sample

    def _compute_quality_score(self, sample: Sample) -> Optional[float]:
        return None

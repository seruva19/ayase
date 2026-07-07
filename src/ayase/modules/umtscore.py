"""UMTScore — Unified Multi-modal Transformer Score.

Video-text alignment scoring using UMT (Unified Multi-modal Transformer)
features (FETV, Liu et al.). Measures how well video content matches a text
description via cross-modal similarity.

The real backend is the ``umt`` package. A CLIP stand-in is not UMTScore —
the previous CLIP path ignored the caption entirely and returned a constant
1.0 — so it is removed. If ``umt`` is not installed the module reports
itself unavailable.

umtscore — higher = better alignment (0-1 range), populated only with a real
backend.
"""

import logging
from typing import List, Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UMTScoreModule(PipelineModule):
    name = "umtscore"
    description = "UMTScore video-text alignment via UMT features"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "umtscore": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import umt

            self._model = umt
            self._backend = "native"
            self._ml_available = True
            logger.info("UMTScore (native umt) initialised")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "UMTScore unavailable: the umt package is not installed; "
            "umtscore will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            caption = getattr(sample, "caption", None)
            if not caption:
                return sample

            caption_text = caption.text if hasattr(caption, "text") else str(caption)
            score = self._score_native(sample, caption_text)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.umtscore = float(score)

        except Exception as e:
            logger.warning("UMTScore failed for %s: %s", sample.path, e)

        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        return super().process_batch(samples)

    def _score_native(self, sample: Sample, caption: str) -> Optional[float]:
        return float(self._model.score(str(sample.path), caption))

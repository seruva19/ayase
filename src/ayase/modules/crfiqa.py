"""CR-FIQA -- Relative Classifiability Face Image Quality (CVPR 2023).

Ou et al. "CR-FIQA: Face Image Quality Assessment by Learning Sample
Relative Classifiability" -- quality is measured as the relative
classifiability of a face embedding, predicted by a regression head
trained alongside ArcFace.

``crfiqa_score`` is produced only by the trained CR-FIQA model. When the
CR-FIQA weights/package are not available the metric is left unset (no
embedding-norm proxy).
GitHub: https://github.com/fdbtrs/CR-FIQA

crfiqa_score -- higher = better quality (0-1)
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CRFIQAModule(PipelineModule):
    name = "crfiqa"
    description = "CR-FIQA face quality via classifiability (CVPR 2023)"
    default_config = {
        "subsample": 4,
    }
    metric_groups = {
        "crfiqa_score": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 4)
        self._model = None
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return

        # The trained CR-FIQA regression head is required. There is no
        # canonical pip package; a real integration must expose a ``crfiqa``
        # module with a callable predictor. Without it the metric is skipped.
        try:
            import crfiqa

            self._model = crfiqa
            self._ml_available = True
            self._backend = "crfiqa"
            logger.info("CR-FIQA initialised (native model)")
        except ImportError:
            logger.warning(
                "CR-FIQA unavailable: the trained CR-FIQA model is not installed "
                "(github.com/fdbtrs/CR-FIQA); crfiqa_score skipped."
            )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._predict(sample)
            if score is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.crfiqa_score = float(score)
        except Exception as e:
            logger.warning("CR-FIQA failed for %s: %s", sample.path, e)

        return sample

    def _predict(self, sample: Sample) -> Optional[float]:
        return float(self._model.predict(str(sample.path)))

    def on_dispose(self) -> None:
        self._model = None
        import gc

        gc.collect()
        super().on_dispose()

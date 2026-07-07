"""Video ATLAS — Assessment of Temporal Artifacts and Stalls (Bampis et al., 2018).

Video ATLAS predicts continuous QoE for streaming video from a set of
features (VQA, rebuffering, memory) combined by a *trained* regressor
(github.com/christosbampis/ATLAS_release).

There is no installable ``video_atlas`` package with a callable scorer, and
a frame-difference / Laplacian / stall heuristic is not Video ATLAS. Rather
than emit a heuristic under the ATLAS name, this module reports itself
unavailable until a real backend is wired in.

video_atlas_score — higher = better, populated only with a real backend.
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VideoATLASModule(PipelineModule):
    name = "video_atlas"
    description = "Video ATLAS temporal artifacts+stalls assessment (2018)"
    default_config = {"subsample": 16}
    metric_groups = {
        "video_atlas_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 16)
        self._model = None
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # Try a genuine, callable video_atlas package; anything else is a proxy.
        try:
            import video_atlas  # noqa: F401

            if hasattr(video_atlas, "score") or hasattr(video_atlas, "evaluate"):
                self._model = video_atlas
                self._backend = "video_atlas_pkg"
                self._ml_available = True
                logger.info("Video ATLAS initialised with installed video_atlas package")
                return
        except ImportError:
            pass
        except Exception as e:  # pragma: no cover - defensive
            logger.debug("video_atlas import failed: %s", e)

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "Video ATLAS unavailable: no callable video_atlas package with a trained "
            "QoE regressor is installed; video_atlas_score will not be populated."
        )

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            from ayase.models import QualityMetrics

            scorer = getattr(self._model, "score", None) or getattr(self._model, "evaluate", None)
            score = scorer(str(sample.path))
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.video_atlas_score = float(score)
        except Exception as e:
            logger.warning("Video ATLAS failed: %s", e)
        return sample

"""STREAM — Spatio-Temporal Evaluation and Analysis Metric (ICLR 2024).

GitHub: https://github.com/pro2nit/STREAM
Paper:  arXiv:2403.09669

The published STREAM operates on deep per-frame features (SwAV / DINOv2), scores
temporal naturalness (STREAM-T) from the skewness of the temporal power spectrum
and spatial fidelity/diversity (STREAM-S) from the real-vs-generated feature
signals. A grayscale-pixel + FFT-of-frame-difference-variance approximation is
not that computation, so it is not emitted under the STREAM name. This module
reports itself unavailable until the real STREAM backend is wired in.

Dataset-level fields (populated only with a real backend):
    stream_spatial, stream_temporal
"""
import logging
from typing import List, Optional

from ayase.models import Sample, QualityMetrics  # noqa: F401 (kept for API parity)
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class STREAMModule(BatchMetricModule):
    name = "stream_metric"
    description = "STREAM spatial/temporal generation eval (ICLR 2024)"
    default_config = {"subsample": 8}
    metric_info = {
        "stream_spatial": "STREAM-S spatial fidelity/diversity (dataset-level, real backend only)",
        "stream_temporal": "STREAM-T temporal naturalness (dataset-level, real backend only)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None
        self._stream = None

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        try:
            import stream  # type: ignore  # official STREAM backend

            self._stream = stream
            self._backend = "stream"
            logger.info("STREAM initialised (stream backend)")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        self._stream = None
        logger.info(
            "STREAM unavailable: the official STREAM (SwAV/DINOv2) backend is not "
            "installed; stream_spatial/stream_temporal will not be populated."
        )

    def extract_features(self, sample: Sample) -> Optional[object]:
        """Extract STREAM deep features via the real backend, else nothing."""
        if self._stream is None:
            return None
        try:
            extractor = getattr(self._stream, "extract_features", None)
            if extractor is None:
                return None
            return extractor(str(sample.path))
        except Exception as e:
            logger.warning("STREAM feature extraction failed: %s", e)
            return None

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> Optional[float]:
        """Compute STREAM-S / STREAM-T via the real backend and store them."""
        if self._stream is None:
            return None
        try:
            spatial = getattr(self._stream, "stream_S", None)
            temporal = getattr(self._stream, "stream_T", None)
            if spatial is None or temporal is None:
                return None
            spatial_score = float(spatial(features, reference_features))
            temporal_score = float(temporal(features, reference_features))
            if hasattr(self, "pipeline") and self.pipeline and hasattr(
                self.pipeline, "add_dataset_metric"
            ):
                self.pipeline.add_dataset_metric("stream_spatial", spatial_score)
                self.pipeline.add_dataset_metric("stream_temporal", temporal_score)
            return spatial_score
        except Exception as e:
            logger.warning("STREAM computation failed: %s", e)
            return None

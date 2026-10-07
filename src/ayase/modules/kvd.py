"""KVD (Kernel Video Distance) module.

KVD is an alternative to FVD that uses kernel methods (Maximum Mean Discrepancy)
instead of Gaussian assumptions. Better for non-Gaussian feature distributions.
Lower KVD = better video generation quality.

This is a dataset-level metric that compares two distributions of videos.
"""

import logging
from typing import Optional, List

import numpy as np

from ayase.models import Sample
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class KVDModule(BatchMetricModule):
    name = "kvd"
    provenance = "adapted"
    sources = {
        "kvd": "KVD (Unterthiner et al. 2018) — https://arxiv.org/abs/1812.01717",
    }
    deviations = {
        "kvd": "Uses Ayase's StyleGAN-V-derived 16-frame I3D feature path; without a reference the metric is not emitted",
    }
    description = "Kernel Video Distance using Maximum Mean Discrepancy (batch metric)"
    default_config = {
        "device": "auto",
    }
    models = [
        {
            "id": "i3d_torchscript.pt",
            "type": "local",
            "url": "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt",
            "task": "I3D Kinetics-400 video feature extractor (shared with FVD)",
        },
    ]
    metric_info = {
        "kvd": "Kernel Video Distance via MMD over video features (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = self.config.get("device", "auto")
        self.device = None
        self._ml_available = False
        self._feature_model = None
        self._fvd_delegate = None
        self._backend = None

    def setup(self) -> None:
        try:
            from ayase.runtime import resolve_torch_device

            self.device = resolve_torch_device(self.device_config)

            # Load feature extractor (reuse FVD's I3D torchscript backbone).
            from ayase.modules.fvd import FVDModule

            fvd_module = FVDModule(self.config)
            fvd_module.setup()
            self._feature_model = fvd_module._r3d_model
            self._ml_available = fvd_module._ml_available

            if self._ml_available:
                # Reuse a single FVD delegate for feature extraction rather than
                # constructing a fresh module per sample.
                self._fvd_delegate = fvd_module
                self._backend = "i3d"
                logger.info(f"KVD module initialized on {self.device}")
            else:
                self._backend = "unavailable"

        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup KVD: {e}")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract features using the shared FVD I3D delegate."""
        if not sample.is_video or self._fvd_delegate is None:
            return None
        return self._fvd_delegate.extract_features(sample)

    def _poly_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Published KVD kernel: K(x, y) = (x·y/d + 1)^3 (same as KID)."""
        d = X.shape[1]
        return (X @ Y.T / d + 1.0) ** 3

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Unbiased MMD² with the polynomial kernel (lower = more similar)."""
        K_XX = self._poly_kernel(X, X)
        K_YY = self._poly_kernel(Y, Y)
        K_XY = self._poly_kernel(X, Y)

        m = X.shape[0]
        n = Y.shape[0]
        np.fill_diagonal(K_XX, 0)
        np.fill_diagonal(K_YY, 0)

        mmd_sq = K_XX.sum() / (m * (m - 1))
        mmd_sq += K_YY.sum() / (n * (n - 1))
        mmd_sq -= 2 * K_XY.mean()

        return float(max(mmd_sq, 0.0))

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> Optional[float]:
        """Compute KVD using Maximum Mean Discrepancy.

        Args:
            features: List of feature vectors from generated/test videos
            reference_features: Optional list of features from real/reference videos

        Returns:
            KVD score (lower is better), or None without a reference set
        """
        try:
            # Convert to numpy array
            features_array = np.stack(features, axis=0)

            if reference_features is not None and len(reference_features) > 0:
                ref_array = np.stack(reference_features, axis=0)
            else:
                logger.info(
                    "KVD: no reference features provided; "
                    "metric is undefined without a reference set"
                )
                return None

            # Published KVD = unbiased polynomial-kernel MMD² (no rescaling).
            return self._compute_mmd(features_array, ref_array)

        except Exception as e:
            logger.error(f"Failed to compute KVD: {e}")
            return float('inf')

    def on_dispose(self) -> None:
        """Compute KVD after all samples processed."""
        if len(self._feature_cache) < 2:
            logger.info(f"KVD: Not enough samples ({len(self._feature_cache)}) for metric computation")
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            kvd_score = self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None
            )
            if kvd_score is None:
                return

            logger.info(
                f"KVD computed: {kvd_score:.2f} "
                f"(generated: {len(self._feature_cache)}, "
                f"reference: {len(self._reference_cache)})"
            )

            # Store in pipeline stats if available
            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("kvd", kvd_score)

        except Exception as e:
            logger.error(f"Failed to compute KVD: {e}")

        finally:
            self._feature_cache = []
            self._reference_cache = []

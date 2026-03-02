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
    description = "Kernel Video Distance using Maximum Mean Discrepancy (batch metric)"
    default_config = {
        "feature_extractor": "i3d",  # Same as FVD
        "kernel": "rbf",  # RBF kernel for MMD
        "bandwidth": 1.0,  # Kernel bandwidth
        "device": "auto",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.feature_extractor_type = self.config.get("feature_extractor", "i3d")
        self.kernel_type = self.config.get("kernel", "rbf")
        self.bandwidth = self.config.get("bandwidth", 1.0)
        self.device_config = self.config.get("device", "auto")
        self.device = None
        self._ml_available = False
        self._feature_model = None

    def setup(self) -> None:
        try:
            import torch

            # Set device
            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            # Load feature extractor (reuse FVD's I3D)
            from ayase.modules.fvd import FVDModule

            fvd_module = FVDModule(self.config)
            fvd_module.setup()
            self._feature_model = fvd_module._i3d_model
            self._ml_available = fvd_module._ml_available

            if self._ml_available:
                logger.info(f"KVD module initialized on {self.device}")

        except Exception as e:
            logger.warning(f"Failed to setup KVD: {e}")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract features using same method as FVD."""
        if not sample.is_video or self._feature_model is None:
            return None

        # Reuse FVD's feature extraction
        from ayase.modules.fvd import FVDModule

        fvd_temp = FVDModule(self.config)
        fvd_temp._i3d_model = self._feature_model
        fvd_temp._ml_available = True
        return fvd_temp.extract_features(sample)

    def _rbf_kernel(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute RBF (Gaussian) kernel matrix.

        Args:
            X: Feature matrix (N, D)
            Y: Feature matrix (M, D)

        Returns:
            Kernel matrix (N, M)
        """
        # Compute pairwise squared distances
        XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
        YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
        XY = X @ Y.T
        dists = XX + YY - 2 * XY

        # RBF kernel
        K = np.exp(-dists / (2 * self.bandwidth ** 2))
        return K

    def _compute_mmd(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute Maximum Mean Discrepancy.

        Args:
            X: Features from distribution 1
            Y: Features from distribution 2

        Returns:
            MMD score (lower = more similar distributions)
        """
        # Kernel matrices
        K_XX = self._rbf_kernel(X, X)
        K_YY = self._rbf_kernel(Y, Y)
        K_XY = self._rbf_kernel(X, Y)

        m = X.shape[0]
        n = Y.shape[0]

        # MMD^2 = E[K(X,X)] + E[K(Y,Y)] - 2*E[K(X,Y)]
        mmd_sq = (K_XX.sum() - np.trace(K_XX)) / (m * (m - 1))
        mmd_sq += (K_YY.sum() - np.trace(K_YY)) / (n * (n - 1))
        mmd_sq -= 2 * K_XY.mean()

        return float(max(mmd_sq, 0.0))  # MMD^2 can be slightly negative due to numerical errors

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute KVD using Maximum Mean Discrepancy.

        Args:
            features: List of feature vectors from generated/test videos
            reference_features: Optional list of features from real/reference videos

        Returns:
            KVD score (lower is better)
        """
        try:
            # Convert to numpy array
            features_array = np.stack(features, axis=0)

            if reference_features is not None and len(reference_features) > 0:
                ref_array = np.stack(reference_features, axis=0)
            else:
                # Split features in half
                mid = len(features_array) // 2
                ref_array = features_array[:mid]
                features_array = features_array[mid:]

            # Compute MMD
            mmd = self._compute_mmd(features_array, ref_array)

            # Scale to similar range as FVD (multiply by large constant)
            kvd = mmd * 1000.0

            return float(kvd)

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

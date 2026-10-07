"""KID (Kernel Inception Distance) module.

KID measures the distance between distributions of generated and reference images
through Maximum Mean Discrepancy (MMD) in the Inception feature space. Unlike FID,
KID has an unbiased estimator and works correctly on small sample sizes (from ~50
images). Lower KID = better generation quality. Typical range: 0.0-0.1.

This is a dataset-level metric that compares two distributions of images/videos.
"""

import logging
from pathlib import Path
from typing import Optional, List

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class KIDModule(BatchMetricModule):
    name = "kid"
    provenance = "published"
    sources = {
        "kid": "KID (Bińkowski et al., ICLR 2018) — clean-fid https://github.com/GaParmar/clean-fid or torch-fidelity",
        "kid_std": "KID (Bińkowski et al., ICLR 2018) — clean-fid https://github.com/GaParmar/clean-fid or torch-fidelity",
    }
    deviations = {
        "kid": "only the official clean-fid/torch-fidelity backends; without a reference the metric is not emitted; kid_std only via torch-fidelity",
    }
    description = "Kernel Inception Distance for image generation evaluation (batch metric)"
    default_config = {
        "subset_size": 100,  # Subset size for KID estimation
        "num_subsets": 100,  # Number of subsets for averaging
    }
    models = [
        {
            "id": "cleanfid",
            "type": "pip_package",
            "install": "pip install clean-fid",
            "task": "KID backend",
        },
        {
            "id": "torch-fidelity",
            "type": "pip_package",
            "install": "pip install torch-fidelity",
            "task": "KID backend",
        },
    ]
    metric_info = {
        "kid": "Kernel Inception Distance estimate (lower=better)",
        "kid_std": "Standard deviation over KID subsets",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subset_size = self.config.get("subset_size", 100)
        self.num_subsets = self.config.get("num_subsets", 100)
        self._ml_available = False
        self._backend = None  # "cleanfid" or "torch_fidelity"

    def setup(self) -> None:
        # Only the published backends — there is no native substitute.
        try:
            import cleanfid  # noqa: F401
            self._backend = "cleanfid"
            self._ml_available = True
            logger.info("KID module initialized with clean-fid backend")
            return
        except ImportError:
            pass

        try:
            import torch_fidelity  # noqa: F401
            self._backend = "torch_fidelity"
            self._ml_available = True
            logger.info("KID module initialized with torch-fidelity backend")
            return
        except ImportError:
            pass

        self._backend = "unavailable"
        logger.info(
            "KID unavailable: requires clean-fid or torch-fidelity "
            "(pip install clean-fid / torch-fidelity)"
        )

    def extract_features(self, sample: Sample):
        """Cache the sample path — backends compute features in bulk."""
        return str(sample.path) if self._ml_available else None

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> Optional[float]:
        """Compute KID between feature distributions.

        ``features`` / ``reference_features`` are lists of image paths.
        Returns KID (lower=better), or None without a reference set.
        """
        if self._backend == "cleanfid":
            return self._compute_cleanfid(features, reference_features)
        elif self._backend == "torch_fidelity":
            return self._compute_torch_fidelity(features, reference_features)
        return None

    def _compute_cleanfid(self, features: List, reference_features: Optional[List]) -> Optional[float]:
        """Compute KID using clean-fid library."""
        try:
            from cleanfid import fid as cleanfid_module

            # clean-fid expects directories of images
            # features here are paths (strings)
            if reference_features and len(reference_features) > 0:
                # Create temp dirs with symlinks
                import tempfile
                import os

                with tempfile.TemporaryDirectory() as gen_dir, \
                     tempfile.TemporaryDirectory() as ref_dir:

                    for i, p in enumerate(features):
                        src = Path(p)
                        if src.exists():
                            dst = Path(gen_dir) / f"{i}{src.suffix}"
                            try:
                                os.symlink(src, dst)
                            except OSError:
                                import shutil
                                shutil.copy2(src, dst)

                    for i, p in enumerate(reference_features):
                        src = Path(p)
                        if src.exists():
                            dst = Path(ref_dir) / f"{i}{src.suffix}"
                            try:
                                os.symlink(src, dst)
                            except OSError:
                                import shutil
                                shutil.copy2(src, dst)

                    score = cleanfid_module.compute_kid(gen_dir, ref_dir)
                    return float(score)

            logger.warning("KID (clean-fid): no reference features, cannot compute")
            return None

        except Exception as e:
            logger.error(f"Failed to compute KID via clean-fid: {e}")
            return None

    def _compute_torch_fidelity(
        self, features: List, reference_features: Optional[List]
    ) -> Optional[float]:
        """Compute KID using torch-fidelity library."""
        try:
            import torch_fidelity

            if reference_features and len(reference_features) > 0:
                import tempfile
                import os

                with tempfile.TemporaryDirectory() as gen_dir, \
                     tempfile.TemporaryDirectory() as ref_dir:

                    for i, p in enumerate(features):
                        src = Path(p)
                        if src.exists():
                            dst = Path(gen_dir) / f"{i}{src.suffix}"
                            try:
                                os.symlink(src, dst)
                            except OSError:
                                import shutil
                                shutil.copy2(src, dst)

                    for i, p in enumerate(reference_features):
                        src = Path(p)
                        if src.exists():
                            dst = Path(ref_dir) / f"{i}{src.suffix}"
                            try:
                                os.symlink(src, dst)
                            except OSError:
                                import shutil
                                shutil.copy2(src, dst)

                    metrics = torch_fidelity.calculate_metrics(
                        input1=gen_dir,
                        input2=ref_dir,
                        kid=True,
                        kid_subset_size=self.subset_size,
                        kid_subsets=self.num_subsets,
                    )
                    kid_std = metrics.get("kernel_inception_distance_std")
                    if (
                        kid_std is not None
                        and hasattr(self, "pipeline")
                        and self.pipeline
                        and hasattr(self.pipeline, "add_dataset_metric")
                    ):
                        self.pipeline.add_dataset_metric("kid_std", float(kid_std))
                    return float(metrics.get("kernel_inception_distance_mean", float("inf")))

            logger.warning("KID (torch-fidelity): no reference features, cannot compute")
            return None

        except Exception as e:
            logger.error(f"Failed to compute KID via torch-fidelity: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        """Extract and cache features from sample.

        Does not modify the sample directly. Features are accumulated for
        batch computation in on_dispose().
        """
        if not self._ml_available:
            return sample

        features = self.extract_features(sample)
        if features is not None:
            self._feature_cache.append(features)

        # Check if sample has reference for paired comparison
        reference_path = getattr(sample, "reference_path", None)
        if reference_path is not None and isinstance(reference_path, (str, Path)):
            try:
                ref_path = Path(reference_path) if isinstance(reference_path, str) else reference_path
                if ref_path.exists():
                    ref_sample = Sample(
                        path=ref_path,
                        is_video=sample.is_video,
                    )
                    ref_features = self.extract_features(ref_sample)
                    if ref_features is not None:
                        self._reference_cache.append(ref_features)
            except Exception as e:
                logger.debug(f"Failed to extract reference features: {e}")

        return sample

    def on_dispose(self) -> None:
        """Compute KID after all samples processed."""
        if len(self._feature_cache) < 2:
            logger.info(
                f"KID: Not enough samples ({len(self._feature_cache)}) for metric computation"
            )
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            kid_score = self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None,
            )
            if kid_score is None:
                return

            logger.info(
                f"KID computed: {kid_score:.6f} "
                f"(generated: {len(self._feature_cache)}, "
                f"reference: {len(self._reference_cache)})"
            )

            # Store in pipeline stats
            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("kid", kid_score)

        except Exception as e:
            logger.error(f"Failed to compute KID: {e}")

        finally:
            self._feature_cache = []
            self._reference_cache = []

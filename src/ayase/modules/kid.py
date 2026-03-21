"""KID (Kernel Inception Distance) module.

KID measures the distance between distributions of generated and reference images
through Maximum Mean Discrepancy (MMD) in the Inception feature space. Unlike FID,
KID has an unbiased estimator and works correctly on small sample sizes (from ~50
images). Lower KID = better generation quality. Typical range: 0.0-0.1.

This is a dataset-level metric that compares two distributions of images/videos.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class KIDModule(BatchMetricModule):
    name = "kid"
    description = "Kernel Inception Distance for image generation evaluation (batch metric)"
    default_config = {
        "feature_layer": "2048",  # Inception feature layer
        "subset_size": 100,  # Subset size for KID estimation
        "num_subsets": 100,  # Number of subsets for averaging
        "degree": 3,  # Polynomial kernel degree
        "gamma": None,  # Kernel gamma (None = auto: 1/feature_dim)
        "coef0": 1.0,  # Polynomial kernel coefficient
        "device": "auto",
        "batch_size": 32,
        "resize": 299,  # Input size for InceptionV3
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subset_size = self.config.get("subset_size", 100)
        self.num_subsets = self.config.get("num_subsets", 100)
        self.degree = self.config.get("degree", 3)
        self.gamma = self.config.get("gamma", None)
        self.coef0 = self.config.get("coef0", 1.0)
        self.device_config = self.config.get("device", "auto")
        self.batch_size = self.config.get("batch_size", 32)
        self.resize = self.config.get("resize", 299)
        self.device = None
        self._ml_available = False
        self._backend = None  # "cleanfid", "torch_fidelity", or "native"
        self._inception_model = None
        self._transform = None

    def setup(self) -> None:
        # Tier 1: clean-fid
        try:
            import cleanfid  # noqa: F401
            self._backend = "cleanfid"
            self._ml_available = True
            logger.info("KID module initialized with clean-fid backend")
            return
        except ImportError:
            pass

        # Tier 2: torch-fidelity
        try:
            import torch_fidelity  # noqa: F401
            self._backend = "torch_fidelity"
            self._ml_available = True
            logger.info("KID module initialized with torch-fidelity backend")
            return
        except ImportError:
            pass

        # Tier 3: Native InceptionV3 + polynomial kernel MMD
        try:
            import torch
            from torchvision import models, transforms
            from torchvision.models import Inception_V3_Weights

            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            # Load InceptionV3 for feature extraction
            model = models.inception_v3(
                weights=Inception_V3_Weights.IMAGENET1K_V1,
                transform_input=False,
            )
            # Remove final classification layer to get 2048-d features
            model.fc = torch.nn.Identity()
            model = model.to(self.device)
            model.eval()
            self._inception_model = model

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.resize, self.resize)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._backend = "native"
            self._ml_available = True
            logger.info(f"KID module initialized with native InceptionV3 on {self.device}")

        except ImportError as e:
            logger.warning(f"Missing dependencies for KID (torch/torchvision required): {e}")
        except Exception as e:
            logger.warning(f"Failed to setup KID: {e}")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract InceptionV3 features from a sample image or video frame.

        Args:
            sample: Sample to extract features from

        Returns:
            Feature vector (numpy array of shape (2048,)), or None if failed
        """
        if not self._ml_available or self._backend != "native":
            # For cleanfid/torch-fidelity backends, features are computed in bulk
            # at on_dispose time. Cache the path instead.
            return str(sample.path) if self._ml_available else None

        try:
            import torch

            # Load representative frame
            frame = self._load_frame(sample)
            if frame is None:
                return None

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform and extract features
            tensor = self._transform(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                features = self._inception_model(tensor)
                # inception_v3 may return InceptionOutputs during eval
                if hasattr(features, "logits"):
                    features = features.logits
                features = features.cpu().numpy().flatten()

            return features

        except Exception as e:
            logger.debug(f"Failed to extract Inception features from {sample.path}: {e}")
            return None

    def _load_frame(self, sample: Sample) -> Optional[np.ndarray]:
        """Load a representative frame from an image or video.

        For images: loads the image directly.
        For videos: extracts the middle frame.

        Returns:
            BGR frame as numpy array, or None if loading failed
        """
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    cap.release()
                    return None
                # Sample middle frame
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                frame = cv2.imread(str(sample.path))
                return frame
        except Exception as e:
            logger.debug(f"Failed to load frame: {e}")
            return None

    def _polynomial_kernel(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute polynomial kernel K(x, y) = (gamma * x.y + coef0)^degree.

        Args:
            x: Features array (n, d)
            y: Features array (m, d)

        Returns:
            Kernel matrix (n, m)
        """
        d = x.shape[1]
        gamma = self.gamma if self.gamma is not None else 1.0 / d
        return (gamma * (x @ y.T) + self.coef0) ** self.degree

    def _compute_kid_mmd(
        self, features: np.ndarray, ref_features: np.ndarray
    ) -> Tuple[float, float]:
        """Compute KID via polynomial kernel MMD with subsampling.

        Args:
            features: Generated features (N, d)
            ref_features: Reference features (M, d)

        Returns:
            Tuple of (mean KID, std KID) across subsets
        """
        n = min(len(features), len(ref_features), self.subset_size)
        if n < 2:
            return float("inf"), 0.0

        kid_values = []
        for _ in range(self.num_subsets):
            # Random subsets
            idx_x = np.random.choice(len(features), size=n, replace=False)
            idx_y = np.random.choice(len(ref_features), size=n, replace=False)

            x = features[idx_x]
            y = ref_features[idx_y]

            # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
            kxx = self._polynomial_kernel(x, x)
            kyy = self._polynomial_kernel(y, y)
            kxy = self._polynomial_kernel(x, y)

            # Unbiased estimator: exclude diagonal
            np.fill_diagonal(kxx, 0)
            np.fill_diagonal(kyy, 0)

            m = n  # subset size
            mmd2 = (
                kxx.sum() / (m * (m - 1))
                + kyy.sum() / (m * (m - 1))
                - 2.0 * kxy.sum() / (m * m)
            )
            kid_values.append(float(mmd2))

        return float(np.mean(kid_values)), float(np.std(kid_values))

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> float:
        """Compute KID between feature distributions.

        Args:
            features: List of feature vectors (or paths for cleanfid/torch-fidelity)
            reference_features: Optional list of reference features

        Returns:
            KID score (lower is better)
        """
        if self._backend == "cleanfid":
            return self._compute_cleanfid(features, reference_features)
        elif self._backend == "torch_fidelity":
            return self._compute_torch_fidelity(features, reference_features)
        else:
            return self._compute_native(features, reference_features)

    def _compute_cleanfid(self, features: List, reference_features: Optional[List]) -> float:
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
            return float("inf")

        except Exception as e:
            logger.error(f"Failed to compute KID via clean-fid: {e}")
            return float("inf")

    def _compute_torch_fidelity(
        self, features: List, reference_features: Optional[List]
    ) -> float:
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
                    return float(metrics.get("kernel_inception_distance_mean", float("inf")))

            logger.warning("KID (torch-fidelity): no reference features, cannot compute")
            return float("inf")

        except Exception as e:
            logger.error(f"Failed to compute KID via torch-fidelity: {e}")
            return float("inf")

    def _compute_native(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]]
    ) -> float:
        """Compute KID using native InceptionV3 + polynomial kernel MMD."""
        try:
            features_array = np.stack(features, axis=0)

            if reference_features is not None and len(reference_features) > 0:
                ref_array = np.stack(reference_features, axis=0)
            else:
                # Split features in half for self-comparison
                mid = len(features_array) // 2
                if mid < 2:
                    return float("inf")
                ref_array = features_array[:mid]
                features_array = features_array[mid:]

            kid_mean, kid_std = self._compute_kid_mmd(features_array, ref_array)

            logger.info(f"KID computed: {kid_mean:.6f} +/- {kid_std:.6f}")

            # Store std via pipeline if available
            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("kid_std", kid_std)

            return kid_mean

        except Exception as e:
            logger.error(f"Failed to compute KID (native): {e}")
            return float("inf")

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

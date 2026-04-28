"""FVD (Fréchet Video Distance) module.

FVD measures the distance between distributions of real and generated videos.
It uses R3D-18 (3D ResNet-18) features and computes the Fréchet distance.
Lower FVD = better video generation quality. Typical ranges: 50-500 (lower is better).

This is a dataset-level metric that compares two distributions of videos.
"""

import logging
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class FVDModule(BatchMetricModule):
    name = "fvd"
    description = "Fréchet Video Distance for video generation evaluation (batch metric)"
    default_config = {
        "i3d_weights": "kinetics400",  # Pretrained on Kinetics-400
        "num_frames": 16,  # I3D expects 16-frame clips
        "batch_size": 8,
        "device": "auto",
        "subsample_videos": None,  # Max videos to process (None = all)
    }
    models = [
        {
            "id": "torchvision/r3d_18",
            "type": "torchvision",
            "task": "Kinetics-400 R3D-18 video feature extractor",
        },
    ]
    metric_info = {
        "fvd": "Frechet Video Distance between generated and reference video distributions (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.i3d_weights = self.config.get("i3d_weights", "kinetics400")
        self.num_frames = self.config.get("num_frames", 16)
        self.batch_size = self.config.get("batch_size", 8)
        self.device_config = self.config.get("device", "auto")
        self.subsample_videos = self.config.get("subsample_videos", None)
        self.device = None
        self._ml_available = False
        self._r3d_model = None
        self._processed_count = 0

    def setup(self) -> None:
        try:
            import torch
            import torch.nn as nn
            from torchvision.models.video import r3d_18, R3D_18_Weights

            # Set device
            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            # Load R3D-18 pretrained on Kinetics-400 (3D CNN for video features)
            weights = R3D_18_Weights.KINETICS400_V1
            self._r3d_model = r3d_18(weights=weights)
            self._r3d_model.fc = nn.Identity()
            self._r3d_model = self._r3d_model.to(self.device)
            self._r3d_model.eval()

            self._ml_available = True
            logger.info(f"FVD module initialized with R3D-18 on {self.device}")

        except ImportError as e:
            logger.warning(f"Missing dependencies for FVD (torch/torchvision required): {e}")
        except Exception as e:
            logger.warning(f"Failed to setup FVD: {e}")

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract I3D features from a video sample.

        Args:
            sample: Video sample to extract features from

        Returns:
            Feature vector (numpy array), or None if extraction failed
        """
        if not sample.is_video:
            return None

        # Check subsample limit
        if self.subsample_videos is not None and self._processed_count >= self.subsample_videos:
            return None

        try:
            # Load video frames
            frames = self._load_video_frames(sample.path, self.num_frames)
            if frames is None or len(frames) != self.num_frames:
                return None

            import torch

            # Preprocess frames for I3D
            # Expected input: (B, C, T, H, W)
            frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)  # (1, C, T, H, W)
            frames_tensor = frames_tensor.float().to(self.device)

            # Normalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1).to(self.device)
            frames_tensor = (frames_tensor / 255.0 - mean) / std

            # Extract features
            with torch.no_grad():
                features = self._r3d_model(frames_tensor)
                features = features.cpu().numpy().flatten()

            self._processed_count += 1
            return features

        except Exception as e:
            logger.debug(f"Failed to extract I3D features from {sample.path}: {e}")
            return None

    def extract_reference_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract reference features without consuming subsample budget."""
        previous = self._processed_count
        try:
            return self.extract_features(sample)
        finally:
            self._processed_count = previous

    def _load_video_frames(self, video_path: Path, num_frames: int) -> Optional[np.ndarray]:
        """Load uniformly sampled frames from video.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to sample

        Returns:
            Array of shape (T, H, W, C) with sampled frames, or None if failed
        """
        try:
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total_frames < num_frames:
                cap.release()
                return None

            # Sample frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
            frames = []

            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    cap.release()
                    return None

                # Convert BGR to RGB and resize to 224x224
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (224, 224))
                frames.append(frame_resized)

            cap.release()

            return np.stack(frames, axis=0)  # (T, H, W, C)

        except Exception as e:
            logger.debug(f"Failed to load video frames: {e}")
            return None

    def compute_distribution_metric(
        self, features: List[np.ndarray], reference_features: Optional[List[np.ndarray]] = None
    ) -> float:
        """Compute Fréchet distance between feature distributions.

        Args:
            features: List of feature vectors from generated/test videos
            reference_features: Optional list of features from real/reference videos

        Returns:
            FVD score (lower is better)
        """
        try:
            from scipy import linalg

            # Convert to numpy array
            features_array = np.stack(features, axis=0)

            if reference_features is not None and len(reference_features) > 0:
                ref_array = np.stack(reference_features, axis=0)
            else:
                # If no reference provided, split features in half for comparison
                mid = len(features_array) // 2
                ref_array = features_array[:mid]
                features_array = features_array[mid:]

            # Compute statistics
            mu1 = np.mean(features_array, axis=0)
            sigma1 = np.cov(features_array, rowvar=False)

            mu2 = np.mean(ref_array, axis=0)
            sigma2 = np.cov(ref_array, rowvar=False)

            # Compute Fréchet distance
            # FD = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1*sigma2))
            diff = mu1 - mu2
            covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

            if np.iscomplexobj(covmean):
                covmean = covmean.real

            fvd = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)

            return float(fvd)

        except Exception as e:
            logger.error(f"Failed to compute FVD: {e}")
            return float('inf')

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
                # Create temporary reference sample
                ref_sample = Sample(
                    path=Path(reference_path) if isinstance(reference_path, str) else reference_path,
                    is_video=True,
                )
                ref_features = self.extract_reference_features(ref_sample)
                if ref_features is not None:
                    self._reference_cache.append(ref_features)
            except Exception as e:
                logger.debug(f"Failed to extract reference features: {e}")

        return sample

    def on_dispose(self) -> None:
        """Compute FVD after all samples processed."""
        if len(self._feature_cache) < 2:
            logger.info(f"FVD: Not enough samples ({len(self._feature_cache)}) for metric computation")
            self._feature_cache = []
            self._reference_cache = []
            return

        try:
            fvd_score = self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache if self._reference_cache else None
            )

            logger.info(
                f"FVD computed: {fvd_score:.2f} "
                f"(generated: {len(self._feature_cache)}, "
                f"reference: {len(self._reference_cache)})"
            )

            # Store in pipeline stats if available
            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric("fvd", fvd_score)

        except Exception as e:
            logger.error(f"Failed to compute FVD: {e}")

        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._processed_count = 0

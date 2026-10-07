"""FVD (Fréchet Video Distance) module.

FVD measures the distance between distributions of real and generated videos.
This implementation uses a Kinetics-400 I3D TorchScript feature extractor and
computes the Fréchet distance.
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
    provenance = "adapted"
    sources = {
        "fvd": "FVD (Unterthiner et al., 2018) — https://arxiv.org/abs/1812.01717; StyleGAN-V TorchScript port — https://github.com/universome/stylegan-v/blob/master/src/metrics/frechet_video_distance.py",
    }
    deviations = {
        "fvd": "Uses the StyleGAN-V I3D TorchScript convention with 16 uniformly sampled 224×224 frames in [-1,1], rather than claiming equivalence to every FVD implementation; without a reference the metric is not emitted",
    }
    description = "Fréchet Video Distance for video generation evaluation (batch metric)"
    default_config = {
        "num_frames": 16,  # I3D expects 16-frame clips
        "batch_size": 8,
        "device": "auto",
        "subsample_videos": None,  # Max videos to process (None = all)
        "models_dir": "models",
    }
    models = [
        {
            "id": "i3d_torchscript.pt",
            "type": "local",
            "url": "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt",
            "task": "I3D Kinetics-400 video feature extractor (StyleGAN-V FVD)",
        },
    ]
    metric_info = {
        "fvd": "Frechet Video Distance between generated and reference video distributions (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.num_frames = self.config.get("num_frames", 16)
        self.batch_size = self.config.get("batch_size", 8)
        self.device_config = self.config.get("device", "auto")
        self.subsample_videos = self.config.get("subsample_videos", None)
        # Legacy configs may pass backbone="r3d18" (or the removed
        # "content_debiased"/"dinov2"); the published metric is I3D-only.
        backbone = self.config.get("backbone", "i3d")
        if backbone != "i3d":
            logger.warning(
                f"FVD: backbone '{backbone}' is not the published configuration; "
                "using I3D (the StyleGAN-V/cd-fvd convention)"
            )
        self.backbone = "i3d"
        self.metric_name = "fvd"
        self.device = None
        self._ml_available = False
        self._r3d_model = None
        self._processed_count = 0
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            self.device = torch.device(resolve_torch_device(self.device_config))
            self._setup_i3d()

            if self._ml_available:
                self._backend = "i3d"

        except ImportError as e:
            logger.warning(f"Missing dependencies for FVD (torch required): {e}")
        except Exception as e:
            logger.warning(f"Failed to setup FVD: {e}")

    def _setup_i3d(self) -> None:
        import torch
        from ayase.runtime import shared_runtime_resource
        from ayase.config import download_model_file

        models_dir = self.config.get("models_dir", "models")
        ckpt = download_model_file(
            "fvd/i3d_torchscript.pt",
            "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt",
            models_dir,
        )

        def load_i3d():
            return torch.jit.load(str(ckpt), map_location="cpu").to(self.device).eval()

        self._r3d_model = shared_runtime_resource(
            self,
            ("fvd_i3d_torchscript", str(self.device)),
            load_i3d,
        )
        self._ml_available = True
        logger.info(
            f"FVD module initialized with I3D torchscript on {self.device}"
        )

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        """Extract I3D features from a video sample."""
        if not sample.is_video:
            return None

        if self.subsample_videos is not None and self._processed_count >= self.subsample_videos:
            return None

        try:
            frames = self._load_video_frames(sample.path, self.num_frames)
            if frames is None or len(frames) != self.num_frames:
                return None

            features = self._extract_i3d_features(frames)

            if features is None:
                return None

            self._processed_count += 1
            return features

        except Exception as e:
            logger.debug(f"Failed to extract features from {sample.path}: {e}")
            return None

    def _extract_i3d_features(self, frames: np.ndarray) -> Optional[np.ndarray]:
        import torch

        # StyleGAN-V/cd-fvd protocol: (T,H,W,C) uint8 → (1,C,T,H,W) in [-1, 1].
        frames_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0)
        frames_tensor = frames_tensor.float().to(self.device) / 127.5 - 1.0

        with torch.no_grad():
            features = self._r3d_model(frames_tensor)
            return features.cpu().numpy().flatten()

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
    ) -> Optional[float]:
        """Compute Fréchet distance between feature distributions.

        Args:
            features: List of feature vectors from generated/test videos
            reference_features: Optional list of features from real/reference videos

        Returns:
            FVD score (lower is better), or None without a reference set
        """
        try:
            from scipy import linalg

            # Convert to numpy array
            features_array = np.stack(features, axis=0)

            if reference_features is not None and len(reference_features) > 0:
                ref_array = np.stack(reference_features, axis=0)
            else:
                logger.info(
                    "FVD: no reference features provided; "
                    "metric is undefined without a reference set"
                )
                return None

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
            if fvd_score is None:
                return

            logger.info(
                f"{self.metric_name} computed: {fvd_score:.2f} "
                f"(generated: {len(self._feature_cache)}, "
                f"reference: {len(self._reference_cache)})"
            )

            # Store in pipeline stats if available
            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric(self.metric_name, fvd_score)

        except Exception as e:
            logger.error(f"Failed to compute FVD: {e}")

        finally:
            self._feature_cache = []
            self._reference_cache = []
            self._processed_count = 0

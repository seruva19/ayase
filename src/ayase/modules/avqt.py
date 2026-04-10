"""AVQT -- Apple Advanced Video Quality Tool.

Apple's perceptual video quality metric for content delivery.
Full-reference metric using deep perceptual features with multi-scale
comparison modelling the human visual system.

Implementation:
    1. Tier 1: AVQT CLI tool (if installed on macOS).
    2. Tier 2: VGG-16 multi-scale feature comparison (deep FR metric).
       - Extract VGG-16 features at conv2_2, conv3_3, conv4_3, conv5_3.
       - Compute MS-SSIM-like comparison at each feature scale.
       - Temporal pooling across frames with hysteresis weighting.

avqt_score -- higher = better quality (0-1)
"""

import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class AVQTModule(ReferenceBasedModule):
    name = "avqt"
    description = "Apple AVQT perceptual video quality (full-reference)"
    metric_field = "avqt_score"
    default_config = {
        "subsample": 8,
        "hysteresis_weight": 0.1,  # Weight for temporal hysteresis
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.hysteresis_weight = self.config.get("hysteresis_weight", 0.1)
        self._cli_available = False
        self._vgg = None
        self._transform = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Try AVQT CLI tool (macOS only)
        try:
            result = subprocess.run(
                ["avqt", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self._cli_available = True
                self._ml_available = True
                logger.info("AVQT (CLI) initialised")
                return
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass

        # Tier 2: VGG-16 multi-scale deep FR metric
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
            self._vgg = vgg16.to(self._device)

            # Feature layers: conv2_2(8), conv3_3(15), conv4_3(22), conv5_3(29)
            self._feature_layers = [8, 15, 22, 29]
            self._layer_weights = [0.15, 0.25, 0.30, 0.30]

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._ml_available = True
            logger.info("AVQT initialised with VGG-16 deep features on %s", self._device)

        except ImportError:
            logger.warning(
                "AVQT: torch/torchvision not installed. Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("AVQT setup failed: %s", e)

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if not self._ml_available:
            return None

        if self._cli_available:
            return self._compute_cli(sample_path, reference_path)

        return self._compute_vgg(sample_path, reference_path)

    def _compute_cli(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Run AVQT CLI tool."""
        try:
            result = subprocess.run(
                ["avqt", "--ref", str(reference_path), "--dis", str(sample_path)],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split("\n"):
                    line = line.strip()
                    if "score" in line.lower() or "avqt" in line.lower():
                        parts = line.split()
                        for part in reversed(parts):
                            try:
                                return float(np.clip(float(part), 0.0, 1.0))
                            except ValueError:
                                continue
            return None
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("AVQT CLI failed: %s", e)
            return None

    def _compute_vgg(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """Compute AVQT-like quality using VGG-16 multi-scale features."""
        dist_frames = self._read_frames(sample_path)
        ref_frames = self._read_frames(reference_path)

        if not dist_frames or not ref_frames:
            return None

        n_frames = min(len(dist_frames), len(ref_frames))
        dist_frames = dist_frames[:n_frames]
        ref_frames = ref_frames[:n_frames]

        frame_scores = []
        prev_score = None

        for i in range(n_frames):
            ref_feats = self._extract_vgg_features(ref_frames[i])
            dist_feats = self._extract_vgg_features(dist_frames[i])

            if ref_feats is None or dist_feats is None:
                continue

            # Multi-scale feature comparison
            scale_scores = []
            for rf, df, w in zip(ref_feats, dist_feats, self._layer_weights):
                # Cosine similarity between feature maps
                rf_flat = rf.flatten()
                df_flat = df.flatten()
                cos_sim = float(np.dot(rf_flat, df_flat) / (
                    np.linalg.norm(rf_flat) * np.linalg.norm(df_flat) + 1e-10
                ))
                # Also compute L2 distance for complementary info
                l2_dist = float(np.mean((rf - df) ** 2))
                l2_quality = 1.0 / (1.0 + l2_dist * 5.0)

                combined = 0.5 * max(cos_sim, 0.0) + 0.5 * l2_quality
                scale_scores.append(w * combined)

            frame_quality = sum(scale_scores)

            # Temporal hysteresis: blend with previous score
            if prev_score is not None:
                frame_quality = (
                    (1.0 - self.hysteresis_weight) * frame_quality
                    + self.hysteresis_weight * prev_score
                )
            prev_score = frame_quality

            frame_scores.append(frame_quality)

        if not frame_scores:
            return None

        # Mean temporal pooling
        score = float(np.mean(frame_scores))
        return float(np.clip(score, 0.0, 1.0))

    def _extract_vgg_features(self, frame: np.ndarray) -> Optional[List[np.ndarray]]:
        """Extract multi-scale VGG-16 features from a frame."""
        import torch

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)

            features = []
            x = tensor
            with torch.no_grad():
                for idx, layer in enumerate(self._vgg):
                    x = layer(x)
                    if idx in self._feature_layers:
                        feat = x.cpu().numpy()[0]
                        features.append(feat)

            return features if len(features) == len(self._feature_layers) else None
        except Exception as e:
            logger.debug("VGG feature extraction failed: %s", e)
            return None

    def _read_frames(self, path: Path) -> List[np.ndarray]:
        """Read frames from video or image."""
        frames = []
        is_video = path.suffix.lower() in {
            ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv",
        }

        if is_video:
            cap = cv2.VideoCapture(str(path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(path))
            if img is not None:
                frames.append(img)

        return frames

    def on_dispose(self) -> None:
        self._vgg = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

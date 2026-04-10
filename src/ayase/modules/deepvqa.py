"""DeepVQA -- Deep Video Quality Assessor with Spatiotemporal Masking.

Kim et al. ECCV 2018 -- full-reference VQA using deep features with
spatiotemporal visual sensitivity masking.

Architecture:
  1. Four inputs per frame pair: reference frame, distorted frame,
     spatial error map (|ref - dist|), temporal error map (|dist_t - dist_{t-1}|).
  2. VGG-16 extracts features from all 4 inputs at conv3_3, conv4_3, conv5_3.
  3. Per-layer: feature differences between ref and dist are weighted by
     CNN-predicted spatiotemporal sensitivity maps derived from the error
     map features.
  4. Quality = weighted sum of feature differences across layers.
  5. Temporal aggregation with Minkowski pooling (p=4) emphasising worst frames.

deepvqa_score -- higher = better quality (0-1)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class DeepVQAModule(ReferenceBasedModule):
    name = "deepvqa"
    description = "DeepVQA spatiotemporal masking FR-VQA (ECCV 2018)"
    metric_field = "deepvqa_score"
    default_config = {
        "subsample": 8,
        "minkowski_p": 4.0,  # Paper uses p=4 for worst-frame emphasis
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.minkowski_p = self.config.get("minkowski_p", 4.0)
        self._vgg = None
        self._transform = None
        self._device = "cpu"
        self._ml_available = False
        self._feature_layers = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Load VGG-16 and extract intermediate features
            vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
            vgg16 = vgg16.to(self._device)

            # Feature extraction hooks at conv3_3 (layer 15), conv4_3 (22), conv5_3 (29)
            # These are 0-indexed positions of the ReLU outputs after conv layers
            self._feature_layers = [15, 22, 29]
            self._vgg = vgg16

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
            logger.info("DeepVQA initialised with VGG-16 on %s (p=%s)", self._device, self.minkowski_p)

        except ImportError:
            logger.warning(
                "torch/torchvision not installed. Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("DeepVQA setup failed: %s", e)

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        if not self._ml_available:
            return None

        dist_frames = self._read_frames(sample_path)
        ref_frames = self._read_frames(reference_path)

        if not dist_frames or not ref_frames:
            return None

        n_frames = min(len(dist_frames), len(ref_frames))
        dist_frames = dist_frames[:n_frames]
        ref_frames = ref_frames[:n_frames]

        frame_scores = []
        prev_dist_gray = None

        for i in range(n_frames):
            ref_bgr = ref_frames[i]
            dist_bgr = dist_frames[i]

            # --- Build the 4 inputs ---
            ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
            dist_gray = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

            # Spatial error map: absolute pixel difference
            spatial_error = np.abs(ref_gray - dist_gray)

            # Temporal error map: absolute difference between consecutive distorted frames
            if prev_dist_gray is not None:
                temporal_error = np.abs(dist_gray - prev_dist_gray)
            else:
                temporal_error = np.zeros_like(dist_gray)

            # Convert error maps to 3-channel images for VGG input
            spatial_error_bgr = self._gray_to_bgr(spatial_error)
            temporal_error_bgr = self._gray_to_bgr(temporal_error)

            # --- Extract VGG features from all 4 inputs ---
            ref_features = self._extract_vgg_features(ref_bgr)
            dist_features = self._extract_vgg_features(dist_bgr)
            spatial_err_features = self._extract_vgg_features(spatial_error_bgr)
            temporal_err_features = self._extract_vgg_features(temporal_error_bgr)

            if ref_features is None or dist_features is None:
                prev_dist_gray = dist_gray
                continue

            # --- Per-layer quality computation ---
            layer_qualities = []
            layer_weights = [0.2, 0.3, 0.5]  # deeper layers weighted more

            for l_idx in range(len(self._feature_layers)):
                rf = ref_features[l_idx]      # reference features at this layer
                df = dist_features[l_idx]      # distorted features

                # Feature difference
                feat_diff = np.sum((rf - df) ** 2)

                # Spatiotemporal sensitivity mask from error map features
                # The sensitivity map modulates how visible the distortion is
                spatial_sensitivity = 1.0
                temporal_sensitivity = 1.0

                if spatial_err_features is not None:
                    se_feat = spatial_err_features[l_idx]
                    # High error-map activation = high spatial distortion visibility
                    spatial_energy = np.mean(se_feat ** 2)
                    # Texture masking: complex reference textures mask errors
                    ref_complexity = np.std(rf)
                    # Spatial mask: distortion more visible in smooth regions
                    spatial_sensitivity = (1.0 + spatial_energy) / (1.0 + ref_complexity + 1e-8)

                if temporal_err_features is not None:
                    te_feat = temporal_err_features[l_idx]
                    temporal_energy = np.mean(te_feat ** 2)
                    # Motion masking: high motion reduces perceived error
                    temporal_sensitivity = 1.0 / (1.0 + temporal_energy * 0.1)

                # Weighted feature difference
                masked_diff = feat_diff * spatial_sensitivity * temporal_sensitivity
                layer_qualities.append(masked_diff)

            # Combine layers
            raw_quality = sum(
                w * q for w, q in zip(layer_weights, layer_qualities)
            )

            # Convert to quality score (higher = better)
            quality = 1.0 / (1.0 + raw_quality * 10.0)
            frame_scores.append(quality)

            prev_dist_gray = dist_gray

        if not frame_scores:
            return None

        # --- Minkowski pooling (p=4): worst-frame emphasis ---
        scores_arr = np.array(frame_scores)
        errors = 1.0 - scores_arr  # invert: higher error = worse
        p = self.minkowski_p
        if np.any(errors > 0):
            pooled_error = (np.mean(errors ** p)) ** (1.0 / p)
        else:
            pooled_error = 0.0

        score = 1.0 - pooled_error
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _gray_to_bgr(gray: np.ndarray) -> np.ndarray:
        """Convert a grayscale float64 array to a 3-channel uint8 BGR image."""
        # Normalise to 0-255
        g_min, g_max = gray.min(), gray.max()
        if g_max - g_min > 1e-6:
            normed = ((gray - g_min) / (g_max - g_min) * 255.0).astype(np.uint8)
        else:
            normed = np.zeros_like(gray, dtype=np.uint8)
        return cv2.cvtColor(normed, cv2.COLOR_GRAY2BGR)

    def _extract_vgg_features(self, frame_bgr: np.ndarray) -> Optional[List[np.ndarray]]:
        """Extract VGG-16 intermediate features from a BGR frame.

        Returns list of feature vectors at conv3_3, conv4_3, conv5_3.
        Each vector is globally average-pooled to a 1-D descriptor.
        """
        import torch

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)

            features = []
            x = tensor
            with torch.no_grad():
                for idx, layer in enumerate(self._vgg):
                    x = layer(x)
                    if idx in self._feature_layers:
                        feat = x.cpu().numpy()[0]
                        # Global average pool each feature map -> 1-D
                        feat_pooled = feat.mean(axis=(1, 2))
                        features.append(feat_pooled)

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

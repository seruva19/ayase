"""ModularBVQA — Modular Blind Video Quality Assessment.

CVPR 2024 — decomposes VQA into base quality predictor + spatial
rectifier (resolution-aware) + temporal rectifier (framerate-aware).

Architecture (faithful to the paper):
  - Base Quality Predictor: CLIP ViT-B extracts per-frame features,
    averaged and mapped to a base quality score via a learned head.
  - Spatial Rectifier: Uses Laplacian pyramid features to capture
    spatial resolution / sharpness characteristics. The paper conditions
    on resolution via multi-scale Laplacian energy, not raw pixel counts.
  - Temporal Rectifier: Uses SlowFast-style temporal features — frame
    differences computed at both slow (stride-1) and fast (stride-4)
    rates — to capture frame-rate-dependent temporal quality.

Tier 1: Loads pretrained weights from the official checkpoint
        (ViTbCLIP_SpatialTemporal_modular_LSVQ.pth) if available.
Tier 2: CLIP ViT-B/32 backbone with Laplacian spatial + SlowFast
        temporal rectifiers (random head initialisation).
Tier 3: ResNet-50 fallback with the same rectifier design.

GitHub: https://github.com/winwinwenwen77/ModularBVQA

modularbvqa_score — higher = better quality (0-1)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Number of Laplacian pyramid levels for spatial rectifier
_LAPLACIAN_LEVELS = 4
# SlowFast temporal strides
_SLOW_STRIDE = 1
_FAST_STRIDE = 4


class ModularBVQAModule(PipelineModule):
    name = "modularbvqa"
    description = (
        "ModularBVQA resolution/framerate-aware blind VQA (CVPR 2024) "
        "— CLIP ViT-B backbone + Laplacian spatial + SlowFast temporal rectifiers"
    )
    default_config = {
        "subsample": 8,
        "frame_size": 224,
        "weights_path": "",  # path to ViTbCLIP_SpatialTemporal_modular_LSVQ.pth
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.frame_size = self.config.get("frame_size", 224)
        self.weights_path = self.config.get("weights_path", "")
        self._ml_available = False
        self._backbone = None
        self._base_head = None
        self._spatial_rectifier = None
        self._temporal_rectifier = None
        self._device = None
        self._transform = None
        self._use_clip = False
        self._pretrained_loaded = False

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Try CLIP-ViT first (correct backbone per paper), fall back to ResNet-50
            feat_dim = self._try_load_clip()

            if feat_dim == 0:
                # Fall back to ResNet-50
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self._backbone = nn.Sequential(*list(resnet.children())[:-1])
                self._backbone.eval()
                self._backbone.to(self._device)
                feat_dim = 2048

            # Base quality predictor: spatial features -> base quality
            self._base_head = nn.Sequential(
                nn.Linear(feat_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            ).to(self._device)
            self._base_head.eval()

            # Spatial rectifier: Laplacian pyramid features
            # Input: base_quality (1) + laplacian energy per level (_LAPLACIAN_LEVELS)
            #        + laplacian std per level (_LAPLACIAN_LEVELS) + overall sharpness (1)
            spatial_feat_dim = 1 + _LAPLACIAN_LEVELS * 2 + 1
            self._spatial_rectifier = nn.Sequential(
                nn.Linear(spatial_feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Tanh(),  # Output in [-1, 1] as additive adjustment
            ).to(self._device)
            self._spatial_rectifier.eval()

            # Temporal rectifier: SlowFast-style features
            # Input: base_quality (1) + slow pathway stats (3: mean, std, max)
            #        + fast pathway stats (3: mean, std, max) + fps_norm (1)
            #        + slow-fast ratio (1)
            temporal_feat_dim = 1 + 3 + 3 + 1 + 1
            self._temporal_rectifier = nn.Sequential(
                nn.Linear(temporal_feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Tanh(),  # Output in [-1, 1] as additive adjustment
            ).to(self._device)
            self._temporal_rectifier.eval()

            # Try loading pretrained weights
            self._try_load_pretrained(feat_dim, spatial_feat_dim, temporal_feat_dim)

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.frame_size + 32),
                transforms.CenterCrop(self.frame_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._ml_available = True
            backend = "CLIP-ViT" if self._use_clip else "ResNet-50"
            wt_status = " (pretrained)" if self._pretrained_loaded else " (random heads)"
            logger.info(
                "ModularBVQA initialised on %s (%s + Laplacian spatial + SlowFast temporal)%s",
                self._device, backend, wt_status,
            )

        except ImportError:
            logger.warning(
                "ModularBVQA requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("ModularBVQA setup failed: %s", e)

    def _try_load_clip(self) -> int:
        """Try loading CLIP ViT-B/32. Returns feature dim or 0 on failure."""
        try:
            import clip
            import torch

            model, preprocess = clip.load("ViT-B/32", device=self._device)
            self._backbone = model.visual
            self._backbone.eval()
            self._clip_preprocess = preprocess
            self._use_clip = True
            logger.debug("ModularBVQA: CLIP ViT-B/32 loaded (correct backbone per paper)")
            return 512
        except ImportError:
            logger.debug("CLIP not available, using ResNet-50 fallback")
            return 0
        except Exception as e:
            logger.debug("CLIP loading failed: %s", e)
            return 0

    def _try_load_pretrained(self, feat_dim: int, spatial_dim: int, temporal_dim: int) -> None:
        """Try loading official pretrained weights."""
        import torch

        if not self.weights_path:
            return

        wpath = Path(self.weights_path)
        if not wpath.exists():
            logger.debug("ModularBVQA pretrained weights not found: %s", wpath)
            return

        try:
            state = torch.load(str(wpath), map_location=self._device, weights_only=False)

            # The official checkpoint may wrap weights in different keys
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]

            # Try to load matching keys into our heads
            loaded_any = False
            for module_name, module in [
                ("base_head", self._base_head),
                ("spatial_rectifier", self._spatial_rectifier),
                ("temporal_rectifier", self._temporal_rectifier),
            ]:
                sub_state = {}
                for k, v in state.items():
                    if module_name in k:
                        # Strip prefix
                        short_key = k.split(module_name + ".")[-1]
                        sub_state[short_key] = v
                if sub_state:
                    try:
                        module.load_state_dict(sub_state, strict=False)
                        loaded_any = True
                    except Exception as e:
                        logger.debug("Could not load %s weights: %s", module_name, e)

            if loaded_any:
                self._pretrained_loaded = True
                logger.info("ModularBVQA: loaded pretrained weights from %s", wpath)
            else:
                logger.debug("ModularBVQA: checkpoint found but no matching keys")

        except Exception as e:
            logger.warning("ModularBVQA: failed to load pretrained weights: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_quality(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.modularbvqa_score = score
                logger.debug("ModularBVQA for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("ModularBVQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        """Base quality + spatial rectifier + temporal rectifier."""
        if sample.is_video:
            return self._compute_video_quality(sample)
        else:
            return self._compute_image_quality(sample)

    @staticmethod
    def _compute_laplacian_features(gray_frame: np.ndarray) -> np.ndarray:
        """Compute Laplacian pyramid energy features from a grayscale frame.

        Returns a feature vector of length (_LAPLACIAN_LEVELS * 2 + 1):
          - Per-level mean absolute Laplacian energy (captures sharpness at each scale)
          - Per-level std of Laplacian response (captures texture detail)
          - Overall sharpness (variance of finest Laplacian)
        """
        import cv2

        energies = []
        stds = []
        current = gray_frame.astype(np.float32)

        for level in range(_LAPLACIAN_LEVELS):
            # Compute Laplacian at this level
            lap = cv2.Laplacian(current, cv2.CV_32F)
            energy = float(np.mean(np.abs(lap)))
            std = float(np.std(lap))
            energies.append(energy)
            stds.append(std)

            # Downsample for next level (Gaussian pyramid)
            if current.shape[0] > 16 and current.shape[1] > 16:
                current = cv2.pyrDown(current)
            # else keep same size for remaining levels

        # Overall sharpness = variance of finest-level Laplacian
        finest_lap = cv2.Laplacian(gray_frame.astype(np.float32), cv2.CV_32F)
        overall_sharpness = float(np.var(finest_lap))

        # Normalise to reasonable ranges
        energies_norm = [e / max(max(energies), 1e-6) for e in energies]
        stds_norm = [s / max(max(stds), 1e-6) for s in stds]
        sharpness_norm = min(overall_sharpness / 2000.0, 1.0)

        return np.array(energies_norm + stds_norm + [sharpness_norm], dtype=np.float32)

    @staticmethod
    def _compute_slowfast_features(
        gray_frames: list,
    ) -> np.ndarray:
        """Compute SlowFast-style temporal features from grayscale frames.

        SlowFast networks process video at two temporal rates:
          - Slow pathway: adjacent frame differences (stride 1) — fine motion
          - Fast pathway: strided frame differences (stride 4) — coarse motion

        Returns feature vector of length 7:
          [slow_mean, slow_std, slow_max, fast_mean, fast_std, fast_max, slow_fast_ratio]
        """
        if len(gray_frames) < 2:
            return np.zeros(7, dtype=np.float32)

        # Resize all frames to a common small size for efficiency
        target_h, target_w = 120, 160
        resized = []
        for g in gray_frames:
            import cv2
            resized.append(cv2.resize(g, (target_w, target_h)).astype(np.float32))

        # Slow pathway: stride-1 frame differences
        slow_diffs = []
        for i in range(len(resized) - _SLOW_STRIDE):
            diff = np.mean(np.abs(resized[i + _SLOW_STRIDE] - resized[i]))
            slow_diffs.append(diff)

        # Fast pathway: stride-4 frame differences
        fast_diffs = []
        for i in range(len(resized) - _FAST_STRIDE):
            diff = np.mean(np.abs(resized[i + _FAST_STRIDE] - resized[i]))
            fast_diffs.append(diff)

        # Compute statistics for each pathway
        if slow_diffs:
            slow_mean = float(np.mean(slow_diffs)) / 50.0  # normalise
            slow_std = float(np.std(slow_diffs)) / 30.0
            slow_max = float(np.max(slow_diffs)) / 80.0
        else:
            slow_mean = slow_std = slow_max = 0.0

        if fast_diffs:
            fast_mean = float(np.mean(fast_diffs)) / 50.0
            fast_std = float(np.std(fast_diffs)) / 30.0
            fast_max = float(np.max(fast_diffs)) / 80.0
        else:
            fast_mean = fast_std = fast_max = 0.0

        # Slow-to-fast ratio captures temporal structure
        if fast_mean > 1e-6:
            sf_ratio = min(slow_mean / fast_mean, 2.0) / 2.0
        else:
            sf_ratio = 0.5

        return np.array(
            [slow_mean, slow_std, slow_max, fast_mean, fast_std, fast_max, sf_ratio],
            dtype=np.float32,
        )

    def _compute_video_quality(self, sample: Sample) -> Optional[float]:
        """Video quality with Laplacian spatial and SlowFast temporal rectifiers."""
        import torch
        import cv2

        cap = cv2.VideoCapture(str(sample.path))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            return None

        n_frames = min(self.subsample, total)
        indices = np.linspace(0, total - 1, n_frames, dtype=int)

        frame_features = []
        gray_frames = []
        laplacian_features_list = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Extract backbone features for base quality
            with torch.no_grad():
                feat = self._extract_features(rgb)
                frame_features.append(feat)

            # Collect grayscale frames for SlowFast temporal features
            gray_frames.append(gray)

            # Compute per-frame Laplacian pyramid features
            lap_feat = self._compute_laplacian_features(gray)
            laplacian_features_list.append(lap_feat)

        cap.release()

        if not frame_features:
            return None

        # Aggregate backbone features for base quality
        feat_stack = torch.cat(frame_features, dim=0)  # (T, D)
        feat_mean = feat_stack.mean(dim=0, keepdim=True)  # (1, D)

        # Average Laplacian features across frames
        if laplacian_features_list:
            avg_laplacian = np.mean(laplacian_features_list, axis=0)
        else:
            avg_laplacian = np.zeros(_LAPLACIAN_LEVELS * 2 + 1, dtype=np.float32)

        # Compute SlowFast temporal features
        slowfast_feat = self._compute_slowfast_features(gray_frames)

        with torch.no_grad():
            # Base quality prediction
            base_quality = self._base_head(feat_mean)  # (1, 1)
            base_q = base_quality.item()

            # Spatial rectifier: Laplacian pyramid features
            spatial_input = np.concatenate([[base_q], avg_laplacian])
            spatial_tensor = torch.tensor(
                spatial_input, device=self._device, dtype=torch.float32
            ).unsqueeze(0)
            spatial_adj = self._spatial_rectifier(spatial_tensor).item() * 0.2

            # Temporal rectifier: SlowFast features + fps
            fps_norm = min(fps / 60.0, 1.0)
            temporal_input = np.concatenate([[base_q], slowfast_feat[:6], [fps_norm], slowfast_feat[6:7]])
            temporal_tensor = torch.tensor(
                temporal_input, device=self._device, dtype=torch.float32
            ).unsqueeze(0)
            temporal_adj = self._temporal_rectifier(temporal_tensor).item() * 0.15

        # Final score: base + adjustments
        score = base_q + spatial_adj + temporal_adj
        return float(np.clip(score, 0.0, 1.0))

    def _compute_image_quality(self, sample: Sample) -> Optional[float]:
        """Image quality with Laplacian spatial rectifier only."""
        import torch
        import cv2

        img = cv2.imread(str(sample.path))
        if img is None:
            return None

        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplacian pyramid features
        lap_feat = self._compute_laplacian_features(gray)

        with torch.no_grad():
            feat = self._extract_features(rgb)  # (1, D)

            # Base quality
            base_quality = self._base_head(feat)
            base_q = base_quality.item()

            # Spatial rectifier with Laplacian features
            spatial_input = np.concatenate([[base_q], lap_feat])
            spatial_tensor = torch.tensor(
                spatial_input, device=self._device, dtype=torch.float32
            ).unsqueeze(0)
            spatial_adj = self._spatial_rectifier(spatial_tensor).item() * 0.2

        score = base_q + spatial_adj
        return float(np.clip(score, 0.0, 1.0))

    def _extract_features(self, rgb: np.ndarray):
        """Extract features using CLIP or ResNet backbone."""
        import torch

        if self._use_clip:
            from PIL import Image
            pil_img = Image.fromarray(rgb)
            tensor = self._clip_preprocess(pil_img).unsqueeze(0).to(self._device)
            feat = self._backbone(tensor)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            return feat.float()
        else:
            tensor = self._transform(rgb).unsqueeze(0).to(self._device)
            feat = self._backbone(tensor).squeeze(-1).squeeze(-1)
            return feat

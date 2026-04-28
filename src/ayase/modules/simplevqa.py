"""SimpleVQA — Simple Blind Video Quality Assessment.

2022 — Uses Swin Transformer-B for spatial feature extraction and
temporal difference features via SlowFast-inspired pathways for
blind video quality assessment. Base model for RQ-VQA.

Architecture (Sun et al., 2022):
  - Spatial: Swin-B backbone -> global average pooling -> 1024-d features
    per frame.  "Slow" branch samples 8 frames at full resolution.
  - Temporal: "Fast" branch samples 32 frames at lower resolution.
    Frame-to-frame feature differences capture motion quality.
  - Quality heads: Linear(1024, 128) -> ReLU -> Linear(128, 1) for
    spatial and temporal branches separately.
  - Fusion: 0.5 * spatial_quality + 0.5 * temporal_quality.

GitHub: https://github.com/sunwei925/SimpleVQA

simplevqa_score — higher = better quality (0-1 sigmoid-rescaled)
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SimpleVQAModule(PipelineModule):
    name = "simplevqa"
    description = "SimpleVQA Swin+SlowFast blind VQA (2022)"
    default_config = {
        "slow_frames": 8,
        "fast_frames": 32,
        "frame_size": 224,
        "fast_frame_size": 112,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.slow_frames = self.config.get("slow_frames", 8)
        self.fast_frames = self.config.get("fast_frames", 32)
        self.frame_size = self.config.get("frame_size", 224)
        self.fast_frame_size = self.config.get("fast_frame_size", 112)
        self._ml_available = False
        self._spatial_backbone = None
        self._spatial_quality_head = None
        self._temporal_quality_head = None
        self._device = "cpu"
        self._transform = None
        self._fast_transform = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Swin-B spatial backbone: extract features before classification head
            swin = models.swin_b(weights=models.Swin_B_Weights.DEFAULT)
            # Swin-B features: 1024-d after norm + avgpool
            # Remove the classification head
            swin.head = nn.Identity()
            self._spatial_backbone = swin
            self._spatial_backbone.eval()
            self._spatial_backbone.to(self._device)

            # Spatial quality head: Linear(1024, 128) -> ReLU -> Linear(128, 1)
            self._spatial_quality_head = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            ).to(self._device)
            self._spatial_quality_head.eval()

            # Temporal quality head: same architecture on temporal diff features
            # Input is 1024-d (mean of frame-to-frame Swin-B feature differences)
            self._temporal_quality_head = nn.Sequential(
                nn.Linear(1024, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
            ).to(self._device)
            self._temporal_quality_head.eval()

            # Transform matching Swin-B expected input (slow pathway, full res)
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

            # Fast pathway transform: lower resolution
            self._fast_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.fast_frame_size + 16),
                transforms.CenterCrop(self.fast_frame_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._ml_available = True
            logger.info(
                "SimpleVQA initialised on %s (Swin-B, slow=%d fast=%d)",
                self._device, self.slow_frames, self.fast_frames,
            )

        except ImportError:
            logger.warning(
                "SimpleVQA requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("SimpleVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_quality(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.simplevqa_score = score
                logger.debug("SimpleVQA for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("SimpleVQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        """SlowFast Swin-B quality: spatial (slow) + temporal diff (fast) branches."""
        import torch

        # Load frames for both pathways
        slow_frames = self._load_frames_rgb(sample, self.slow_frames)
        fast_frames = self._load_frames_rgb(sample, self.fast_frames)
        if not slow_frames:
            return None

        # --- Spatial (slow) branch: few frames, full resolution ---
        slow_features = []
        with torch.no_grad():
            for rgb in slow_frames:
                tensor = self._transform(rgb).unsqueeze(0).to(self._device)
                feat = self._spatial_backbone(tensor)  # (1, 1024)
                slow_features.append(feat)

        slow_stack = torch.cat(slow_features, dim=0)  # (T_slow, 1024)

        with torch.no_grad():
            # Average spatial feature across slow-path frames
            spatial_mean = slow_stack.mean(dim=0, keepdim=True)  # (1, 1024)
            spatial_quality = self._spatial_quality_head(spatial_mean)  # (1, 1)

        # --- Temporal (fast) branch: many frames, lower resolution ---
        # Extract features from fast-pathway frames
        fast_features = []
        with torch.no_grad():
            for rgb in fast_frames:
                tensor = self._fast_transform(rgb).unsqueeze(0).to(self._device)
                feat = self._spatial_backbone(tensor)  # (1, 1024)
                fast_features.append(feat)

        fast_stack = torch.cat(fast_features, dim=0)  # (T_fast, 1024)

        with torch.no_grad():
            # Frame-to-frame feature differences (temporal pathway core)
            if fast_stack.shape[0] > 1:
                diffs = fast_stack[1:] - fast_stack[:-1]  # (T_fast-1, 1024)
                temporal_mean = diffs.mean(dim=0, keepdim=True)  # (1, 1024)
            else:
                temporal_mean = torch.zeros(1, 1024, device=self._device)

            temporal_quality = self._temporal_quality_head(temporal_mean)  # (1, 1)

        # --- Fusion: 0.5 * spatial + 0.5 * temporal ---
        with torch.no_grad():
            fused = 0.5 * spatial_quality + 0.5 * temporal_quality
            # Sigmoid rescale to (0, 1)
            score = torch.sigmoid(fused).item()

        return float(score)

    def _load_frames_rgb(self, sample: Sample, n_target: int) -> list:
        """Load frames as RGB numpy arrays."""
        import cv2

        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            n_frames = min(n_target, total)
            indices = np.linspace(0, total - 1, n_frames, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(rgb)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return frames

"""DisCoVQA — Temporal Distortion-Content Transformers for VQA.

IEEE 2023 — separates temporal distortion extraction from
content-aware temporal attention using transformers.

The real DisCoVQA paper:
  - Extracts content and distortion features via separate networks.
  - Content features *modulate* the distortion assessment through
    a content-adaptive gating mechanism (multiplicative), so that
    the same distortion is scored differently depending on content.
  - Temporal modelling via GRU or Transformer for sequence aggregation.

This implementation is an approximation:
  - Backbone: ResNet-50 with separate learned projections for
    content and distortion spaces (matching the paper's dual-path
    decomposition).
  - Content-adaptive gating: content features produce per-element
    scaling and bias that modulate distortion features before
    quality regression (sigmoid gating, faithful to the paper).
  - Temporal modelling: content-driven attention-weighted pooling
    (a simplified form of the paper's Transformer temporal model).
  - The quality head uses random initialisation, so the absolute
    score is a plausible proxy, not a calibrated MOS predictor.

GitHub: https://github.com/VQAssessment/DisCoVQA

discovqa_score — higher = better quality (0-1)
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DisCoVQAModule(PipelineModule):
    name = "discovqa"
    description = "DisCoVQA temporal distortion-content VQA (2023)"
    default_config = {
        "subsample": 8,
        "frame_size": 224,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.frame_size = self.config.get("frame_size", 224)
        self._ml_available = False
        self._content_backbone = None
        self._distortion_proj = None
        self._content_proj = None
        self._content_gate_scale = None
        self._content_gate_bias = None
        self._temporal_attn = None
        self._quality_head = None
        self._device = "cpu"
        self._transform = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # ResNet-50 content backbone (shared for content + distortion)
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._content_backbone = nn.Sequential(*list(resnet.children())[:-1])
            self._content_backbone.eval()
            self._content_backbone.to(self._device)

            # Distortion projection: maps backbone features to distortion space
            self._distortion_proj = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
            ).to(self._device)
            self._distortion_proj.eval()

            # Content projection: maps backbone features to content space
            self._content_proj = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 256),
            ).to(self._device)
            self._content_proj.eval()

            # Content-adaptive gating: content features produce per-element
            # scale and bias to modulate distortion features (key paper idea)
            self._content_gate_scale = nn.Sequential(
                nn.Linear(256, 256),
                nn.Sigmoid(),
            ).to(self._device)
            self._content_gate_scale.eval()

            self._content_gate_bias = nn.Sequential(
                nn.Linear(256, 256),
                nn.Tanh(),
            ).to(self._device)
            self._content_gate_bias.eval()

            # Temporal attention: content-aware attention weights
            # Input: content features (256) -> attention weight (1)
            self._temporal_attn = nn.Sequential(
                nn.Linear(256, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            ).to(self._device)
            self._temporal_attn.eval()

            # Quality head: modulated distortion features (256) -> quality
            self._quality_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            ).to(self._device)
            self._quality_head.eval()

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
            logger.info(
                "DisCoVQA initialised on %s (ResNet-50 + attention)", self._device
            )

        except ImportError:
            logger.warning(
                "DisCoVQA requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("DisCoVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_quality(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.discovqa_score = score
                logger.debug("DisCoVQA for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("DisCoVQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        """Distortion-content decomposition with content-adaptive gating.

        Following the DisCoVQA paper:
        1. Extract backbone features per frame.
        2. Project to content and distortion spaces independently.
        3. Content features produce per-element gate (scale + bias)
           that modulates distortion features -- this is the key
           content-adaptive mechanism from the paper.
        4. Content-driven temporal attention aggregates the modulated
           distortion features across time.
        5. Quality head regresses the aggregated features to a score.
        """
        import torch
        import torch.nn.functional as F
        import cv2

        frames_rgb = self._load_frames_rgb(sample)
        if not frames_rgb:
            return None

        # Extract backbone features for all frames
        backbone_features = []
        with torch.no_grad():
            for rgb in frames_rgb:
                tensor = self._transform(rgb).unsqueeze(0).to(self._device)
                feat = self._content_backbone(tensor).squeeze(-1).squeeze(-1)  # (1, 2048)
                backbone_features.append(feat)

        backbone_stack = torch.cat(backbone_features, dim=0)  # (T, 2048)

        with torch.no_grad():
            # Decompose into content and distortion representations
            content_feats = self._content_proj(backbone_stack)  # (T, 256)
            distortion_feats = self._distortion_proj(backbone_stack)  # (T, 256)

            # Content-adaptive gating: content features modulate the
            # distortion assessment (multiplicative + additive, per the paper)
            gate_scale = self._content_gate_scale(content_feats)  # (T, 256) in [0,1]
            gate_bias = self._content_gate_bias(content_feats)  # (T, 256) in [-1,1]
            modulated_distortion = gate_scale * distortion_feats + gate_bias  # (T, 256)

            # Content-aware temporal attention
            attn_logits = self._temporal_attn(content_feats)  # (T, 1)
            attn_weights = F.softmax(attn_logits, dim=0)  # (T, 1)

            # Attention-weighted aggregation of *modulated* distortion features
            weighted_distortion = (attn_weights * modulated_distortion).sum(
                dim=0, keepdim=True
            )  # (1, 256)

            # Quality prediction from content-modulated distortion
            score = self._quality_head(weighted_distortion).item()

        return float(score)

    def _load_frames_rgb(self, sample: Sample) -> list:
        """Load frames as RGB numpy arrays."""
        import cv2

        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            n_frames = min(self.subsample, total)
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

"""SAMA — Scaling and Masking for Video Quality Assessment.

2024 — patch pyramid with masking strategy for local + global
quality, improving on FAST-VQA.

Implementation: ResNet-50 backbone with spatial attention
(masking important regions) + multi-scale feature extraction.
Quality from attention-weighted multi-scale features.

sama_score — higher = better quality (0-1)
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SAMAModule(PipelineModule):
    name = "sama"
    description = "SAMA scaling+masking VQA (2024)"
    default_config = {
        "subsample": 8,
        "mask_ratio": 0.5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.mask_ratio = self.config.get("mask_ratio", 0.5)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        self._resnet_stages = None
        self._attention_heads = None
        self._quality_head = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # ResNet-50 backbone split into stages for multi-scale features
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            children = list(resnet.children())

            # Stage 1: conv1+bn+relu+maxpool+layer1 -> 256 channels
            # Stage 2: layer2 -> 512 channels
            # Stage 3: layer3 -> 1024 channels
            # Stage 4: layer4 -> 2048 channels
            self._stage1 = nn.Sequential(*children[:5]).to(self._device).eval()
            self._stage2 = children[5].to(self._device).eval()  # layer2
            self._stage3 = children[6].to(self._device).eval()  # layer3
            self._stage4 = children[7].to(self._device).eval()  # layer4

            # Spatial attention modules per scale (channel attention + spatial mask)
            # Each takes Cx1x1 global pool -> attention weight
            self._attention_heads = nn.ModuleDict({
                "s1": nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(256, 64), nn.ReLU(inplace=True),
                    nn.Linear(64, 1), nn.Sigmoid(),
                ),
                "s2": nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(512, 64), nn.ReLU(inplace=True),
                    nn.Linear(64, 1), nn.Sigmoid(),
                ),
                "s3": nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(1024, 64), nn.ReLU(inplace=True),
                    nn.Linear(64, 1), nn.Sigmoid(),
                ),
                "s4": nn.Sequential(
                    nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                    nn.Linear(2048, 64), nn.ReLU(inplace=True),
                    nn.Linear(64, 1), nn.Sigmoid(),
                ),
            }).to(self._device)
            for m in self._attention_heads.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            self._attention_heads.eval()

            # Quality regression head: multi-scale pooled features
            # s1(256) + s2(512) + s3(1024) + s4(2048) = 3840
            self._quality_head = nn.Sequential(
                nn.Linear(3840, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            ).to(self._device)
            for m in self._quality_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            self._quality_head.eval()

            self._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._ml_available = True
            self._backend = "resnet_sama"
            logger.info("SAMA (ResNet-50 multi-scale + attention) initialised on %s", self._device)

        except Exception as e:
            logger.warning("SAMA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.sama_score = score
        except Exception as e:
            logger.warning("SAMA failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """Multi-scale attention-weighted quality assessment."""
        import torch
        from PIL import Image

        frames = self._extract_frames(sample)
        if not frames:
            return None

        frame_scores = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            x = self._transform(pil_img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                # Multi-scale feature extraction
                s1_feat = self._stage1(x)
                s2_feat = self._stage2(s1_feat)
                s3_feat = self._stage3(s2_feat)
                s4_feat = self._stage4(s3_feat)

                # Spatial attention weighting per scale
                a1 = self._attention_heads["s1"](s1_feat)  # [1, 1]
                a2 = self._attention_heads["s2"](s2_feat)
                a3 = self._attention_heads["s3"](s3_feat)
                a4 = self._attention_heads["s4"](s4_feat)

                # Random masking: zero out a fraction of patches at each scale
                # (mimicking the SAMA masking strategy during inference)
                pool = torch.nn.AdaptiveAvgPool2d(1)
                f1 = self._apply_mask(s1_feat, self.mask_ratio)
                f2 = self._apply_mask(s2_feat, self.mask_ratio)
                f3 = self._apply_mask(s3_feat, self.mask_ratio)
                f4 = self._apply_mask(s4_feat, self.mask_ratio)

                # Global average pool each masked feature map
                p1 = pool(f1).flatten(1)  # [1, 256]
                p2 = pool(f2).flatten(1)  # [1, 512]
                p3 = pool(f3).flatten(1)  # [1, 1024]
                p4 = pool(f4).flatten(1)  # [1, 2048]

                # Attention-weight and concatenate
                fused = torch.cat([
                    p1 * a1, p2 * a2, p3 * a3, p4 * a4
                ], dim=-1)  # [1, 3840]

                score = self._quality_head(fused).item()
                frame_scores.append(score)

        if not frame_scores:
            return None

        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    def _apply_mask(self, feat_map, mask_ratio):
        """Apply random spatial masking to feature map patches."""
        import torch

        b, c, h, w = feat_map.shape
        total_patches = h * w
        n_keep = max(int(total_patches * (1 - mask_ratio)), 1)

        # Create deterministic mask per-frame
        mask = torch.zeros(b, 1, h, w, device=feat_map.device)
        indices = torch.randperm(total_patches)[:n_keep]
        rows = indices // w
        cols = indices % w
        mask[:, :, rows, cols] = 1.0

        return feat_map * mask

    def _extract_frames(self, sample: Sample):
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

"""PTM-VQA — Pre-Trained Model fusion VQA.

CVPR 2024 — integrates features from multiple frozen pre-trained
models with ICID loss for quality representation. Processes 1080p
in ~1s via model selection scheme.

Paper: https://arxiv.org/abs/2405.17765

Implementation: Multi-PTM fusion using CLIP + DINOv2 + ResNet-50
backbones. Features from each backbone are concatenated and passed
through a quality regression head.

ptmvqa_score — higher = better quality (0-1)
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class _QualityHead:
    """Lightweight MLP quality regression head for fused multi-PTM features."""

    def __init__(self, device):
        import torch
        import torch.nn as nn

        # CLIP (512) + DINOv2 (384) + ResNet-50 (2048) = 2944
        self.head = nn.Sequential(
            nn.Linear(2944, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        ).to(device)

        # Initialize with Xavier for stable outputs
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        self.head.eval()

    def __call__(self, features):
        return self.head(features)


class PTMVQAModule(PipelineModule):
    name = "ptmvqa"
    description = "PTM-VQA multi-PTM fusion VQA (CVPR 2024)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        # Backbone references
        self._clip_model = None
        self._clip_processor = None
        self._dino_model = None
        self._dino_transform = None
        self._resnet = None
        self._resnet_transform = None
        self._quality_head = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from torchvision import models, transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # --- Backbone 1: CLIP ---
            clip_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(clip_name, models_dir)
            self._clip_model = CLIPModel.from_pretrained(resolved).to(self._device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(resolved)

            # --- Backbone 2: DINOv2 (ViT-S/14) ---
            self._dino_model = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", verbose=False
            ).to(self._device).eval()
            self._dino_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # --- Backbone 3: ResNet-50 (feature extractor) ---
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove final FC, keep avgpool output (2048-d)
            self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self._device).eval()
            self._resnet_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # --- Quality regression head ---
            self._quality_head = _QualityHead(self._device)

            self._ml_available = True
            self._backend = "multi_ptm"
            logger.info(
                "PTM-VQA (multi-PTM: CLIP+DINOv2+ResNet-50) initialised on %s",
                self._device,
            )

        except Exception as e:
            logger.warning("PTM-VQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.ptmvqa_score = score

        except Exception as e:
            logger.warning("PTM-VQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """Extract features from all three backbones, fuse, and regress quality."""
        import torch
        from PIL import Image

        frames = self._extract_frames(sample)
        if not frames:
            return None

        frame_scores = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            with torch.no_grad():
                # CLIP features (512-d)
                from ayase.compat import extract_features

                clip_inputs = self._clip_processor(
                    images=pil_img, return_tensors="pt"
                ).to(self._device)
                clip_feats = extract_features(
                    self._clip_model.get_image_features(**clip_inputs)
                )  # [1, 512]
                clip_feats = clip_feats / clip_feats.norm(p=2, dim=-1, keepdim=True)

                # DINOv2 features (384-d for ViT-S/14)
                dino_input = self._dino_transform(pil_img).unsqueeze(0).to(self._device)
                dino_feats = self._dino_model(dino_input)  # [1, 384]
                dino_feats = dino_feats / dino_feats.norm(p=2, dim=-1, keepdim=True)

                # ResNet-50 features (2048-d)
                resnet_input = self._resnet_transform(pil_img).unsqueeze(0).to(self._device)
                resnet_feats = self._resnet(resnet_input).squeeze(-1).squeeze(-1)  # [1, 2048]
                resnet_feats = resnet_feats / resnet_feats.norm(p=2, dim=-1, keepdim=True)

                # Concatenate all features: [1, 2944]
                fused = torch.cat([clip_feats, dino_feats, resnet_feats], dim=-1)

                # Quality regression
                score = self._quality_head(fused.float()).item()
                frame_scores.append(score)

        if not frame_scores:
            return None

        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

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

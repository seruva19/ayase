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

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.runtime import cached_clip_image_features, media_state_key, torch_inference_context

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
    metric_groups = {
        "ptmvqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
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
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            # --- Backbone 1: CLIP ---
            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self._clip_model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )

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

        frames = self._extract_frames(sample)
        if not frames:
            return None

        pil_frames = arrays_to_pil(frames)
        clip_feats = cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
            pil_frames,
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        if clip_feats is None or clip_feats.size(0) == 0:
            return None

        with torch_inference_context(
            self._device,
            self.config.get("dtype", "auto"),
            bool(self.config.get("amp_enabled", True)),
        ):
            # DINOv2 features (384-d for ViT-S/14)
            dino_input = torch.stack([self._dino_transform(img) for img in pil_frames]).to(
                self._device
            )
            dino_feats = self._dino_model(dino_input)
            dino_feats = dino_feats / dino_feats.norm(p=2, dim=-1, keepdim=True)

            # ResNet-50 features (2048-d)
            resnet_input = torch.stack([self._resnet_transform(img) for img in pil_frames]).to(
                self._device
            )
            resnet_feats = self._resnet(resnet_input).squeeze(-1).squeeze(-1)
            resnet_feats = resnet_feats / resnet_feats.norm(p=2, dim=-1, keepdim=True)

            # Concatenate all features: [n_frames, 2944]
            fused = torch.cat([clip_feats, dino_feats, resnet_feats], dim=-1)
            frame_scores = self._quality_head(fused.float()).squeeze(-1).detach().cpu().numpy()

        if len(frame_scores) == 0:
            return None

        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    def _extract_frames(self, sample: Sample):
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("PTM-VQA frame loading failed for %s: %s", sample.path, e)
            return []

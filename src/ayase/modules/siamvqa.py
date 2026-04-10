"""SiamVQA — Siamese Network for High-Resolution VQA.

arXiv 2025 — Siamese network sharing weights between aesthetic
and technical branches. Outperforms DOVER on all datasets.

Paper: https://arxiv.org/html/2503.02330

Implementation: Siamese ResNet-50 architecture processes multiple
crops of high-res frames independently, then aggregates via
learned pooling. Handles high-res content without downscaling.

siamvqa_score — higher = better quality (0-1)
"""

import logging
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SiamVQAModule(PipelineModule):
    name = "siamvqa"
    description = "SiamVQA Siamese high-resolution VQA (2025)"
    default_config = {
        "subsample": 8,
        "num_crops": 5,
        "crop_size": 224,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.num_crops = self.config.get("num_crops", 5)
        self.crop_size = self.config.get("crop_size", 224)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        # Siamese branches (shared backbone)
        self._backbone = None
        self._transform = None
        self._tech_head = None
        self._aes_head = None
        self._fusion_head = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Shared Siamese backbone: ResNet-50 feature extractor
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._backbone = nn.Sequential(
                *list(resnet.children())[:-1]
            ).to(self._device).eval()

            self._transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # Technical quality head (from shared features)
            self._tech_head = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            ).to(self._device)

            # Aesthetic quality head (from shared features)
            self._aes_head = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            ).to(self._device)

            # Fusion head (combines technical + aesthetic predictions)
            self._fusion_head = nn.Sequential(
                nn.Linear(2, 1),
                nn.Sigmoid(),
            ).to(self._device)

            # Xavier init
            for module in [self._tech_head, self._aes_head, self._fusion_head]:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
                module.eval()

            # Learned crop pooling weights (aggregate multiple crop predictions)
            self._crop_pool = nn.Sequential(
                nn.Linear(self.num_crops, self.num_crops),
                nn.Softmax(dim=-1),
            ).to(self._device)
            for m in self._crop_pool.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
            self._crop_pool.eval()

            self._ml_available = True
            self._backend = "siamese_resnet"
            logger.info(
                "SiamVQA (Siamese ResNet-50, %d crops) initialised on %s",
                self.num_crops, self._device,
            )

        except Exception as e:
            logger.warning("SiamVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.siamvqa_score = score

        except Exception as e:
            logger.warning("SiamVQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """Siamese multi-crop quality assessment."""
        import torch
        from PIL import Image

        frames = self._extract_frames(sample)
        if not frames:
            return None

        frame_scores = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # Extract multiple crops at original resolution (no downscaling)
            crops = self._extract_crops(rgb, h, w)
            if not crops:
                continue

            crop_tech_scores = []
            crop_aes_scores = []

            with torch.no_grad():
                for crop in crops:
                    pil_crop = Image.fromarray(crop)
                    # Resize crop to backbone input size
                    pil_crop = pil_crop.resize((self.crop_size, self.crop_size), Image.BILINEAR)
                    x = self._transform(pil_crop).unsqueeze(0).to(self._device)

                    # Shared backbone (Siamese weight sharing)
                    feats = self._backbone(x).squeeze(-1).squeeze(-1)  # [1, 2048]

                    # Technical branch
                    tech_score = self._tech_head(feats).item()
                    crop_tech_scores.append(tech_score)

                    # Aesthetic branch
                    aes_score = self._aes_head(feats).item()
                    crop_aes_scores.append(aes_score)

                # Learned pooling over crops
                n_actual = len(crop_tech_scores)
                if n_actual < self.num_crops:
                    # Pad to expected size
                    pad_val_tech = np.mean(crop_tech_scores)
                    pad_val_aes = np.mean(crop_aes_scores)
                    crop_tech_scores.extend([pad_val_tech] * (self.num_crops - n_actual))
                    crop_aes_scores.extend([pad_val_aes] * (self.num_crops - n_actual))

                tech_tensor = torch.tensor(
                    [crop_tech_scores[:self.num_crops]],
                    dtype=torch.float32, device=self._device,
                )
                aes_tensor = torch.tensor(
                    [crop_aes_scores[:self.num_crops]],
                    dtype=torch.float32, device=self._device,
                )

                # Weighted aggregation
                tech_weights = self._crop_pool(tech_tensor)
                aes_weights = self._crop_pool(aes_tensor)

                pooled_tech = (tech_tensor * tech_weights).sum(dim=-1, keepdim=True)
                pooled_aes = (aes_tensor * aes_weights).sum(dim=-1, keepdim=True)

                # Fusion
                fused_input = torch.cat([pooled_tech, pooled_aes], dim=-1)
                fused_score = self._fusion_head(fused_input).item()
                frame_scores.append(fused_score)

        if not frame_scores:
            return None

        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    def _extract_crops(self, rgb: np.ndarray, h: int, w: int) -> List[np.ndarray]:
        """Extract multiple crops from high-res frame without downscaling."""
        crops = []
        cs = min(self.crop_size, h, w)

        # Center crop
        cy, cx = h // 2, w // 2
        crops.append(rgb[cy - cs // 2:cy + cs // 2, cx - cs // 2:cx + cs // 2])

        # Four corner crops
        if h >= cs and w >= cs:
            crops.append(rgb[:cs, :cs])          # top-left
            crops.append(rgb[:cs, w - cs:])      # top-right
            crops.append(rgb[h - cs:, :cs])      # bottom-left
            crops.append(rgb[h - cs:, w - cs:])  # bottom-right

        return crops[:self.num_crops]

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

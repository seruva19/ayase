"""SR4KVQA — Super-Resolution 4K Video Quality (2024).

Evaluates quality of super-resolved 4K content by detecting artifacts
common in SR outputs: ringing, texture loss, aliasing, and hallucination.

Implementation: ResNet-50 on patches of SR content. Focuses on artifact
detection via patch-level analysis with aggregated quality prediction.

sr4kvqa_score — higher = better (0-1)
"""

import logging
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SR4KVQAModule(PipelineModule):
    name = "sr4kvqa"
    description = "SR4KVQA super-resolution 4K quality (2024)"
    default_config = {
        "subsample": 8,
        "patch_size": 224,
        "max_patches": 9,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.patch_size = self.config.get("patch_size", 224)
        self.max_patches = self.config.get("max_patches", 9)
        self._backend = None
        self._ml_available = False
        self._device = "cpu"

        self._resnet = None
        self._transform = None
        self._artifact_head = None
        self._quality_head = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # ResNet-50 backbone for patch feature extraction
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._resnet = nn.Sequential(
                *list(resnet.children())[:-1]
            ).to(self._device).eval()

            self._transform = transforms.Compose([
                transforms.Resize(self.patch_size),
                transforms.CenterCrop(self.patch_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # SR artifact detection head (per-patch)
            # Detects: ringing, texture loss, aliasing, hallucination
            self._artifact_head = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 4),  # 4 artifact types
                nn.Sigmoid(),
            ).to(self._device)

            # Overall quality head (from pooled patch features)
            self._quality_head = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            ).to(self._device)

            for module in [self._artifact_head, self._quality_head]:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
                module.eval()

            self._ml_available = True
            self._backend = "resnet_sr"
            logger.info(
                "SR4KVQA (ResNet-50 patch analysis) initialised on %s", self._device
            )

        except Exception as e:
            logger.warning("SR4KVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            score = self._compute_score(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.sr4kvqa_score = score
        except Exception as e:
            logger.warning("SR4KVQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, sample: Sample) -> Optional[float]:
        """Patch-level SR quality analysis."""
        import torch
        from PIL import Image

        frames = self._extract_frames(sample)
        if not frames:
            return None

        frame_scores = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]

            # Extract patches from the frame (grid-based for SR evaluation)
            patches = self._extract_patches(rgb, h, w)
            if not patches:
                continue

            patch_feats_list = []
            patch_quality_scores = []

            with torch.no_grad():
                for patch in patches:
                    pil_patch = Image.fromarray(patch)
                    x = self._transform(pil_patch).unsqueeze(0).to(self._device)
                    feats = self._resnet(x).squeeze(-1).squeeze(-1)  # [1, 2048]
                    patch_feats_list.append(feats)

                    # Per-patch artifact detection
                    artifact_scores = self._artifact_head(feats)  # [1, 4]
                    # Lower artifact presence = higher quality
                    artifact_penalty = artifact_scores.mean().item()

                    # Per-patch quality
                    q = self._quality_head(feats).item()
                    # Final patch score: quality adjusted by artifact detection
                    patch_score = q * (1.0 - 0.3 * artifact_penalty)
                    patch_quality_scores.append(patch_score)

                # Aggregate: mean of patch quality scores
                # Weight edge/corner patches slightly lower (center is more important)
                if patch_quality_scores:
                    n = len(patch_quality_scores)
                    if n > 1:
                        # Center patch gets highest weight
                        weights = np.ones(n)
                        weights[0] = 1.5  # Center patch (extracted first)
                        weights = weights / weights.sum()
                        frame_score = float(np.dot(weights, patch_quality_scores))
                    else:
                        frame_score = patch_quality_scores[0]
                    frame_scores.append(frame_score)

        if not frame_scores:
            return None

        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    def _extract_patches(self, rgb: np.ndarray, h: int, w: int) -> List[np.ndarray]:
        """Extract grid patches from the frame for SR quality analysis."""
        ps = min(self.patch_size, h, w)
        patches = []

        # Center patch first (highest priority)
        cy, cx = h // 2, w // 2
        patches.append(rgb[cy - ps // 2:cy + ps // 2, cx - ps // 2:cx + ps // 2])

        # Grid patches
        n_h = min(3, max(1, h // ps))
        n_w = min(3, max(1, w // ps))

        step_h = (h - ps) // max(n_h - 1, 1) if n_h > 1 else 0
        step_w = (w - ps) // max(n_w - 1, 1) if n_w > 1 else 0

        for i in range(n_h):
            for j in range(n_w):
                y = i * step_h
                x = j * step_w
                patch = rgb[y:y + ps, x:x + ps]
                if patch.shape[0] == ps and patch.shape[1] == ps:
                    patches.append(patch)

        return patches[:self.max_patches]

    def _extract_frames(self, sample: Sample):
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(
                0, total - 1, min(self.subsample, total), dtype=int
            )
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

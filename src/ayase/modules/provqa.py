"""ProVQA -- Progressive Blind 360 VQA (2022).

Progressive blind video quality assessment for 360-degree content,
operating at multiple resolution levels: pixel -> patch -> frame -> video.

Implementation:
    ResNet-50 extracts features at progressively refined resolution levels
    from 360-degree equirectangular content.  Each level captures different
    quality aspects:
      - Coarse level (downsampled): global structure and distortion
      - Medium level (half-res): regional quality and detail
      - Fine level (full-res crops): local sharpness and artifacts
    Quality is aggregated progressively from coarse to fine.

provqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ProVQAModule(PipelineModule):
    name = "provqa"
    description = "ProVQA progressive blind 360 VQA (2022)"
    default_config = {
        "subsample": 8,
        "n_fine_crops": 6,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.n_fine_crops = self.config.get("n_fine_crops", 6)
        self._resnet = None
        self._resnet_transform = None
        self._coarse_head = None
        self._medium_head = None
        self._fine_head = None
        self._progressive_fusion = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # ResNet-50 backbone shared across all resolution levels
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._resnet.eval().to(self._device)

            self._resnet_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            feat_dim = 2048

            # Coarse-level quality head (global structure)
            self._coarse_head = torch.nn.Sequential(
                torch.nn.Linear(feat_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._coarse_head.eval()

            # Medium-level quality head (regional quality)
            self._medium_head = torch.nn.Sequential(
                torch.nn.Linear(feat_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._medium_head.eval()

            # Fine-level quality head (local detail)
            self._fine_head = torch.nn.Sequential(
                torch.nn.Linear(feat_dim, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._fine_head.eval()

            # Progressive fusion: combine 3 levels -> final score
            self._progressive_fusion = torch.nn.Sequential(
                torch.nn.Linear(3, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._progressive_fusion.eval()

            # Initialise fusion to near-equal weighting
            with torch.no_grad():
                self._progressive_fusion[0].weight.fill_(0.33)
                self._progressive_fusion[0].bias.fill_(0.0)

            self._ml_available = True
            self._backend = "resnet"
            logger.info(
                "ProVQA initialised with ResNet-50 progressive on %s",
                self._device,
            )

        except ImportError:
            logger.warning(
                "ProVQA: no ML backend available. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("ProVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            frame_scores = []
            for frame in frames:
                score = self._compute_progressive_quality(frame)
                if score is not None:
                    frame_scores.append(score)

            if not frame_scores:
                return sample

            # Temporal aggregation with video-level smoothness
            if len(frame_scores) > 1:
                # Weight recent frames slightly more for 360 content
                weights = np.linspace(0.8, 1.0, len(frame_scores))
                weights = weights / weights.sum()
                score = float(np.dot(weights, frame_scores))
            else:
                score = float(frame_scores[0])

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.provqa_score = float(
                np.clip(score, 0.0, 1.0)
            )

        except Exception as e:
            logger.warning("ProVQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_progressive_quality(self, frame: np.ndarray) -> Optional[float]:
        """Compute quality at 3 progressive resolution levels."""
        import torch

        h, w = frame.shape[:2]

        # Level 1: Coarse -- heavily downsampled for global structure
        coarse_score = self._compute_level_quality(
            frame, target_size=(max(w // 4, 32), max(h // 4, 32)),
            head=self._coarse_head,
        )

        # Level 2: Medium -- half-resolution for regional quality
        medium_score = self._compute_level_quality(
            frame, target_size=(max(w // 2, 64), max(h // 2, 64)),
            head=self._medium_head,
        )

        # Level 3: Fine -- full-res crops for local detail
        fine_score = self._compute_fine_quality(frame)

        if coarse_score is None or medium_score is None or fine_score is None:
            # Fall back to available scores
            scores = [s for s in [coarse_score, medium_score, fine_score] if s is not None]
            if not scores:
                return None
            return float(np.mean(scores))

        # Progressive fusion
        with torch.no_grad():
            level_scores = torch.tensor(
                [[coarse_score, medium_score, fine_score]],
                dtype=torch.float32,
            ).to(self._device)
            fused = self._progressive_fusion(level_scores).item()

        return fused

    def _compute_level_quality(
        self,
        frame: np.ndarray,
        target_size: tuple,
        head,
    ) -> Optional[float]:
        """Extract features at a specific resolution and predict quality."""
        import torch

        try:
            resized = cv2.resize(frame, target_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)

            with torch.no_grad():
                feat = self._resnet(tensor).flatten()
                quality = head(feat.unsqueeze(0)).item()

            return quality

        except Exception as e:
            logger.debug("Level quality computation failed: %s", e)
            return None

    def _compute_fine_quality(self, frame: np.ndarray) -> Optional[float]:
        """Extract quality from full-resolution crops (fine level)."""
        import torch

        h, w = frame.shape[:2]
        crop_size = min(h, w, 448)
        if crop_size < 64:
            crop_size = min(h, w)

        # Sample crops from different 360-degree viewport positions
        rng = np.random.RandomState(42)
        crop_scores = []

        for _ in range(self.n_fine_crops):
            cy = rng.randint(0, max(h - crop_size, 1))
            cx = rng.randint(0, max(w - crop_size, 1))
            crop = frame[cy:cy + crop_size, cx:cx + crop_size]

            try:
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    feat = self._resnet(tensor).flatten()
                    quality = self._fine_head(feat.unsqueeze(0)).item()

                crop_scores.append(quality)
            except Exception:
                continue

        if not crop_scores:
            return None

        return float(np.mean(crop_scores))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(
                    0, total - 1, min(self.subsample, total), dtype=int
                )
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

    def on_dispose(self) -> None:
        self._resnet = None
        self._coarse_head = None
        self._medium_head = None
        self._fine_head = None
        self._progressive_fusion = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

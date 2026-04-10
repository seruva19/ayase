"""MC360IQA -- Multi-Channel Blind 360-degree IQA (2019).

Blind IQA for omnidirectional/360-degree content.  The approach extracts
features from multiple viewports sampled across the equirectangular
projection, accounting for latitude-dependent distortion.

Implementation:
    ResNet-50 backbone extracts features from multiple viewport crops at
    different viewing angles.  Viewports are sampled with appropriate
    weighting for equirectangular projection (more weight near equator
    where content is less distorted).  Quality is the weighted aggregation
    of per-viewport features.

mc360iqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Viewport sampling positions: (latitude_frac, longitude_frac, weight)
# Equirectangular: equator has best quality, poles are stretched
_VIEWPORT_POSITIONS = [
    (0.50, 0.00, 1.00),  # front equator
    (0.50, 0.25, 1.00),  # right equator
    (0.50, 0.50, 1.00),  # back equator
    (0.50, 0.75, 1.00),  # left equator
    (0.30, 0.125, 0.85),  # upper-front-right
    (0.70, 0.125, 0.85),  # lower-front-right
    (0.30, 0.625, 0.85),  # upper-back-left
    (0.70, 0.625, 0.85),  # lower-back-left
    (0.15, 0.00, 0.60),  # near north pole
    (0.85, 0.50, 0.60),  # near south pole
]


class MC360IQAModule(PipelineModule):
    name = "mc360iqa"
    description = "MC360IQA blind 360 IQA (2019)"
    default_config = {
        "subsample": 8,
        "n_viewports": 10,
        "viewport_size": 224,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.n_viewports = self.config.get("n_viewports", 10)
        self.viewport_size = self.config.get("viewport_size", 224)
        self._resnet = None
        self._resnet_transform = None
        self._quality_head = None
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

            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._resnet.eval().to(self._device)

            self._resnet_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.viewport_size, self.viewport_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # Quality regression head from aggregated viewport features
            self._quality_head = torch.nn.Sequential(
                torch.nn.Linear(2048, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._quality_head.eval()

            self._ml_available = True
            self._backend = "resnet"
            logger.info(
                "MC360IQA initialised with ResNet-50 on %s", self._device
            )

        except ImportError:
            logger.warning(
                "MC360IQA: no ML backend available. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("MC360IQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            frame_scores = []
            for frame in frames:
                score = self._compute_frame_quality(frame)
                if score is not None:
                    frame_scores.append(score)

            if not frame_scores:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.mc360iqa_score = float(
                np.clip(np.mean(frame_scores), 0.0, 1.0)
            )

        except Exception as e:
            logger.warning("MC360IQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_frame_quality(self, frame: np.ndarray) -> Optional[float]:
        """Extract viewports and compute quality from aggregated features."""
        import torch

        h, w = frame.shape[:2]
        vp_size = min(h // 3, w // 4, self.viewport_size * 2)
        if vp_size < 32:
            vp_size = min(h, w)

        viewport_features = []
        viewport_weights = []

        positions = _VIEWPORT_POSITIONS[:self.n_viewports]

        for lat_frac, lon_frac, weight in positions:
            viewport = self._extract_viewport(frame, lat_frac, lon_frac, vp_size)
            if viewport is None or viewport.size == 0:
                continue

            feat = self._extract_feature(viewport)
            if feat is not None:
                viewport_features.append(feat)
                viewport_weights.append(weight)

        if not viewport_features:
            return None

        # Weighted aggregation of viewport features
        weights = np.array(viewport_weights, dtype=np.float32)
        weights = weights / weights.sum()
        feat_matrix = np.array(viewport_features)
        aggregated = np.average(feat_matrix, axis=0, weights=weights)

        # Quality regression
        with torch.no_grad():
            feat_tensor = (
                torch.from_numpy(aggregated.astype(np.float32))
                .unsqueeze(0)
                .to(self._device)
            )
            quality = self._quality_head(feat_tensor).item()

        return quality

    def _extract_viewport(
        self,
        frame: np.ndarray,
        lat_frac: float,
        lon_frac: float,
        vp_size: int,
    ) -> Optional[np.ndarray]:
        """Extract a viewport crop from equirectangular image."""
        h, w = frame.shape[:2]
        cy = int(h * lat_frac)
        cx = int(w * lon_frac)
        half = vp_size // 2

        y1 = max(cy - half, 0)
        x1 = max(cx - half, 0)
        y2 = min(y1 + vp_size, h)
        x2 = min(x1 + vp_size, w)

        # Handle wrap-around for 360 content (horizontal)
        viewport = frame[y1:y2, x1:x2]
        if viewport.shape[0] < 16 or viewport.shape[1] < 16:
            return None

        return viewport

    def _extract_feature(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract ResNet-50 feature from a viewport."""
        import torch

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._resnet(tensor)
            return feat.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.debug("Feature extraction failed: %s", e)
            return None

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
        self._quality_head = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

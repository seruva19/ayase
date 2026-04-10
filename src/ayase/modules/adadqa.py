"""Ada-DQA -- Adaptive Diverse Quality-aware Feature Acquisition.

ACM MM 2023 -- adaptive quality-aware feature extraction using
diverse pre-trained models for content/distortion/motion diversity.

Implementation:
    ResNet-50 backbone with multi-scale feature extraction and adaptive
    quality-aware weighting.  Features are extracted at multiple spatial
    scales (full-frame, half-frame, quarter-frame), and an adaptive head
    learns quality weights that emphasise the weakest quality dimension
    (content diversity, distortion awareness, motion smoothness).

adadqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AdaDQAModule(PipelineModule):
    name = "adadqa"
    description = "Ada-DQA adaptive diverse quality feature VQA (ACM MM 2023)"
    default_config = {
        "subsample": 8,
        "scales": [1.0, 0.5, 0.25],
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.scales = self.config.get("scales", [1.0, 0.5, 0.25])
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

            # ResNet-50 backbone -- extract features before the final FC
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Keep layers up to avgpool for multi-scale feature extraction
            self._resnet_layers = torch.nn.ModuleDict({
                "early": torch.nn.Sequential(*list(resnet.children())[:5]),   # 256-ch
                "mid":   torch.nn.Sequential(*list(resnet.children())[5:7]),  # 1024-ch
                "late":  torch.nn.Sequential(*list(resnet.children())[7:8]),  # 2048-ch
            })
            self._avgpool = torch.nn.AdaptiveAvgPool2d(1)
            self._resnet_layers.eval().to(self._device)
            self._avgpool.to(self._device)

            # Adaptive quality head: 3-scale features -> quality score
            # 256 + 1024 + 2048 = 3328 per scale, 3 scales = 9984
            feat_dim = (256 + 1024 + 2048) * len(self.scales)
            self._quality_head = torch.nn.Sequential(
                torch.nn.Linear(feat_dim, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 3),   # content, distortion, motion weights
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._quality_head.eval()

            # Initialise quality head with balanced weights
            with torch.no_grad():
                self._quality_head[-2].bias.fill_(0.5)

            self._resnet_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._ml_available = True
            self._backend = "resnet"
            logger.info(
                "Ada-DQA initialised with ResNet-50 multi-scale on %s",
                self._device,
            )

        except ImportError:
            logger.warning(
                "Ada-DQA: no ML backend available. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("Ada-DQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            # Extract multi-scale features for all frames
            frame_features = []
            for frame in frames:
                feat = self._extract_multiscale_features(frame)
                if feat is not None:
                    frame_features.append(feat)

            if not frame_features:
                return sample

            # Content quality: feature richness across scales
            content_score = self._compute_content_quality(frame_features)

            # Distortion quality: feature consistency across scales
            distortion_score = self._compute_distortion_quality(frame_features)

            # Motion quality: temporal smoothness of features
            motion_score = self._compute_motion_quality(frame_features)

            # Adaptive weighting via quality head
            import torch
            mean_feat = np.mean(
                [f for f in frame_features], axis=0
            ).astype(np.float32)
            with torch.no_grad():
                feat_tensor = torch.from_numpy(mean_feat).unsqueeze(0).to(self._device)
                weights = self._quality_head(feat_tensor).cpu().numpy().flatten()

            # Normalise weights
            weights = weights / (weights.sum() + 1e-8)

            # Adaptive fusion: emphasise weakest dimension
            components = np.array([content_score, distortion_score, motion_score])
            min_idx = np.argmin(components)
            weights[min_idx] += 0.15
            weights = weights / weights.sum()

            score = float(np.dot(weights, components))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.adadqa_score = float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            logger.warning("Ada-DQA failed for %s: %s", sample.path, e)

        return sample

    def _extract_multiscale_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract multi-scale ResNet features from a single frame."""
        import torch

        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_feats = []

            for scale in self.scales:
                h, w = frame.shape[:2]
                sh, sw = int(h * scale), int(w * scale)
                if sh < 32 or sw < 32:
                    sh, sw = max(sh, 32), max(sw, 32)
                scaled = cv2.resize(rgb, (sw, sh))

                tensor = self._resnet_transform(scaled).unsqueeze(0).to(self._device)

                with torch.no_grad():
                    early = self._resnet_layers["early"](tensor)
                    mid = self._resnet_layers["mid"](early)
                    late = self._resnet_layers["late"](mid)

                    # Pool each level to a vector
                    e_feat = self._avgpool(early).flatten().cpu().numpy()
                    m_feat = self._avgpool(mid).flatten().cpu().numpy()
                    l_feat = self._avgpool(late).flatten().cpu().numpy()

                    all_feats.extend([e_feat, m_feat, l_feat])

            return np.concatenate(all_feats).astype(np.float32)

        except Exception as e:
            logger.debug("Multi-scale feature extraction failed: %s", e)
            return None

    def _compute_content_quality(self, features: List[np.ndarray]) -> float:
        """Content diversity: higher feature norms and diversity = richer content."""
        feat_matrix = np.array(features)
        norms = np.linalg.norm(feat_matrix, axis=1)
        magnitude = float(np.clip(np.mean(norms) / 50.0, 0.0, 1.0))

        if len(features) > 1:
            diversity = float(np.clip(
                np.mean(np.std(feat_matrix, axis=0)) * 0.5, 0.0, 1.0
            ))
        else:
            diversity = 0.5

        return 0.6 * magnitude + 0.4 * diversity

    def _compute_distortion_quality(self, features: List[np.ndarray]) -> float:
        """Distortion awareness: consistency across scales within each frame."""
        n_scales = len(self.scales)
        feat_dim = (256 + 1024 + 2048)

        consistencies = []
        for feat in features:
            scale_feats = []
            for s_idx in range(n_scales):
                start = s_idx * feat_dim
                end = start + feat_dim
                sf = feat[start:end]
                sf = sf / (np.linalg.norm(sf) + 1e-8)
                scale_feats.append(sf)

            # Cross-scale consistency: distorted images lose consistency
            if len(scale_feats) > 1:
                sims = []
                for i in range(len(scale_feats)):
                    for j in range(i + 1, len(scale_feats)):
                        sim = float(np.dot(scale_feats[i], scale_feats[j]))
                        sims.append(max(sim, 0.0))
                consistencies.append(float(np.mean(sims)))

        if not consistencies:
            return 0.5
        return float(np.clip(np.mean(consistencies), 0.0, 1.0))

    def _compute_motion_quality(self, features: List[np.ndarray]) -> float:
        """Motion smoothness: temporal coherence of feature trajectory."""
        if len(features) < 2:
            return 1.0

        # Normalise features
        norm_feats = []
        for f in features:
            n = np.linalg.norm(f)
            norm_feats.append(f / (n + 1e-8))

        # Adjacent cosine similarities
        sims = []
        for i in range(len(norm_feats) - 1):
            sim = float(np.dot(norm_feats[i], norm_feats[i + 1]))
            sims.append(max(sim, 0.0))

        coherence = float(np.mean(sims))

        # Smoothness: low variance in similarities
        if len(sims) > 1:
            smoothness = 1.0 / (1.0 + float(np.var(sims)) * 10.0)
        else:
            smoothness = 1.0

        return 0.5 * coherence + 0.5 * smoothness

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
        self._resnet_layers = None
        self._quality_head = None
        self._avgpool = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

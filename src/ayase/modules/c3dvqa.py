"""C3DVQA (3D CNN Video Quality Assessment) module.

2020 — Uses 3D convolutions over spatiotemporal video volumes
to capture both spatial distortions and temporal artefacts in
a single forward pass. Operates on short video clips and pools
predictions across the full video.

This implementation uses a lightweight 3D feature extractor
with temporal convolutions for quality prediction.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class C3DVQAModule(PipelineModule):
    name = "c3dvqa"
    description = "3D CNN spatiotemporal video quality assessment"
    default_config = {
        "clip_length": 16,
        "subsample": 4,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None

    def setup(self) -> None:
        try:
            import torch
            import torchvision.models.video as video_models

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Use R3D-18 as 3D feature extractor with pretrained weights
            try:
                weights = video_models.R3D_18_Weights.DEFAULT
            except AttributeError:
                weights = "DEFAULT"
            model = video_models.r3d_18(weights=weights)
            # Remove classifier, use as feature extractor
            self._feature_extractor = torch.nn.Sequential(
                model.stem,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4,
                torch.nn.AdaptiveAvgPool3d(1),
            ).to(device)
            self._feature_extractor.eval()
            self._device = device
            self._ml_available = True
            logger.info("C3DVQA initialised (R3D-18 backbone) on %s", device)
        except (ImportError, Exception) as e:
            logger.info("C3DVQA ML backbone unavailable, using handcrafted 3D features: %s", e)
            self._ml_available = False

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            import cv2

            clip_length = self.config.get("clip_length", 16)
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Sample clips uniformly
            n_clips = self.config.get("subsample", 4)
            clip_starts = np.linspace(0, max(0, total - clip_length), n_clips, dtype=int)

            clip_scores = []
            for start in clip_starts:
                frames = []
                for j in range(clip_length):
                    idx = min(start + j, total - 1)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)

                if len(frames) < 4:
                    continue

                score = self._score_clip(frames)
                clip_scores.append(score)

            cap.release()

            if clip_scores:
                # Percentile pooling (worst clips matter more)
                sorted_scores = sorted(clip_scores)
                n = len(sorted_scores)
                p10 = sorted_scores[max(0, int(n * 0.1))]
                final = 0.3 * p10 + 0.5 * np.mean(clip_scores) + 0.2 * np.median(clip_scores)
                sample.quality_metrics.c3dvqa_score = float(np.clip(final, 0.0, 1.0))
        except Exception as e:
            logger.warning("C3DVQA failed: %s", e)
        return sample

    def _score_clip(self, frames) -> float:
        """Score a clip of frames using 3D features."""
        if self._ml_available:
            return self._score_clip_ml(frames)
        return self._score_clip_handcrafted(frames)

    def _score_clip_ml(self, frames) -> float:
        """Score using 3D CNN features."""
        import torch
        import cv2

        target_size = (112, 112)
        processed = []
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, target_size)
            tensor = torch.from_numpy(resized).float() / 255.0
            processed.append(tensor)

        # Stack: (T, H, W, C) -> (C, T, H, W)
        clip = torch.stack(processed).permute(3, 0, 1, 2).unsqueeze(0)
        clip = clip.to(self._device)

        with torch.no_grad():
            features = self._feature_extractor(clip)
            features = features.squeeze()

        # Feature statistics as quality indicator
        feat_mean = features.mean().item()
        feat_std = features.std().item()
        # Higher activation magnitude and diversity suggest richer content
        activation_score = min(1.0, abs(feat_mean) / 2.0)
        diversity_score = min(1.0, feat_std / 1.0)

        return 0.5 * activation_score + 0.5 * diversity_score

    def _score_clip_handcrafted(self, frames) -> float:
        """Score using handcrafted 3D spatiotemporal features."""
        import cv2

        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float64) for f in frames]

        # Spatial quality: average frame sharpness
        sharpness_scores = []
        for g in grays:
            lap_var = cv2.Laplacian(g, cv2.CV_64F).var()
            sharpness_scores.append(min(1.0, lap_var / 500.0))

        # Temporal quality: 3D gradient (spatiotemporal edges)
        temporal_scores = []
        for i in range(1, len(grays)):
            # Temporal derivative
            dt = grays[i] - grays[i - 1]
            # Spatial gradients at current frame
            dx = cv2.Sobel(grays[i], cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(grays[i], cv2.CV_64F, 0, 1, ksize=3)

            # 3D gradient magnitude
            grad_3d = np.sqrt(dx ** 2 + dy ** 2 + dt ** 2)
            # Coherent 3D structure suggests natural content
            coherence = np.mean(grad_3d) / (np.std(grad_3d) + 1e-8)
            temporal_scores.append(min(1.0, coherence / 2.0))

        spatial = float(np.mean(sharpness_scores))
        temporal = float(np.mean(temporal_scores)) if temporal_scores else 0.8

        return 0.5 * spatial + 0.5 * temporal

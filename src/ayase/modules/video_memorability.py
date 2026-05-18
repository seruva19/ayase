"""Video Memorability module.

Estimates content memorability using deep feature statistics.

**Important:** This is an *approximation* based on feature-space statistics
(magnitude, diversity, uniqueness), NOT a trained memorability predictor.
The normalization constants are empirically tuned and have not been validated
against ground-truth memorability datasets (e.g. VideoMem, Memento10k).
Scores should be treated as relative indicators, not calibrated probabilities.

Backend tiers:
  1. **CLIP features** — semantic richness, frame diversity, uniqueness
  2. **DINOv2 features** — similar feature-space analysis
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)


class VideoMemorabilityModule(PipelineModule):
    name = "video_memorability"
    description = "Content memorability approximation (CLIP/DINOv2 feature statistics)"
    default_config = {
        "subsample": 5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._ml_available = False
        self._feature_model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: CLIP feature extraction + memorability regression
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            device = resolve_torch_device(self.config.get("device", "auto"))
            models_dir = self.config.get("models_dir", "models")
            model_name = "openai/clip-vit-base-patch32"

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    model_name,
                    self.config,
                    device=device,
                    cache_dir=models_dir,
                ).to(device).eval()
                processor = CLIPProcessor.from_pretrained(model_name, cache_dir=models_dir)
                return model, processor

            self._feature_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    model_name,
                    device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )
            self._device = device
            self._backend = "clip"
            self._ml_available = True
            logger.info("VideoMemorability using CLIP features on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("CLIP unavailable: %s", e)

        # Tier 1b: DINOv2 feature extraction
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            self._feature_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
            self._device = device
            self._backend = "dinov2"
            self._ml_available = True
            logger.info("VideoMemorability using DINOv2 features on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.warning("VideoMemorability: no ML backend available: %s", e)

    def process(self, sample: Sample) -> Sample:
        """Process sample to predict memorability."""
        if not self._ml_available:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            import cv2

            frames = self._load_frames(sample)
            if not frames:
                return sample

            if self._backend == "clip":
                memorability = self._compute_clip_memorability(frames)
            else:
                memorability = self._compute_dinov2_memorability(frames)

            sample.quality_metrics.video_memorability = float(np.clip(memorability, 0.0, 1.0))
            logger.debug("Memorability for %s: %.3f", sample.path.name, memorability)

        except Exception as e:
            logger.warning("Memorability processing failed for %s: %s", sample.path, e)

        return sample

    def _compute_clip_memorability(self, frames: list) -> float:
        """Compute memorability using CLIP features.

        Memorability correlates with semantic distinctiveness and
        emotional valence, which CLIP captures well.
        """
        import torch
        import cv2
        from PIL import Image

        pil_frames = []
        for f in frames[:8]:  # Limit frames
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb))

        inputs = self._clip_processor(images=pil_frames, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = extract_features(self._feature_model.get_image_features(**inputs))
            features = outputs.cpu().numpy()  # [N, D]

        # Memorability indicators from CLIP space:
        # 1. Feature magnitude (semantic richness)
        feat_norms = np.linalg.norm(features, axis=1)
        richness = float(np.mean(feat_norms)) / 30.0  # Normalize

        # 2. Feature diversity across frames (narrative arc)
        if len(features) >= 2:
            pairwise_dists = []
            for i in range(len(features) - 1):
                dist = np.linalg.norm(features[i] - features[i + 1])
                pairwise_dists.append(dist)
            diversity = float(np.mean(pairwise_dists)) / 15.0
        else:
            diversity = 0.5

        # 3. Feature uniqueness (distance from mean)
        mean_feat = np.mean(features, axis=0)
        uniqueness = float(np.std(np.linalg.norm(features - mean_feat, axis=1))) / 5.0

        # Combine (empirical weights from memorability literature)
        memorability = 0.4 * min(1.0, richness) + 0.35 * min(1.0, diversity) + 0.25 * min(1.0, uniqueness)
        return float(memorability)

    def _compute_dinov2_memorability(self, frames: list) -> float:
        """Compute memorability using DINOv2 features."""
        import torch
        import cv2

        features_list = []
        for f in frames[:8]:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (224, 224))
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor = (tensor - mean) / std
            tensor = tensor.to(self._device)

            with torch.no_grad():
                features = self._feature_model(tensor)
                if isinstance(features, dict):
                    features = features.get("x_norm_clstoken", features.get("last_hidden_state", None))
                    if features is not None:
                        features = features[:, 0] if features.dim() == 3 else features
                features_list.append(features.cpu().numpy().flatten())

        if not features_list:
            return 0.5  # Cannot compute without features

        features = np.stack(features_list)

        # Similar memorability scoring
        feat_norms = np.linalg.norm(features, axis=1)
        richness = float(np.mean(feat_norms)) / 50.0

        if len(features) >= 2:
            pairwise_dists = []
            for i in range(len(features) - 1):
                dist = np.linalg.norm(features[i] - features[i + 1])
                pairwise_dists.append(dist)
            diversity = float(np.mean(pairwise_dists)) / 25.0
        else:
            diversity = 0.5

        memorability = 0.5 * min(1.0, richness) + 0.5 * min(1.0, diversity)
        return float(memorability)

    def _load_frames(self, sample: Sample) -> list:
        subsample = self.config.get("subsample", 5)
        return sample_frames(sample.path, max_frames=subsample, color="bgr")

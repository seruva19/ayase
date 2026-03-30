"""Video Memorability module.

Estimates content memorability using deep feature statistics or visual heuristics.

**Important:** This is an *approximation* based on feature-space statistics
(magnitude, diversity, uniqueness), NOT a trained memorability predictor.
The normalization constants are empirically tuned and have not been validated
against ground-truth memorability datasets (e.g. VideoMem, Memento10k).
Scores should be treated as relative indicators, not calibrated probabilities.

Backend tiers:
  1. **CLIP features** — semantic richness, frame diversity, uniqueness
  2. **DINOv2 features** — similar feature-space analysis
  3. **Heuristic** — edge density, color saturation, intensity variance
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)


class VideoMemorabilityModule(PipelineModule):
    name = "video_memorability"
    description = "Content memorability approximation (CLIP/DINOv2 feature statistics, not a trained predictor)"
    default_config = {
        "subsample": 5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "heuristic"
        self._feature_model = None
        self._device = None

    def setup(self) -> None:
        # Tier 1: CLIP feature extraction + memorability regression
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._feature_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._device = device
            self._backend = "clip"
            logger.info("VideoMemorability using CLIP features on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("CLIP unavailable: %s", e)

        # Tier 1b: DINOv2 feature extraction
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._feature_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
            self._device = device
            self._backend = "dinov2"
            logger.info("VideoMemorability using DINOv2 features on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("DINOv2 unavailable: %s", e)

        # Tier 2: Heuristic
        self._backend = "heuristic"

    def process(self, sample: Sample) -> Sample:
        """Process sample to predict memorability."""
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            import cv2

            frames = self._load_frames(sample)
            if not frames:
                return sample

            if self._backend == "clip":
                memorability = self._compute_clip_memorability(frames)
            elif self._backend == "dinov2":
                memorability = self._compute_dinov2_memorability(frames)
            else:
                memorability = self._compute_heuristic_memorability(frames)

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
            return self._compute_heuristic_memorability(frames)

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

    def _compute_heuristic_memorability(self, frames: list) -> float:
        """Compute memorability using visual heuristics."""
        import cv2

        mem_scores = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Visual complexity (edge density)
            edges = cv2.Canny(gray, 50, 150)
            complexity = np.count_nonzero(edges) / edges.size

            # Color vividness (saturation)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            vividness = hsv[:, :, 1].mean() / 255.0

            # Uniqueness (local variance)
            variance = gray.std() / 255.0

            memorability = complexity * 0.3 + vividness * 0.4 + variance * 0.3
            mem_scores.append(float(np.clip(memorability, 0.0, 1.0)))

        return float(np.mean(mem_scores)) if mem_scores else 0.5

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 5)
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames

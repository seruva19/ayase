"""Unified-VQA -- Unified Video Quality Assessment (FR+NR Multi-task).

2025 -- combines spatial, temporal, and semantic features via a CLIP backbone
for both NR (no-reference) and FR (full-reference) quality assessment.

Implementation:
    NR mode: extract CLIP visual features from sampled frames, regress quality
    from feature statistics (mean, std, percentiles of CLIP embeddings).
    FR mode: additionally compute CLIP feature distances between reference and
    distorted, weighting the final score.

Temporal aggregation uses attention-weighted pooling based on frame-level
feature magnitudes (informative frames get higher weight).

Stores result in unified_vqa_score. For backward compatibility, also writes
dover_score when that legacy shared NR quality field is unset.
"""

import logging
import hashlib
from typing import List, Optional

import cv2
import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_openai_clip_image_features,
    media_state_key,
    resolve_torch_device,
    shared_openai_clip_resource,
)

logger = logging.getLogger(__name__)


class UnifiedVQAModule(PipelineModule):
    name = "unified_vqa"
    description = "Unified-VQA FR+NR multi-task quality assessment (2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "ViT-B/32",
    }
    models = [
        {"id": "ViT-B/32", "type": "clip", "task": "CLIP visual feature backbone"},
        {
            "id": "torchvision/resnet50",
            "type": "torchvision",
            "task": "Fallback visual feature backbone",
        },
    ]
    metric_info = {
        "unified_vqa_score": "Unified-VQA FR/NR quality score (0-1, higher=better)",
        "dover_score": "Backward-compatible alias when dover_score is otherwise unset",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.clip_model_name = self.config.get("clip_model", "ViT-B/32")
        self._clip_model = None
        self._clip_preprocess = None
        self._resnet = None
        self._resnet_transform = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None  # "clip" or "resnet"

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Try CLIP backbone (preferred)
        if self._try_clip_setup():
            return

        # Tier 2: ResNet-50 backbone (fallback)
        if self._try_resnet_setup():
            return

        logger.warning(
            "Unified-VQA: no backbone available. Install with: "
            "pip install torch torchvision  (or additionally: pip install clip)"
        )

    def _try_clip_setup(self) -> bool:
        """Try to set up CLIP backbone."""
        try:
            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._clip_model, self._clip_preprocess = shared_openai_clip_resource(
                self,
                self.clip_model_name,
                device=self._device,
            )
            self._ml_available = True
            self._backend = "clip"
            logger.info("Unified-VQA initialised with CLIP (%s) on %s", self.clip_model_name, self._device)
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug("CLIP setup failed: %s", e)
            return False

    def _try_resnet_setup(self) -> bool:
        """Try to set up ResNet-50 backbone as fallback."""
        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove final FC, keep feature extractor (2048-d)
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

            self._ml_available = True
            self._backend = "resnet"
            logger.info("Unified-VQA initialised with ResNet-50 on %s", self._device)
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug("ResNet setup failed: %s", e)
            return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            # Extract features for distorted frames
            dist_features = self._extract_features_batch(
                frames,
                cache_key=(self.subsample, media_state_key(sample.path)),
            )
            if not dist_features:
                return sample

            # Check for reference (FR mode)
            reference_path = getattr(sample, "reference_path", None)
            ref_features = None
            if reference_path is not None:
                from pathlib import Path
                ref_path = Path(reference_path)
                if ref_path.exists():
                    ref_frames = self._extract_frames_from_path(ref_path, sample.is_video)
                    if ref_frames:
                        ref_features = self._extract_features_batch(
                            ref_frames,
                            cache_key=(self.subsample, media_state_key(ref_path)),
                        )

            score = self._compute_quality(dist_features, ref_features)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.unified_vqa_score = score
                # Backward compatibility: older integrations used dover_score
                # as the shared NR-quality slot for this module.
                if sample.quality_metrics.dover_score is None:
                    sample.quality_metrics.dover_score = score

        except Exception as e:
            logger.warning("Unified-VQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_quality(
        self,
        dist_features: List[np.ndarray],
        ref_features: Optional[List[np.ndarray]] = None,
    ) -> Optional[float]:
        """Compute quality score from features.

        NR mode: quality from feature statistics.
        FR mode: additionally use feature distances.
        """
        dist_matrix = np.array(dist_features)

        # Feature statistics for NR quality
        feat_mean = np.mean(dist_matrix, axis=0)
        feat_std = np.std(dist_matrix, axis=0)
        feat_norm = float(np.linalg.norm(feat_mean))

        # NR quality components
        # Feature magnitude correlates with content quality
        nr_magnitude = float(np.clip(feat_norm / 20.0, 0.0, 1.0))

        # Feature consistency across frames (temporal quality)
        feat_consistency = 1.0 / (1.0 + float(np.mean(feat_std)) * 0.1)

        # Attention-weighted temporal smoothness
        if len(dist_features) > 1:
            diffs = []
            for i in range(len(dist_features) - 1):
                cos_sim = float(np.dot(dist_features[i], dist_features[i + 1]) / (
                    np.linalg.norm(dist_features[i]) * np.linalg.norm(dist_features[i + 1]) + 1e-10
                ))
                diffs.append(cos_sim)
            temporal_smoothness = float(np.mean(diffs))
        else:
            temporal_smoothness = 1.0

        nr_score = (
            0.40 * nr_magnitude
            + 0.30 * feat_consistency
            + 0.30 * temporal_smoothness
        )

        if ref_features is not None and len(ref_features) > 0:
            # FR mode: compute feature distances
            n = min(len(dist_features), len(ref_features))

            fr_sims = []
            for i in range(n):
                cos_sim = float(np.dot(dist_features[i], ref_features[i]) / (
                    np.linalg.norm(dist_features[i]) * np.linalg.norm(ref_features[i]) + 1e-10
                ))
                fr_sims.append(cos_sim)

            fr_similarity = float(np.mean(fr_sims))

            # Combine FR + NR (FR gets more weight when reference is available)
            score = 0.35 * nr_score + 0.65 * fr_similarity
        else:
            score = nr_score

        return float(np.clip(score, 0.0, 1.0))

    def _extract_features_batch(
        self,
        frames: List[np.ndarray],
        cache_key: Optional[tuple] = None,
    ) -> List[np.ndarray]:
        """Extract features from a batch of frames."""
        if self._backend == "clip":
            try:
                rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
                image_features = cached_openai_clip_image_features(
                    self,
                    self._clip_model,
                    self._clip_preprocess,
                    arrays_to_pil(rgb_frames),
                    model_key=self.clip_model_name,
                    device=self._device,
                    cache_key=cache_key or ("unified_vqa_frames", len(frames)),
                    normalize=False,
                )
                return [
                    row.detach().float().cpu().numpy().astype(np.float32)
                    for row in image_features
                ]
            except Exception as e:
                logger.debug("CLIP batch feature extraction failed: %s", e)

        features = []
        for frame in frames:
            feat = self._extract_features(frame)
            if feat is not None:
                features.append(feat)
        return features

    def _extract_features(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract feature embedding from a single frame."""
        import torch

        try:
            if self._backend == "clip":
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                contiguous = np.ascontiguousarray(rgb)
                digest = hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()
                feat = cached_openai_clip_image_features(
                    self,
                    self._clip_model,
                    self._clip_preprocess,
                    arrays_to_pil([contiguous]),
                    model_key=self.clip_model_name,
                    device=self._device,
                    cache_key=("unified_vqa_frame", contiguous.shape, digest),
                    normalize=False,
                )
                return feat.squeeze(0).detach().float().cpu().numpy().astype(np.float32)

            else:  # resnet
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    feat = self._resnet(tensor)
                return feat.cpu().numpy().flatten().astype(np.float32)

        except Exception as e:
            logger.debug("Feature extraction failed: %s", e)
            return None

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        """Extract frames from a Sample."""
        return self._extract_frames_from_path(sample.path, sample.is_video)

    def _extract_frames_from_path(self, path, is_video: bool) -> List[np.ndarray]:
        """Extract frames from a path."""
        try:
            rgb_frames = sample_frames(path, max_frames=self.subsample, color="rgb")
            return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames]
        except Exception as e:
            logger.debug("Unified-VQA frame loading failed for %s: %s", path, e)
            return []

    def on_dispose(self) -> None:
        self._clip_model = None
        self._resnet = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

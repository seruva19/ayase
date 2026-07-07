"""PreResQ-R1 -- Fine-Grained Rank-and-Score VQA (2025).

This module implements a CLIP zero-shot rank-and-score quality signal in the
CLIP-IQA family: video frames are compared against an ordered set of
quality-level text prompts (worst -> best) and the softmax over prompt
similarities yields a rank-aware continuous score. It uses a real, pretrained
CLIP backbone (config-selectable via ``clip_model``); it does NOT load the
trained PreResQ-R1 weights, so treat ``presresq_score`` as a CLIP-IQA-style
proxy rather than the published model's output. When CLIP is unavailable the
module emits no score.

presresq_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_openai_clip_image_features,
    cached_openai_clip_text_features,
    media_state_key,
    resolve_torch_device,
    shared_openai_clip_resource,
)

logger = logging.getLogger(__name__)

# Quality ranking prompts: ordered from worst to best
_RANK_PROMPTS = [
    "a very poor quality video frame with severe distortion and blur",
    "a poor quality video frame with noticeable artifacts",
    "a below average quality video frame with some imperfections",
    "an average quality video frame",
    "an above average quality video frame with good clarity",
    "a good quality video frame with sharp details",
    "a very good quality video frame with excellent clarity",
    "an outstanding quality, pristine, professional video frame",
]


class PreResQModule(PipelineModule):
    name = "presresq"
    description = "PreResQ-R1 rank+score VQA (2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "ViT-B/32",
    }
    metric_groups = {
        "presresq_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.clip_model_name = self.config.get("clip_model", "ViT-B/32")
        self._clip_model = None
        self._clip_preprocess = None
        self._text_features = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.test_mode:
            return

        if self._try_clip_setup():
            return

        self._backend = "unavailable"
        logger.warning(
            "PreResQ unavailable: CLIP backend not available (pip install clip)"
        )

    def _try_clip_setup(self) -> bool:
        try:
            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._clip_model, self._clip_preprocess = shared_openai_clip_resource(
                self,
                self.clip_model_name,
                device=self._device,
            )

            # Pre-encode ranking prompts
            self._text_features = cached_openai_clip_text_features(
                self,
                self._clip_model,
                _RANK_PROMPTS,
                model_key=self.clip_model_name,
                device=self._device,
                cache_key=("presresq_rank_prompts",),
            )

            self._ml_available = True
            self._backend = "clip"
            logger.info(
                "PreResQ initialised with CLIP (%s) on %s",
                self.clip_model_name, self._device,
            )
            return True

        except ImportError:
            return False
        except Exception as e:
            logger.debug("CLIP setup failed: %s", e)
            return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            score = self._compute_clip_rank_score(sample, frames)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.presresq_score = float(
                    np.clip(score, 0.0, 1.0)
                )

        except Exception as e:
            logger.warning("PreResQ failed for %s: %s", sample.path, e)

        return sample

    def _compute_clip_rank_score(self, sample: Sample, frames: List[np.ndarray]) -> Optional[float]:
        """Rank-and-score via CLIP: compare frames against quality prompts."""
        n_levels = len(_RANK_PROMPTS)
        # Quality levels evenly spaced from 0 to 1
        quality_levels = np.linspace(0.0, 1.0, n_levels)

        image_features = cached_openai_clip_image_features(
            self,
            self._clip_model,
            self._clip_preprocess,
            arrays_to_pil([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]),
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        if image_features is None or image_features.size(0) == 0:
            return None

        sims = (image_features @ self._text_features.T).detach().float().cpu().numpy()
        exp_sims = np.exp(sims - np.max(sims, axis=1, keepdims=True))
        probs = exp_sims / np.sum(exp_sims, axis=1, keepdims=True)
        frame_scores = np.dot(probs, quality_levels).astype(np.float32)

        if len(frame_scores) == 0:
            return None

        # Temporal aggregation: mean with consistency bonus
        mean_score = float(np.mean(frame_scores))

        # Consistency: low variance across frames = more reliable
        if len(frame_scores) > 1:
            consistency = 1.0 / (1.0 + float(np.var(frame_scores)) * 10.0)
            # Blend: slightly reward consistent quality
            score = 0.85 * mean_score + 0.15 * consistency
        else:
            score = mean_score

        return score

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            rgb_frames = sample_frames(sample.path, max_frames=self.subsample, color="rgb")
            return [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames]
        except Exception as e:
            logger.debug("PreResQ frame loading failed for %s: %s", sample.path, e)
            return []

    def on_dispose(self) -> None:
        self._clip_model = None
        self._text_features = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

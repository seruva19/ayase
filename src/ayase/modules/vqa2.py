"""VQA^2 --- Visual Question Answering for Video Quality Assessment (MM 2025).

GitHub: https://github.com/Q-Future/Visual-Question-Answering-for-Video-Quality-Assessment

Uses a vision-language model to score video quality via prompted VQA.
Backend priority:
  1. Q-Align via pyiqa (``q-align`` metric)
  2. CLIP prompt-based quality scoring as lightweight alternative

vqa2_score --- higher = better (0-1 range)
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    media_state_key,
)

logger = logging.getLogger(__name__)

# Quality assessment prompts for CLIP-based scoring
_QUALITY_LEVELS = [
    ("excellent quality", 1.0),
    ("good quality", 0.75),
    ("fair quality", 0.50),
    ("poor quality", 0.25),
    ("bad quality", 0.0),
]


class VQA2Module(PipelineModule):
    name = "vqa2"
    description = "VQA^2 LMM video quality assessment (MM 2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._ml_available = False
        self._backend = None
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._quality_embeds = None
        self._quality_weights = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # Try pyiqa q-align first (closest to VQA^2 LMM approach)
        if self._try_qalign_setup():
            return

        # Fallback: CLIP prompt-based quality scoring
        self._try_clip_setup()

    def _try_qalign_setup(self) -> bool:
        """Try Q-Align via pyiqa as VQA-based quality scorer."""
        try:
            import pyiqa
            import torch

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = pyiqa.create_metric(
                "qalign", device=str(self._device)
            )
            self._ml_available = True
            self._backend = "qalign"
            logger.info("VQA^2 (Q-Align backend) initialised on %s", self._device)
            return True
        except (ImportError, Exception) as e:
            logger.debug("Q-Align not available: %s", e)
            return False

    def _try_clip_setup(self) -> bool:
        """CLIP prompt-based quality scoring."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self._clip_model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._model, self._processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )

            # Pre-encode quality level prompts
            texts = [q[0] for q in _QUALITY_LEVELS]
            self._quality_weights = np.array([q[1] for q in _QUALITY_LEVELS])

            with torch.no_grad():
                text_inputs = self._processor(
                    text=texts, return_tensors="pt", padding=True
                ).to(self._device)
                self._quality_embeds = self._model.get_text_features(**text_inputs)
                self._quality_embeds = self._quality_embeds / self._quality_embeds.norm(
                    dim=-1, keepdim=True
                )

            self._ml_available = True
            self._backend = "clip"
            logger.info("VQA^2 (CLIP backend) initialised on %s", self._device)
            return True
        except (ImportError, Exception) as e:
            logger.warning("VQA^2 setup failed: %s", e)
            return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            if self._backend == "qalign":
                score = self._process_qalign(sample)
            else:
                score = self._process_clip(sample)

            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.vqa2_score = score
            logger.debug("VQA^2 for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("VQA^2 failed for %s: %s", sample.path, e)
        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        if self._backend != "clip":
            return super().process_batch(samples)

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            for sample in samples:
                frames = self._extract_frames(sample)
                if not frames:
                    continue
                prepared.append(sample)
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self.subsample, media_state_key(sample.path)))

            if not prepared:
                return samples

            feature_groups = cached_clip_image_feature_groups(
                self,
                self._model,
                self._processor,
                image_groups,
                model_key=self._clip_model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for sample, image_features in zip(prepared, feature_groups):
                score = self._score_clip_features(image_features)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.vqa2_score = score
        except Exception as e:
            logger.warning("VQA^2 batch failed: %s", e)

        return samples

    def _process_qalign(self, sample: Sample) -> Optional[float]:
        """Score using Q-Align (LMM-based)."""
        import torch

        try:
            with torch.no_grad():
                if sample.is_video:
                    result = self._model(str(sample.path))
                else:
                    result = self._model(str(sample.path))
                score = float(result.item()) if hasattr(result, "item") else float(result)
                # Q-Align outputs 1-5 scale; normalise to 0-1
                return float(np.clip((score - 1.0) / 4.0, 0.0, 1.0))
        except Exception as e:
            logger.debug("Q-Align scoring failed: %s", e)
            return None

    def _process_clip(self, sample: Sample) -> Optional[float]:
        """CLIP prompt-based VQA: softmax over quality-level prompts."""

        frames = self._extract_frames(sample)
        if not frames:
            return None

        pil_frames = arrays_to_pil(frames)
        image_features = cached_clip_image_features(
            self,
            self._model,
            self._processor,
            pil_frames,
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        return self._score_clip_features(image_features)

    def _score_clip_features(self, image_features) -> Optional[float]:
        if image_features is None or image_features.size(0) == 0:
            return None

        frame_scores = []
        for img_emb in image_features:
            # Cosine similarity with each quality level
            sims = (img_emb @ self._quality_embeds.T).cpu().numpy()
            # Softmax to get probability distribution
            exp_sims = np.exp((sims - sims.max()) * 100.0)  # temperature scaling
            probs = exp_sims / exp_sims.sum()
            # Weighted score
            score = float(np.dot(probs, self._quality_weights))
            frame_scores.append(score)

        if not frame_scores:
            return None
        return float(np.clip(np.mean(frame_scores), 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug("VQA^2 frame loading failed for %s: %s", sample.path, e)
            return []

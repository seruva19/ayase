"""VQAScore text-visual alignment module.

Uses VQA model to score "Does this figure show {text}?" probability.
ECCV 2024, outperforms CLIPScore on compositional text prompts.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_openai_clip_image_feature_groups,
    cached_openai_clip_image_features,
    cached_openai_clip_text_features,
    media_state_key,
    resolve_torch_device,
    shared_openai_clip_resource,
)

logger = logging.getLogger(__name__)


class VQAScoreModule(PipelineModule):
    name = "vqa_score"
    description = "VQAScore text-visual alignment via VQA probability (0-1, higher=better)"
    default_config = {
        "model": "clip-flant5-xxl",
        "clip_model": "ViT-B/32",
        "subsample": 4,
    }
    models = [
        {
            "id": "clip-flant5-xxl",
            "type": "other",
            "task": "Vendored t2v_metrics VQAScore model",
            "install": "bundled in ayase.vendor.t2v_metrics",
        },
        {
            "id": "ViT-B/32",
            "type": "clip",
            "task": "OpenAI CLIP fallback for text-visual alignment",
        },
    ]
    metric_info = {
        "vqa_score_alignment": "VQAScore text-visual alignment probability (0-1, higher=better)",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._clip_model = None
        self._clip_preprocess = None
        self._clip_model_name = self.config.get("clip_model", "ViT-B/32")
        self._device = "cpu"

    def setup(self) -> None:
        try:
            from ayase.vendor.t2v_metrics import VQAScore as VQAScoreMetric

            model_name = self.config.get("model", "clip-flant5-xxl")
            self._model = VQAScoreMetric(model=model_name)
            self._ml_available = True
            logger.info("VQAScore model loaded: %s", model_name)
        except ImportError:
            # Fallback: use CLIP similarity as approximation
            try:
                self._device = resolve_torch_device(self.config.get("device", "auto"))
                self._clip_model, self._clip_preprocess = shared_openai_clip_resource(
                    self,
                    self._clip_model_name,
                    device=self._device,
                )
                self._ml_available = True
                self._model = None  # signals CLIP fallback
                logger.info("VQAScore fallback: using CLIP on %s", self._device)
            except (ImportError, Exception) as e:
                logger.warning("VQAScore unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        if sample.caption is None or not sample.caption.text:
            return sample

        try:
            if self._model is not None:
                score = self._compute_vqascore(sample)
            else:
                score = self._compute_clip_fallback(sample)

            if score is not None:
                sample.quality_metrics.vqa_score_alignment = float(score)
        except Exception as e:
            logger.warning("VQAScore processing failed: %s", e)
        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        if self._model is not None:
            return super().process_batch(samples)

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            subsample = self.config.get("subsample", 4)
            for sample in samples:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                if sample.caption is None or not sample.caption.text:
                    continue
                frames = self._load_clip_frames(sample, subsample)
                if not frames:
                    continue
                prepared.append((sample, sample.caption.text))
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((subsample, media_state_key(sample.path)))

            if not prepared:
                return samples

            feature_groups = cached_openai_clip_image_feature_groups(
                self,
                self._clip_model,
                self._clip_preprocess,
                image_groups,
                model_key=self._clip_model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for (sample, text), image_features in zip(prepared, feature_groups):
                score = self._score_clip_features(image_features, text)
                if score is not None:
                    sample.quality_metrics.vqa_score_alignment = float(score)
        except Exception as e:
            logger.warning("VQAScore batch failed: %s", e)
        return samples

    def _compute_vqascore(self, sample: Sample) -> Optional[float]:
        """Compute VQAScore using t2v_metrics."""
        from PIL import Image

        text = sample.caption.text
        subsample = self.config.get("subsample", 4)

        if sample.is_video:
            import cv2

            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]

            scores = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    score = self._model(images=[pil_img], texts=[text]).item()
                    scores.append(score)
            cap.release()
            return float(np.mean(scores)) if scores else None
        else:
            img = Image.open(str(sample.path)).convert("RGB")
            return self._model(images=[img], texts=[text]).item()

    def _compute_clip_fallback(self, sample: Sample) -> Optional[float]:
        """Fallback: compute CLIP similarity."""
        text = sample.caption.text
        subsample = self.config.get("subsample", 4)

        frames = self._load_clip_frames(sample, subsample)
        if not frames:
            return None

        image_features = cached_openai_clip_image_features(
            self,
            self._clip_model,
            self._clip_preprocess,
            arrays_to_pil(frames),
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=(subsample, media_state_key(sample.path)),
        )
        return self._score_clip_features(image_features, text)

    def _score_clip_features(self, image_features, text: str) -> Optional[float]:
        if image_features is None or image_features.size(0) == 0:
            return None
        text_features = cached_openai_clip_text_features(
            self,
            self._clip_model,
            [text],
            model_key=self._clip_model_name,
            device=self._device,
            cache_key=("vqa_score_caption", text),
        )
        sims = (image_features @ text_features.T).squeeze(-1).detach().float().cpu().numpy()
        scores = np.maximum(sims, 0.0)
        return float(np.mean(scores)) if len(scores) else None

    @staticmethod
    def _load_clip_frames(sample: Sample, subsample: int) -> List[np.ndarray]:
        return sample_frames(sample.path, max_frames=subsample, color="rgb")

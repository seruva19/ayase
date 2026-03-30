"""PickScore module.

This module loads PickScore through the Hugging Face transformers backend.
Images are scored directly; videos are scored by uniformly sampling frames
and averaging frame scores.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)


class PickScoreModule(PipelineModule):
    name = "pickscore"
    description = "PickScore prompt-conditioned human preference scoring (frame-averaged on video)"
    default_config = {
        "model_name": "yuvalkirstain/PickScore_v1",
        "processor_name": "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        "num_frames": 5,
        "device": "auto",
        "warning_threshold": None,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "yuvalkirstain/PickScore_v1")
        self.processor_name = self.config.get(
            "processor_name", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        )
        self.num_frames = self.config.get("num_frames", 5)
        self.device_config = self.config.get("device", "auto")
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = None
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            from ayase.config import resolve_model_path

            self._device = self._resolve_device(torch)
            models_dir = self.config.get("models_dir", "models")
            model_path = resolve_model_path(self.model_name, models_dir)
            processor_path = resolve_model_path(self.processor_name, models_dir)

            self._processor = AutoProcessor.from_pretrained(processor_path)
            self._model = AutoModel.from_pretrained(model_path).eval().to(self._device)
            self._backend = "pickscore"
            logger.info("PickScore model initialized on %s", self._device)
        except ImportError:
            logger.warning("PickScore unavailable: missing dependency.")
        except Exception as e:
            logger.warning("Failed to load PickScore model: %s", e)

    def process(self, sample: Sample) -> Sample:
        if self._backend != "pickscore":
            return sample

        caption_text = self._get_caption_text(sample)
        if not caption_text:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            score = self._score_frames(frames, caption_text)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.pickscore_score = score

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low PickScore score: {score:.3f}",
                        details={"pickscore_score": score},
                    )
                )
        except Exception as e:
            logger.warning("PickScore inference failed: %s", e)

        return sample

    def _resolve_device(self, torch_module) -> str:
        if self.device_config == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return str(self.device_config)

    def _get_caption_text(self, sample: Sample) -> Optional[str]:
        if sample.caption:
            return sample.caption.text
        txt_path = sample.path.with_suffix(".txt")
        if not txt_path.exists():
            return None
        try:
            return txt_path.read_text(encoding="utf-8").strip()
        except Exception:
            logger.debug("Failed to read caption file: %s", txt_path)
            return None

    def _score_frames(self, frames: List[Image.Image], caption: str) -> Optional[float]:
        try:
            import torch

            text_inputs = self._processor(
                text=caption,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self._device)
            image_inputs = self._processor(
                images=frames,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                text_features = extract_features(self._model.get_text_features(**text_inputs))
                image_features = extract_features(self._model.get_image_features(**image_inputs))

            text_features = self._unwrap_features(text_features)
            image_features = self._unwrap_features(image_features)
            if text_features is None or image_features is None:
                return None

            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            scale = self._model.logit_scale.exp() if hasattr(self._model, "logit_scale") else 1.0
            scores = scale * (text_features @ image_features.T)[0]
            return float(scores.mean().item())
        except Exception as e:
            logger.debug("PickScore scoring failed: %s", e)
            return None

    def _unwrap_features(self, output):
        if hasattr(output, "shape"):
            return output
        for attr in ("text_embeds", "image_embeds", "pooler_output", "last_hidden_state"):
            value = getattr(output, attr, None)
            if value is None:
                continue
            if attr == "last_hidden_state" and hasattr(value, "__getitem__"):
                return value[:, 0]
            return value
        return None

    def _load_frames(self, sample: Sample) -> List[Image.Image]:
        try:
            if not sample.is_video:
                bgr = cv2.imread(str(sample.path))
                if bgr is None:
                    return []
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return [Image.fromarray(rgb)]

            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []

            frame_count = min(self.num_frames, total)
            indices = np.linspace(0, total - 1, frame_count, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
            cap.release()
            return frames
        except Exception:
            return []

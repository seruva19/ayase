import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SemanticAlignmentModule(PipelineModule):
    name = "semantic_alignment"
    description = "Checks alignment between video and caption (CLIP Score)"

    default_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "max_frames": 32,
        "warning_threshold": 0.2,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 32)
        self.warning_threshold = self.config.get("warning_threshold", 0.2)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def on_mount(self) -> None:
        super().on_mount()
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CLIP for Alignment on {self._device}...")

            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
            resolved = resolve_model_path(model_name, models_dir)

            self._model = CLIPModel.from_pretrained(resolved, use_safetensors=True).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(resolved)
            self._ml_available = True

        except ImportError:
            logger.warning("Transformers/Torch not installed. CLIP checks disabled.")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        caption_text = None
        if sample.caption:
            caption_text = sample.caption.text
        else:
            txt_path = sample.path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    caption_text = txt_path.read_text().strip()
                except Exception:
                    logger.debug(f"Failed to read caption file: {txt_path}")

        if not caption_text:
            return sample

        try:
            import torch

            frames = self._load_frames(sample)
            if not frames:
                return sample

            # Extract text features once
            text_inputs = self._processor(
                text=[caption_text],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._device)

            with torch.no_grad():
                text_features = self._model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            # Compute cosine similarity for each frame, then average
            similarities = []
            for pil_image in frames:
                image_inputs = self._processor(
                    images=pil_image,
                    return_tensors="pt",
                ).to(self._device)

                with torch.no_grad():
                    image_features = self._model.get_image_features(**image_inputs)
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    sim = (image_features @ text_features.T).item()
                    similarities.append(sim)

            score = float(np.mean(similarities))

            if sample.quality_metrics is None:
                from ayase.models import QualityMetrics
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.clip_score = score

            if score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low semantic alignment: {score:.3f}",
                        details={"clip_score": score, "caption": caption_text[:50] + "..."},
                    )
                )

        except Exception as e:
            logger.warning(f"Semantic alignment check failed: {e}")

        return sample

    def _load_frames(self, sample: Sample) -> List[Image.Image]:
        """Load frames from video (uniformly sampled) or single image."""
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

            n = min(self.max_frames, total)
            indices = np.linspace(0, total - 1, n, dtype=int)

            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
            cap.release()
            return frames
        except Exception:
            return []

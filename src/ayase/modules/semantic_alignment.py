"""Text-video semantic alignment scoring using CLIP cosine similarity.

Averages CLIP image-text similarity across uniformly sampled frames.
Returns clip_score (typically 0-0.4, higher = better alignment). Warns below 0.2."""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features

logger = logging.getLogger(__name__)


class SemanticAlignmentModule(PipelineModule):
    name = "semantic_alignment"
    description = "Checks alignment between video and caption (CLIP Score)"

    default_config = {
        "model_name": "openai/clip-vit-base-patch32",
        "backend": "auto",  # auto | transformers | open_clip
        "pretrained": "laion2b_s34b_b79k",  # open_clip checkpoint tag
        "max_frames": 32,
        "warning_threshold": 0.2,
    }
    models = [
        {
            "id": "openai/clip-vit-base-patch32",
            "type": "huggingface",
            "task": "Default CLIP text-image alignment backend",
        },
        {
            "id": "open_clip/ViT-B-32",
            "type": "clip",
            "install": "pip install open-clip-torch",
            "task": "Legacy OpenCLIP backbone option for image-to-image adapters",
        },
    ]
    metric_info = {
        "clip_score": "CLIP/OpenCLIP text-image cosine similarity (higher=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "openai/clip-vit-base-patch32")
        self.backend = self.config.get("backend", "auto")
        self.pretrained = self.config.get("pretrained", "laion2b_s34b_b79k")
        self.max_frames = self.config.get("max_frames", 32)
        self.warning_threshold = self.config.get("warning_threshold", 0.2)
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._preprocess = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.backend == "open_clip" or (
            self.backend == "auto" and str(self.model_name).startswith("open_clip:")
        ):
            if self._setup_open_clip():
                return
            if self.backend == "open_clip":
                return

        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading CLIP for Alignment on {self._device}...")

            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)

            self._model = CLIPModel.from_pretrained(resolved, use_safetensors=True).to(self._device)
            self._processor = CLIPProcessor.from_pretrained(resolved)
            self._ml_available = True
            self._backend = "transformers"

        except ImportError:
            logger.warning("Transformers/Torch not installed. CLIP checks disabled.")
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")

    def _setup_open_clip(self) -> bool:
        try:
            import torch
            import open_clip

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = str(self.model_name)
            if model_name.startswith("open_clip:"):
                model_name = model_name.split(":", 1)[1]
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=self.pretrained,
                device=self._device,
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            self._model.eval()
            self._ml_available = True
            self._backend = "open_clip"
            logger.info(
                "Loading OpenCLIP for Alignment on %s: %s/%s",
                self._device,
                model_name,
                self.pretrained,
            )
            return True
        except ImportError:
            logger.warning("open_clip_torch not installed. OpenCLIP alignment disabled.")
            return False
        except Exception as e:
            logger.warning("Failed to load OpenCLIP: %s", e)
            return False

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

            text_features = self._encode_text(caption_text)
            if text_features is None:
                return sample

            # Compute cosine similarity for each frame, then average
            similarities = []
            for pil_image in frames:
                image_features = self._encode_image(pil_image)
                if image_features is not None:
                    sim = (image_features @ text_features.T).item()
                    similarities.append(sim)

            if not similarities:
                return sample

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

    def _encode_text(self, text: str):
        import torch

        with torch.no_grad():
            if self._backend == "open_clip":
                tokens = self._tokenizer([text]).to(self._device)
                features = self._model.encode_text(tokens)
            else:
                text_inputs = self._processor(
                    text=[text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self._device)
                features = extract_features(self._model.get_text_features(**text_inputs))
            return features / features.norm(p=2, dim=-1, keepdim=True)

    def _encode_image(self, image: Image.Image):
        import torch

        with torch.no_grad():
            if self._backend == "open_clip":
                tensor = self._preprocess(image).unsqueeze(0).to(self._device)
                features = self._model.encode_image(tensor)
            else:
                image_inputs = self._processor(
                    images=image,
                    return_tensors="pt",
                ).to(self._device)
                features = extract_features(self._model.get_image_features(**image_inputs))
            return features / features.norm(p=2, dim=-1, keepdim=True)

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

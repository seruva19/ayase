"""Text-video semantic alignment scoring using CLIP cosine similarity.

Averages CLIP image-text similarity across uniformly sampled frames.
Returns clip_score (typically 0-0.4, higher = better alignment). Warns below 0.2."""

import logging
from PIL import Image
from typing import List

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule
from ayase.compat import extract_features
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    cached_clip_text_features,
    media_state_key,
    shared_runtime_value,
)

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
    metric_groups = {
        "clip_score": "alignment",
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
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            logger.info(f"Loading CLIP for Alignment on {self._device}...")

            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                    use_safetensors=True,
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
                    "safetensors",
                ),
                load_clip,
            )
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
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
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

    def _get_caption_text(self, sample: Sample) -> str:
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
        return caption_text or ""

    def _apply_score(self, sample: Sample, caption_text: str, score: float) -> None:
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

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        caption_text = self._get_caption_text(sample)

        if not caption_text:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            text_features = self._encode_text(caption_text)
            if text_features is None:
                return sample

            image_features = self._encode_images(sample, frames)
            if image_features is None or image_features.size(0) == 0:
                return sample

            similarities = image_features @ text_features.T
            score = float(similarities.mean().item())

            self._apply_score(sample, caption_text, score)

        except Exception as e:
            logger.warning(f"Semantic alignment check failed: {e}")

        return sample

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        if self._backend != "transformers":
            return super().process_batch(samples)

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            captions = []

            for sample in samples:
                caption_text = self._get_caption_text(sample)
                if not caption_text:
                    continue

                frames = self._load_frames(sample)
                if not frames:
                    continue

                prepared.append((sample, caption_text))
                image_groups.append(frames)
                cache_keys.append((self.max_frames, media_state_key(sample.path)))
                captions.append(caption_text)

            if not prepared:
                return samples

            text_features = cached_clip_text_features(
                self,
                self._model,
                self._processor,
                captions,
                model_key=self.model_name,
                device=self._device,
            )
            image_feature_groups = cached_clip_image_feature_groups(
                self,
                self._model,
                self._processor,
                image_groups,
                model_key=self.model_name,
                device=self._device,
                cache_keys=cache_keys,
            )

            for idx, ((sample, caption_text), image_features) in enumerate(
                zip(prepared, image_feature_groups)
            ):
                if image_features is None or image_features.size(0) == 0:
                    continue
                similarities = image_features @ text_features[idx:idx + 1].T
                score = float(similarities.mean().item())
                self._apply_score(sample, caption_text, score)
        except Exception as e:
            logger.warning("Semantic alignment batch check failed: %s", e)

        return samples

    def _encode_text(self, text: str):
        import torch

        def compute():
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

        if self._backend == "transformers":
            return cached_clip_text_features(
                self,
                self._model,
                self._processor,
                [text],
                model_key=self.model_name,
                device=self._device,
            )

        return shared_runtime_value(
            self,
            ("clip_text_features", self._backend, self.model_name, self._device, text),
            compute,
        )

    def _encode_images(self, sample: Sample, images: List[Image.Image]):
        import torch

        key = (
            "clip_image_features",
            self._backend,
            self.model_name,
            self._device,
            self.max_frames,
            media_state_key(sample.path),
        )

        if self._backend == "transformers":
            return cached_clip_image_features(
                self,
                self._model,
                self._processor,
                images,
                model_key=self.model_name,
                device=self._device,
                cache_key=(self.max_frames, media_state_key(sample.path)),
            )

        def compute():
            with torch.no_grad():
                tensors = [self._preprocess(image) for image in images]
                tensor = torch.stack(tensors, dim=0).to(self._device)
                features = self._model.encode_image(tensor)
                return features / features.norm(p=2, dim=-1, keepdim=True)

        return shared_runtime_value(self, key, compute)

    def _load_frames(self, sample: Sample) -> List[Image.Image]:
        """Load frames from video (uniformly sampled) or single image."""
        try:
            return arrays_to_pil(sample_frames(sample.path, max_frames=self.max_frames, color="rgb"))
        except Exception:
            return []

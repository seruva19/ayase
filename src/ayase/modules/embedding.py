"""X-CLIP video embedding computation for similarity search and downstream analysis.

Extracts normalized video-level features via X-CLIP vision model with MIT
(temporal attention). Stores L2-normalized embedding on sample.embedding."""

import logging
from typing import Optional, List

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class EmbeddingModule(PipelineModule):
    name = "embedding"
    description = "Calculates X-CLIP embeddings for similarity search"
    default_config = {
        "model_name": "microsoft/xclip-base-patch32",
        "num_frames": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "microsoft/xclip-base-patch32")
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from transformers import XCLIPProcessor, XCLIPModel
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            logger.info(f"Loading X-CLIP ({self.model_name}) on {self._device}...")

            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)

            def load_xclip():
                model = from_pretrained_with_attention(
                    XCLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                    use_safetensors=True,
                ).to(self._device).eval()
                model.config.return_dict = True
                processor = XCLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._model, self._processor = shared_runtime_resource(
                self,
                (
                    "hf_xclip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "safetensors",
                ),
                load_xclip,
            )

            self._ml_available = True

        except ImportError:
            logger.warning("Transformers/X-CLIP not installed. Embedding calculation disabled.")
        except Exception as e:
            logger.error(f"Failed to load X-CLIP model: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        # If we already have an embedding (maybe loaded from cache), skip
        if sample.embedding is not None:
            return sample

        try:
            num_frames = self.config.get("num_frames", 8)
            frames = self._extract_frames(sample.path, num_frames=num_frames)
            if not frames:
                return sample

            embedding = self._compute_embedding(frames)

            if embedding is not None:
                sample.embedding = embedding

        except Exception as e:
            logger.warning(f"Embedding calculation failed for {sample.path}: {e}")

        return sample

    def _extract_frames(self, video_path, num_frames=8):
        try:
            frames = sample_frames(video_path, max_frames=num_frames, color="rgb")
            if frames and len(frames) < num_frames:
                frames = frames + [frames[-1]] * (num_frames - len(frames))
        except Exception as e:
            logger.debug(f"Failed to load frames for embedding: {e}")
            frames = []

        return frames if frames else None

    def _compute_embedding(self, frames) -> Optional[List[float]]:
        import torch

        with torch.no_grad():
            # Prepare inputs
            pil_images = arrays_to_pil(frames)

            inputs = self._processor(
                text=None,  # We only want visual embedding
                videos=[pil_images],  # X-CLIP expects list-of-clips, each a list of frames
                return_tensors="pt",
                padding=True,
            )

            pixel_values = inputs["pixel_values"]  # [batch_size*num_frames, C, H, W]
            # X-CLIP expects [batch_size, num_frames, C, H, W]

            # We processed a single video (batch_size=1) with num_frames
            # The processor might have flattened it or not depending on config.
            # Transformers XCLIPProcessor usually returns [batch, num_frames, C, H, W] if passing list of lists?
            # No, we passed a list of images (one video).
            # Let's check shape.

            if len(pixel_values.shape) == 4:
                # It treated them as separate images. Reshape.
                # [num_frames, C, H, W] -> [1, num_frames, C, H, W]
                pixel_values = pixel_values.unsqueeze(0)

            pixel_values = pixel_values.to(self._device)

            # We need to manually call the vision model part to get the video embedding
            # Or use get_image_features if it supports video?
            # XCLIPModel has get_video_features? No, it's typically get_image_features for CLIP.
            # But X-CLIP is different.

            # Let's look at embedding_calculator.py again.
            # It calls model.vision_model, then visual_projection, then model.mit

            # Simplified flow using standard transformers API if available:
            # model.get_video_features(**inputs)?
            # HuggingFace XCLIPModel has get_video_features.

            # Manual forward pass — handles both named-tuple and plain-tuple returns
            # from vision_model and mit (transformers version compatibility)
            batch_size_v = pixel_values.shape[0]
            num_frames_v = pixel_values.shape[1]
            pv_flat = pixel_values.reshape(-1, *pixel_values.shape[2:])

            vision_outputs = self._model.vision_model(pixel_values=pv_flat)
            frame_embeds = vision_outputs[1] if isinstance(vision_outputs, tuple) else vision_outputs.pooler_output
            frame_embeds = self._model.visual_projection(frame_embeds)

            cls_features = frame_embeds.view(batch_size_v, num_frames_v, -1)

            mit_outputs = self._model.mit(cls_features)
            video_features = mit_outputs[1] if isinstance(mit_outputs, tuple) else mit_outputs.pooler_output

            # Normalize
            video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)

            return video_features[0].cpu().tolist()

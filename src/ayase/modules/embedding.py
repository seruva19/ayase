"""X-CLIP video embedding computation for similarity search and downstream analysis.

Extracts normalized video-level features via X-CLIP vision model with MIT
(temporal attention). Stores L2-normalized embedding on sample.embedding."""

import logging
import cv2
import numpy as np
from PIL import Image
from typing import Optional, List

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

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Loading X-CLIP ({self.model_name}) on {self._device}...")

            self._model = XCLIPModel.from_pretrained(
                self.model_name, cache_dir="models", use_safetensors=True
            )
            self._model = self._model.to(self._device)
            self._model.eval()
            self._model.config.return_dict = True

            self._processor = XCLIPProcessor.from_pretrained(self.model_name, cache_dir="models")

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
        frames = []
        try:
            if str(video_path).lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                # It's an image
                img = cv2.imread(str(video_path))
                if img is not None:
                    # Repeat the image num_frames times to simulate a video for X-CLIP
                    # Or just use 1 frame if model supports it, but X-CLIP expects video
                    # Actually X-CLIP is a video model, so it expects a sequence.
                    # We can just duplicate the image.
                    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames = [frame] * num_frames
            else:
                # Video
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if total_frames > 0:
                    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
        except Exception as e:
            logger.debug(f"Failed to load frames for embedding: {e}")

        return frames if frames else None

    def _compute_embedding(self, frames) -> Optional[List[float]]:
        import torch

        with torch.no_grad():
            # Prepare inputs
            pil_images = [Image.fromarray(f) for f in frames]

            inputs = self._processor(
                text=None,  # We only want visual embedding
                images=pil_images,
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

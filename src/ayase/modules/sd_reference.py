"""SD Score — EvalCrafter metric #12.

Generates reference images from the caption using Stable Diffusion XL, then
measures CLIP cosine similarity between video frames and the generated images.
Higher score = video aligns better with what SDXL would produce from the prompt.

Only the real SDXL + CLIP pipeline produces ``sd_score``. Earlier revisions fell
back to a plain CLIP text-image similarity (i.e. CLIPScore) when SDXL was
unavailable; that is a different quantity, not SD Score, so the proxy has been
removed. When SDXL cannot be loaded the score is left ``None``.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_features,
    media_state_key,
)

logger = logging.getLogger(__name__)


class SDReferenceModule(PipelineModule):
    name = "sd_reference"
    description = "SD Score — CLIP similarity between video frames and SDXL-generated reference images"
    default_config = {
        "clip_model": "openai/clip-vit-base-patch32",
        "sdxl_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "num_sd_images": 5,
        "num_video_frames": 8,
        "sd_steps": 20,
        "cache_dir": ".ayase_sd_cache",
    }
    metric_groups = {
        "sd_score": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
        self.sdxl_model_name = self.config.get("sdxl_model", "stabilityai/stable-diffusion-xl-base-1.0")
        self.num_sd_images = self.config.get("num_sd_images", 5)
        self.num_video_frames = self.config.get("num_video_frames", 8)
        self.sd_steps = self.config.get("sd_steps", 20)
        self.cache_dir = Path(self.config.get("cache_dir", ".ayase_sd_cache"))
        self._clip_model = None
        self._clip_processor = None
        self._sd_pipe = None
        self._device = "cpu"
        self._ml_available = False
        self._sd_available = False
        self._backend = None

    def setup(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved_clip = resolve_model_path(self.clip_model_name, models_dir)
            logger.info(f"Loading CLIP for SD Score on {self._device}...")

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved_clip,
                    self.config,
                    device=self._device,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved_clip)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved_clip,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )
            self._ml_available = True
        except Exception as e:
            logger.warning(f"Failed to load CLIP for SD Score: {e}")
            self._backend = "unavailable"
            return

        # SDXL is required for the real SD Score. It needs a CUDA GPU for the
        # fp16 variant; without it the metric is left unset (no proxy).
        if self._device == "cuda":
            try:
                from diffusers import DiffusionPipeline

                resolved_sdxl = resolve_model_path(self.sdxl_model_name, models_dir)
                logger.info(f"Loading SDXL ({self.sdxl_model_name})...")
                self._sd_pipe = DiffusionPipeline.from_pretrained(
                    resolved_sdxl,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                )
                self._sd_pipe.to(self._device)
                self._sd_available = True
                self._backend = "sdxl_clip"
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"SDXL not available ({e}); SD Score left unset.")

        if not self._sd_available:
            self._backend = "unavailable"
            logger.warning("SD Score: SDXL unavailable (requires diffusers + CUDA); sd_score left unset.")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not self._sd_available:
            return sample
        if not sample.caption:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            prompt = sample.caption.text
            frame_embeds = self._embed_frames(sample, frames)  # [T, D]
            sd_embeds = self._get_sd_embeds(prompt)  # [K, D]
            score = self._compute_sd_score(frame_embeds, sd_embeds)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.sd_score = float(score)

        except Exception as e:
            logger.warning(f"SD Score failed for {sample.path}: {e}")

        return sample

    def _embed_frames(self, sample: Sample, frames):
        return cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
            arrays_to_pil(frames),
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=(self.num_video_frames, media_state_key(sample.path)),
        )

    def _get_sd_embeds(self, prompt: str):
        from PIL import Image

        pil_images = []
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        for i in range(self.num_sd_images):
            cache_path = self.cache_dir / f"{prompt_hash}_{i}.png"
            if cache_path.exists():
                pil_img = Image.open(cache_path).convert("RGB")
            else:
                result = self._sd_pipe(
                    prompt, height=512, width=512,
                    num_inference_steps=self.sd_steps,
                )
                pil_img = result.images[0]
                pil_img.save(cache_path)
            pil_images.append(pil_img)

        return cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
            pil_images,
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=(
                "sd_reference_images",
                prompt_hash,
                self.num_sd_images,
                self.sd_steps,
                self.sdxl_model_name,
            ),
        )

    def _compute_sd_score(self, frame_embeds, sd_embeds):
        # [T, D] @ [D, K] -> [T, K], average everything
        sim_matrix = frame_embeds @ sd_embeds.T
        return float(sim_matrix.mean().item())

    def _load_frames(self, sample: Sample):
        try:
            return sample_frames(sample.path, max_frames=self.num_video_frames, color="rgb")
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
        return []

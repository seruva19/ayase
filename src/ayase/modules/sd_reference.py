"""SD Score — EvalCrafter metric #12.

Generates reference images from the caption using Stable Diffusion XL, then
measures CLIP cosine similarity between video frames and the generated images.
Higher score = video aligns better with what SDXL would produce from the prompt.

If SDXL is not available (no GPU / not installed), falls back to CLIP
text-image similarity as a lightweight proxy.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

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

    def setup(self):
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            from ayase.config import resolve_model_path

            models_dir = self.config.get("models_dir", "models")
            resolved_clip = resolve_model_path(self.clip_model_name, models_dir)
            logger.info(f"Loading CLIP for SD Score on {self._device}...")
            self._clip_model = CLIPModel.from_pretrained(resolved_clip).to(self._device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(resolved_clip)
            self._ml_available = True
        except Exception as e:
            logger.warning(f"Failed to load CLIP for SD Score: {e}")
            return

        # Try loading SDXL (optional, heavy)
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
                self._sd_pipe.to("cuda")
                self._sd_available = True
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.info(f"SDXL not available ({e}). SD Score will use CLIP text-image proxy.")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.caption:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            prompt = sample.caption.text
            frame_embeds = self._embed_frames(frames)  # [T, D]

            if self._sd_available:
                sd_embeds = self._get_sd_embeds(prompt)  # [K, D]
                score = self._compute_sd_score(frame_embeds, sd_embeds)
            else:
                # Fallback: CLIP text-image similarity as proxy
                score = self._compute_text_proxy(frame_embeds, prompt)

            from ayase.models import QualityMetrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.sd_score = float(score)

        except Exception as e:
            logger.warning(f"SD Score failed for {sample.path}: {e}")

        return sample

    def _embed_frames(self, frames):
        import torch
        from PIL import Image

        pil_frames = [Image.fromarray(f) for f in frames]
        embeds = []
        with torch.no_grad():
            for img in pil_frames:
                inputs = self._clip_processor(images=img, return_tensors="pt").to(self._device)
                feat = self._clip_model.get_image_features(**inputs)
                embeds.append(feat)
        embeds = torch.cat(embeds, dim=0)
        return embeds / embeds.norm(p=2, dim=-1, keepdim=True)

    def _get_sd_embeds(self, prompt: str):
        import torch
        from PIL import Image

        embeds = []
        for i in range(self.num_sd_images):
            cache_path = self.cache_dir / f"{hashlib.md5(prompt.encode()).hexdigest()}_{i}.png"
            if cache_path.exists():
                pil_img = Image.open(cache_path).convert("RGB")
            else:
                result = self._sd_pipe(
                    prompt, height=512, width=512,
                    num_inference_steps=self.sd_steps,
                )
                pil_img = result.images[0]
                pil_img.save(cache_path)

            with torch.no_grad():
                inputs = self._clip_processor(images=pil_img, return_tensors="pt").to(self._device)
                feat = self._clip_model.get_image_features(**inputs)
                embeds.append(feat)

        embeds = torch.cat(embeds, dim=0)
        return embeds / embeds.norm(p=2, dim=-1, keepdim=True)

    def _compute_sd_score(self, frame_embeds, sd_embeds):
        # [T, D] @ [D, K] -> [T, K], average everything
        sim_matrix = frame_embeds @ sd_embeds.T
        return float(sim_matrix.mean().item())

    def _compute_text_proxy(self, frame_embeds, prompt):
        import torch

        with torch.no_grad():
            inputs = self._clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(self._device)
            text_feat = self._clip_model.get_text_features(**inputs)
            text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)
        sims = frame_embeds @ text_feat.T  # [T, 1]
        return float(sims.mean().item())

    def _load_frames(self, sample: Sample):
        frames = []
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    return frames
                n = min(self.num_video_frames, total)
                indices = np.linspace(0, total - 1, n, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            else:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
        return frames

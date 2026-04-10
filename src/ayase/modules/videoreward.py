"""VideoReward --- Kling Multi-Dimensional Reward Model (NeurIPS 2025).

HuggingFace: https://huggingface.co/KlingTeam/VideoReward

Uses CLIP backbone to encode video frames and compute multi-dimensional
reward scores:
  - Visual Quality (VQ): CLIP similarity with quality text prompts
  - Motion Quality (MQ): temporal consistency of CLIP frame embeddings
  - Text Alignment (TA): CLIP cosine similarity with the sample caption

videoreward_vq, videoreward_mq, videoreward_ta --- all higher = better (0-1)
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_QUALITY_PROMPTS = [
    "a high quality video with sharp details",
    "a clear video with good exposure",
    "a professional video with vivid colors",
]
_LOW_QUALITY_PROMPTS = [
    "a blurry low quality video",
    "a noisy distorted video",
]


class VideoRewardModule(PipelineModule):
    name = "videoreward"
    description = "VideoReward Kling multi-dim reward model (NeurIPS 2025)"
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
        self._model = None
        self._processor = None
        self._device = None
        self._quality_embeds = None
        self._low_quality_embeds = None

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._model = CLIPModel.from_pretrained(self._clip_model_name)
            self._processor = CLIPProcessor.from_pretrained(self._clip_model_name)
            self._model.to(self._device).eval()

            # Pre-encode quality prompts
            with torch.no_grad():
                q_inputs = self._processor(
                    text=_QUALITY_PROMPTS, return_tensors="pt", padding=True
                ).to(self._device)
                self._quality_embeds = self._model.get_text_features(**q_inputs)
                self._quality_embeds = self._quality_embeds / self._quality_embeds.norm(
                    dim=-1, keepdim=True
                )

                lq_inputs = self._processor(
                    text=_LOW_QUALITY_PROMPTS, return_tensors="pt", padding=True
                ).to(self._device)
                self._low_quality_embeds = self._model.get_text_features(**lq_inputs)
                self._low_quality_embeds = self._low_quality_embeds / self._low_quality_embeds.norm(
                    dim=-1, keepdim=True
                )

            self._ml_available = True
            logger.info("VideoReward (CLIP) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("VideoReward setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            scores = self._compute_rewards(frames, sample)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.videoreward_vq = scores["vq"]
            sample.quality_metrics.videoreward_mq = scores["mq"]
            sample.quality_metrics.videoreward_ta = scores["ta"]

            logger.debug(
                "VideoReward for %s: vq=%.3f mq=%.3f ta=%.3f",
                sample.path.name,
                scores["vq"],
                scores["mq"],
                scores["ta"],
            )
        except Exception as e:
            logger.warning("VideoReward failed for %s: %s", sample.path, e)
        return sample

    def _compute_rewards(
        self, frames: List[np.ndarray], sample: Sample
    ) -> Dict[str, float]:
        import torch
        from PIL import Image

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

        # Encode all frames
        frame_embeds = []
        with torch.no_grad():
            for pil_img in pil_frames:
                inputs = self._processor(
                    images=pil_img, return_tensors="pt"
                ).to(self._device)
                emb = self._model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                frame_embeds.append(emb)

        frame_stack = torch.cat(frame_embeds, dim=0)  # (N, D)

        # --- Visual Quality: CLIP similarity with quality prompts ---
        with torch.no_grad():
            pos_sims = (frame_stack @ self._quality_embeds.T).mean(dim=-1)
            neg_sims = (frame_stack @ self._low_quality_embeds.T).mean(dim=-1)
            vq_logits = (pos_sims - neg_sims) * 10.0
            vq_scores = torch.sigmoid(vq_logits)
            vq = float(vq_scores.mean().item())

        # --- Motion Quality: temporal smoothness of frame embeddings ---
        if len(frame_embeds) > 1:
            cosine_sims = []
            for i in range(len(frame_embeds) - 1):
                sim = torch.nn.functional.cosine_similarity(
                    frame_embeds[i], frame_embeds[i + 1]
                ).item()
                cosine_sims.append(sim)
            # High mean similarity + low variance = smooth motion
            mean_sim = float(np.mean(cosine_sims))
            var_sim = float(np.var(cosine_sims))
            mq = float(np.clip(mean_sim * (1.0 / (1.0 + var_sim * 100.0)), 0.0, 1.0))
        else:
            mq = 0.5

        # --- Text Alignment: CLIP cosine with caption ---
        caption = getattr(sample, "caption", None)
        caption_text = None
        if caption and hasattr(caption, "text"):
            caption_text = caption.text
        elif isinstance(caption, str):
            caption_text = caption

        if caption_text:
            with torch.no_grad():
                text_inputs = self._processor(
                    text=[caption_text], return_tensors="pt", padding=True, truncation=True
                ).to(self._device)
                text_emb = self._model.get_text_features(**text_inputs)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                sims = (frame_stack @ text_emb.T).squeeze(-1)
                # Map from CLIP cosine range (~0.15-0.35) to 0-1
                ta = float(torch.clamp((sims.mean() - 0.15) / 0.20, 0.0, 1.0).item())
        else:
            # No caption: use mid-range score
            ta = 0.5

        return {
            "vq": float(np.clip(vq, 0.0, 1.0)),
            "mq": float(np.clip(mq, 0.0, 1.0)),
            "ta": float(np.clip(ta, 0.0, 1.0)),
        }

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return []
        indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, f = cap.read()
            if ret:
                frames.append(f)
        cap.release()
        return frames

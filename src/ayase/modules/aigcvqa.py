"""AIGC-VQA --- Holistic Perception for AIGC Video Quality (CVPRW 2024).

3-branch assessment using CLIP features:
  - Technical quality: CLIP similarity with technical quality prompts
  - Aesthetic quality: CLIP similarity with aesthetic quality prompts
  - Text-video alignment: CLIP cosine similarity with sample caption

aigcvqa_technical, aigcvqa_aesthetic, aigcvqa_alignment --- all 0-1, higher = better
"""

import logging
from typing import Dict, List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_TECHNICAL_POS = [
    "a sharp video with clear details",
    "a video with no noise or compression artifacts",
    "a well-exposed video with good dynamic range",
]
_TECHNICAL_NEG = [
    "a blurry video with noise",
    "a video with compression artifacts and distortion",
    "an overexposed or underexposed video",
]
_AESTHETIC_POS = [
    "a beautiful video with good composition",
    "a visually pleasing video with harmonious colors",
    "an artistically composed video",
]
_AESTHETIC_NEG = [
    "an ugly video with bad composition",
    "a visually unpleasant video with clashing colors",
    "a poorly composed video",
]


class AIGCVQAModule(PipelineModule):
    name = "aigcvqa"
    description = "AIGC-VQA holistic 3-branch AIGC perception (CVPRW 2024)"
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
        self._device = "cpu"
        # Pre-encoded prompt embeddings
        self._tech_pos = None
        self._tech_neg = None
        self._aes_pos = None
        self._aes_neg = None

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

            def _encode_texts(texts):
                with torch.no_grad():
                    inp = self._processor(
                        text=texts, return_tensors="pt", padding=True
                    ).to(self._device)
                    emb = self._model.get_text_features(**inp)
                    return emb / emb.norm(dim=-1, keepdim=True)

            self._tech_pos = _encode_texts(_TECHNICAL_POS)
            self._tech_neg = _encode_texts(_TECHNICAL_NEG)
            self._aes_pos = _encode_texts(_AESTHETIC_POS)
            self._aes_neg = _encode_texts(_AESTHETIC_NEG)

            self._ml_available = True
            logger.info("AIGC-VQA (CLIP 3-branch) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("AIGC-VQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            scores = self._compute_scores(frames, sample)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.aigcvqa_technical = scores["technical"]
            sample.quality_metrics.aigcvqa_aesthetic = scores["aesthetic"]
            sample.quality_metrics.aigcvqa_alignment = scores["alignment"]

            logger.debug(
                "AIGC-VQA for %s: tech=%.3f aes=%.3f align=%.3f",
                sample.path.name,
                scores["technical"],
                scores["aesthetic"],
                scores["alignment"],
            )
        except Exception as e:
            logger.warning("AIGC-VQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_scores(
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

        # --- Technical branch ---
        with torch.no_grad():
            tech_pos_sim = (frame_stack @ self._tech_pos.T).mean(dim=-1)
            tech_neg_sim = (frame_stack @ self._tech_neg.T).mean(dim=-1)
            tech_logits = (tech_pos_sim - tech_neg_sim) * 10.0
            technical = float(torch.sigmoid(tech_logits).mean().item())

        # --- Aesthetic branch ---
        with torch.no_grad():
            aes_pos_sim = (frame_stack @ self._aes_pos.T).mean(dim=-1)
            aes_neg_sim = (frame_stack @ self._aes_neg.T).mean(dim=-1)
            aes_logits = (aes_pos_sim - aes_neg_sim) * 10.0
            aesthetic = float(torch.sigmoid(aes_logits).mean().item())

        # --- Alignment branch: CLIP cosine with caption ---
        caption = getattr(sample, "caption", None)
        caption_text = None
        if caption and hasattr(caption, "text"):
            caption_text = caption.text
        elif isinstance(caption, str):
            caption_text = caption

        if caption_text:
            with torch.no_grad():
                text_inputs = self._processor(
                    text=[caption_text], return_tensors="pt", padding=True,
                    truncation=True,
                ).to(self._device)
                text_emb = self._model.get_text_features(**text_inputs)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                sims = (frame_stack @ text_emb.T).squeeze(-1)
                # Map from CLIP cosine range (~0.15-0.35) to 0-1
                alignment = float(
                    torch.clamp((sims.mean() - 0.15) / 0.20, 0.0, 1.0).item()
                )
        else:
            alignment = 0.5  # No caption available

        return {
            "technical": float(np.clip(technical, 0.0, 1.0)),
            "aesthetic": float(np.clip(aesthetic, 0.0, 1.0)),
            "alignment": float(np.clip(alignment, 0.0, 1.0)),
        }

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(
                0, total - 1, min(self.subsample, total), dtype=int
            )
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

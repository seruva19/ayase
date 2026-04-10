"""T2VEval --- Text-to-Video Generated Video Evaluation (2025).

Evaluates text-video consistency and realness using CLIP:
  - Alignment: CLIP cosine similarity between frames and caption
  - Realness: CLIP discriminative features measuring distance from
    "real" vs "generated" text embeddings
  - Quality: CLIP quality-prompt scoring

t2veval_score --- higher = better (0-1 range)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_REAL_PROMPTS = [
    "a real natural video",
    "an authentic video captured by a camera",
    "a realistic video of a real scene",
]
_FAKE_PROMPTS = [
    "an AI generated synthetic video",
    "a computer generated artificial video",
    "a fake video produced by a machine",
]
_QUALITY_POS = [
    "a high quality well-produced video",
    "a sharp clear video with good detail",
]
_QUALITY_NEG = [
    "a low quality poorly produced video",
    "a blurry noisy video with artifacts",
]


class T2VEvalModule(PipelineModule):
    name = "t2veval"
    description = "T2VEval text-video consistency+realness (2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
        "alignment_weight": 0.35,
        "realness_weight": 0.35,
        "quality_weight": 0.30,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._w_align = self.config.get("alignment_weight", 0.35)
        self._w_real = self.config.get("realness_weight", 0.35)
        self._w_quality = self.config.get("quality_weight", 0.30)
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = None
        self._real_embeds = None
        self._fake_embeds = None
        self._quality_pos = None
        self._quality_neg = None

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

            self._real_embeds = _encode_texts(_REAL_PROMPTS)
            self._fake_embeds = _encode_texts(_FAKE_PROMPTS)
            self._quality_pos = _encode_texts(_QUALITY_POS)
            self._quality_neg = _encode_texts(_QUALITY_NEG)

            self._ml_available = True
            logger.info("T2VEval (CLIP) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("T2VEval setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            score = self._compute_score(frames, sample)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.t2veval_score = score
            logger.debug("T2VEval for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("T2VEval failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(
        self, frames: List[np.ndarray], sample: Sample
    ) -> Optional[float]:
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

        if not frame_embeds:
            return None

        frame_stack = torch.cat(frame_embeds, dim=0)  # (N, D)

        # --- Realness: distance from "real" vs "generated" prompts ---
        with torch.no_grad():
            real_sim = (frame_stack @ self._real_embeds.T).mean(dim=-1)
            fake_sim = (frame_stack @ self._fake_embeds.T).mean(dim=-1)
            real_logits = (real_sim - fake_sim) * 10.0
            realness = float(torch.sigmoid(real_logits).mean().item())

        # --- Visual Quality ---
        with torch.no_grad():
            qp_sim = (frame_stack @ self._quality_pos.T).mean(dim=-1)
            qn_sim = (frame_stack @ self._quality_neg.T).mean(dim=-1)
            q_logits = (qp_sim - qn_sim) * 10.0
            quality = float(torch.sigmoid(q_logits).mean().item())

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
                    text=[caption_text], return_tensors="pt", padding=True,
                    truncation=True,
                ).to(self._device)
                text_emb = self._model.get_text_features(**text_inputs)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
                sims = (frame_stack @ text_emb.T).squeeze(-1)
                # Map from typical CLIP cosine range to 0-1
                alignment = float(
                    torch.clamp((sims.mean() - 0.15) / 0.20, 0.0, 1.0).item()
                )
        else:
            # Without caption: rely on quality + realness only
            alignment = 0.5

        score = (
            self._w_align * alignment
            + self._w_real * realness
            + self._w_quality * quality
        )
        return float(np.clip(score, 0.0, 1.0))

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
                ret, f = cap.read()
                if ret:
                    frames.append(f)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

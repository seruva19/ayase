"""AIGVQA --- Multi-Dimensional AI-Generated VQA (ICCVW 2025).

GitHub: https://github.com/IntMeGroup/AIGVQA

Multi-dimensional quality scoring using CLIP features:
  - Spatial quality via CLIP quality-prompt similarity
  - Temporal consistency via CLIP frame embedding coherence
  - Aesthetic quality via CLIP aesthetic-prompt similarity

Final score is a weighted combination of all three dimensions.

aigvqa_score --- higher = better (0-1 range)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_SPATIAL_POS = [
    "a sharp image with fine details",
    "a clear frame with good resolution",
]
_SPATIAL_NEG = [
    "a blurry image lacking detail",
    "a pixelated low resolution frame",
]
_AESTHETIC_POS = [
    "a beautiful and visually appealing scene",
    "an image with good color and composition",
]
_AESTHETIC_NEG = [
    "an ugly and unpleasant scene",
    "an image with bad colors and poor composition",
]


class AIGVQAModule(PipelineModule):
    name = "aigvqa"
    description = "AIGVQA multi-dimensional AIGC VQA (ICCVW 2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
        "spatial_weight": 0.4,
        "temporal_weight": 0.3,
        "aesthetic_weight": 0.3,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._w_spatial = self.config.get("spatial_weight", 0.4)
        self._w_temporal = self.config.get("temporal_weight", 0.3)
        self._w_aesthetic = self.config.get("aesthetic_weight", 0.3)
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = None
        self._spatial_pos = None
        self._spatial_neg = None
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

            self._spatial_pos = _encode_texts(_SPATIAL_POS)
            self._spatial_neg = _encode_texts(_SPATIAL_NEG)
            self._aes_pos = _encode_texts(_AESTHETIC_POS)
            self._aes_neg = _encode_texts(_AESTHETIC_NEG)

            self._ml_available = True
            logger.info("AIGVQA (CLIP multi-dim) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("AIGVQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            score = self._compute_score(frames)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.aigvqa_score = score
            logger.debug("AIGVQA for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("AIGVQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_score(self, frames: List[np.ndarray]) -> Optional[float]:
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

        # --- Spatial quality ---
        with torch.no_grad():
            sp_pos = (frame_stack @ self._spatial_pos.T).mean(dim=-1)
            sp_neg = (frame_stack @ self._spatial_neg.T).mean(dim=-1)
            sp_logits = (sp_pos - sp_neg) * 10.0
            spatial = float(torch.sigmoid(sp_logits).mean().item())

        # --- Aesthetic quality ---
        with torch.no_grad():
            ae_pos = (frame_stack @ self._aes_pos.T).mean(dim=-1)
            ae_neg = (frame_stack @ self._aes_neg.T).mean(dim=-1)
            ae_logits = (ae_pos - ae_neg) * 10.0
            aesthetic = float(torch.sigmoid(ae_logits).mean().item())

        # --- Temporal consistency: cosine similarity between consecutive frames ---
        if len(frame_embeds) > 1:
            cosine_sims = []
            for i in range(len(frame_embeds) - 1):
                sim = torch.nn.functional.cosine_similarity(
                    frame_embeds[i], frame_embeds[i + 1]
                ).item()
                cosine_sims.append(sim)
            # Combine mean consistency and smoothness (low variance)
            mean_sim = float(np.mean(cosine_sims))
            var_sim = float(np.var(cosine_sims))
            temporal = float(np.clip(
                mean_sim * (1.0 / (1.0 + var_sim * 50.0)), 0.0, 1.0
            ))
        else:
            temporal = 1.0  # Single frame: perfect temporal consistency

        score = (
            self._w_spatial * spatial
            + self._w_temporal * temporal
            + self._w_aesthetic * aesthetic
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

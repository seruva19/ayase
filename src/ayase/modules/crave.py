"""CRAVE --- Content-Rich AIGC Video Evaluator (2025).

GitHub: https://github.com/littlespray/CRAVE

Designed for Sora-era videos. Uses CLIP for content understanding +
temporal consistency.  Scores based on:
  - Content richness: diversity of CLIP features across frames
  - Visual quality: CLIP similarity with quality prompts
  - Temporal coherence: smoothness of CLIP embeddings over time

crave_score --- higher = better quality (0-1 range)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_QUALITY_POS = [
    "a high quality video with rich content",
    "a detailed video with diverse visual elements",
    "a sharp clear professional video",
]
_QUALITY_NEG = [
    "a low quality video with poor content",
    "a repetitive monotonous video",
    "a blurry noisy amateur video",
]


class CRAVEModule(PipelineModule):
    name = "crave"
    description = "CRAVE content-rich AIGC video evaluator (2025)"
    default_config = {
        "subsample": 12,
        "clip_model": "openai/clip-vit-base-patch32",
        "quality_weight": 0.35,
        "richness_weight": 0.35,
        "coherence_weight": 0.30,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 12)
        self._clip_model_name = self.config.get(
            "clip_model", "openai/clip-vit-base-patch32"
        )
        self._w_quality = self.config.get("quality_weight", 0.35)
        self._w_richness = self.config.get("richness_weight", 0.35)
        self._w_coherence = self.config.get("coherence_weight", 0.30)
        self._ml_available = False
        self._model = None
        self._processor = None
        self._device = "cpu"
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

            self._quality_pos = _encode_texts(_QUALITY_POS)
            self._quality_neg = _encode_texts(_QUALITY_NEG)

            self._ml_available = True
            logger.info("CRAVE (CLIP content-rich) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("CRAVE setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            frames = self._extract_frames(sample)
            if len(frames) < 2:
                return sample

            score = self._compute_score(frames)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.crave_score = score
            logger.debug("CRAVE for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("CRAVE failed for %s: %s", sample.path, e)
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

        if len(frame_embeds) < 2:
            return None

        frame_stack = torch.cat(frame_embeds, dim=0)  # (N, D)

        # --- Visual Quality: CLIP similarity with quality prompts ---
        with torch.no_grad():
            pos_sim = (frame_stack @ self._quality_pos.T).mean(dim=-1)
            neg_sim = (frame_stack @ self._quality_neg.T).mean(dim=-1)
            q_logits = (pos_sim - neg_sim) * 10.0
            quality = float(torch.sigmoid(q_logits).mean().item())

        # --- Content Richness: diversity of CLIP features across frames ---
        # Higher diversity of embeddings = richer content
        with torch.no_grad():
            # Pairwise cosine similarity matrix
            sim_matrix = frame_stack @ frame_stack.T  # (N, N)
            n = sim_matrix.shape[0]
            # Mask diagonal
            mask = ~torch.eye(n, dtype=torch.bool, device=self._device)
            mean_pairwise_sim = sim_matrix[mask].mean().item()
            # Lower mean similarity = more diverse content
            # Map: sim=1.0 (no diversity) -> 0, sim=0.5 -> 1
            richness = float(np.clip(1.0 - (mean_pairwise_sim - 0.5) * 2.0, 0.0, 1.0))

        # --- Temporal Coherence: smoothness of consecutive embeddings ---
        cosine_sims = []
        for i in range(len(frame_embeds) - 1):
            sim = torch.nn.functional.cosine_similarity(
                frame_embeds[i], frame_embeds[i + 1]
            ).item()
            cosine_sims.append(sim)
        mean_sim = float(np.mean(cosine_sims))
        var_sim = float(np.var(cosine_sims))
        # High mean + low variance = smooth transitions
        coherence = float(np.clip(
            mean_sim * (1.0 / (1.0 + var_sim * 100.0)), 0.0, 1.0
        ))

        score = (
            self._w_quality * quality
            + self._w_richness * richness
            + self._w_coherence * coherence
        )
        return float(np.clip(score, 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(str(sample.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 1:
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
        return frames

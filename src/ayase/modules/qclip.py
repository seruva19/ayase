"""Q-CLIP --- VLM-Based VQA via Cross-Modal Quality Adaptation (2025).

Uses CLIP model with quality-aware prompts.  Score = CLIP similarity
between video frames and quality text embeddings (similar to CLIP-IQA
but extended for video with temporal quality aggregation).

qclip_score --- higher = better (0-1 range)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Quality-level prompts for softmax scoring (modelled after CLIP-IQA)
_QUALITY_LEVELS = [
    ("excellent quality, perfectly sharp, superb detail", 1.0),
    ("good quality, mostly sharp, clear detail", 0.80),
    ("decent quality, reasonably clear", 0.60),
    ("fair quality, slightly blurry or noisy", 0.40),
    ("poor quality, blurry and noisy", 0.20),
    ("terrible quality, very blurry, heavy noise and artifacts", 0.0),
]


class QCLIPModule(PipelineModule):
    name = "qclip"
    description = "Q-CLIP VLM-based VQA (2025)"
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
        self._quality_embeds = None
        self._quality_weights = None

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

            # Pre-encode quality level prompts
            texts = [q[0] for q in _QUALITY_LEVELS]
            self._quality_weights = np.array([q[1] for q in _QUALITY_LEVELS])

            with torch.no_grad():
                text_inputs = self._processor(
                    text=texts, return_tensors="pt", padding=True
                ).to(self._device)
                self._quality_embeds = self._model.get_text_features(**text_inputs)
                self._quality_embeds = self._quality_embeds / self._quality_embeds.norm(
                    dim=-1, keepdim=True
                )

            self._ml_available = True
            logger.info("Q-CLIP (CLIP quality) initialised on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("Q-CLIP setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            score = self._compute_quality_score(frames)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.qclip_score = score
            logger.debug("Q-CLIP for %s: %.4f", sample.path.name, score)

        except Exception as e:
            logger.warning("Q-CLIP failed for %s: %s", sample.path, e)
        return sample

    def _compute_quality_score(self, frames: List[np.ndarray]) -> Optional[float]:
        """Softmax over quality-level prompts, weighted average as final score."""
        import torch
        from PIL import Image

        pil_frames = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames
        ]

        frame_scores = []
        with torch.no_grad():
            for pil_img in pil_frames:
                inputs = self._processor(
                    images=pil_img, return_tensors="pt"
                ).to(self._device)
                img_emb = self._model.get_image_features(**inputs)
                img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

                # Cosine similarity with each quality level
                sims = (img_emb @ self._quality_embeds.T).squeeze(0).cpu().numpy()
                # Temperature-scaled softmax
                exp_sims = np.exp((sims - sims.max()) * 100.0)
                probs = exp_sims / exp_sims.sum()
                # Weighted quality score
                score = float(np.dot(probs, self._quality_weights))
                frame_scores.append(score)

        if not frame_scores:
            return None

        # For video: also consider temporal consistency
        if len(frame_scores) > 1:
            # Penalise high variance in per-frame quality
            mean_score = float(np.mean(frame_scores))
            score_var = float(np.var(frame_scores))
            temporal_penalty = 1.0 / (1.0 + score_var * 10.0)
            return float(np.clip(mean_score * temporal_penalty, 0.0, 1.0))

        return float(np.clip(frame_scores[0], 0.0, 1.0))

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

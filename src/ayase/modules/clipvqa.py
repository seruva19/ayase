"""CLIPVQA — Video Quality Assessment via CLIP.

IEEE TIP 2024 — CLIP-based frame encoder with self-attention
for spatiotemporal quality. 37% better generalizability than
existing VQA methods across 8 datasets.

GitHub: https://github.com/GZHU-DVL/CLIPVQA

clipvqa_score — higher = better quality
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CLIPVQAModule(PipelineModule):
    name = "clipvqa"
    description = "CLIPVQA CLIP-based spatiotemporal VQA (TIP 2024)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._clip_model = None
        self._clip_processor = None
        self._device = "cpu"
        self._backend = "heuristic"

    def setup(self) -> None:
        try:
            import clipvqa
            self._model = clipvqa
            self._backend = "native"
            logger.info("CLIPVQA (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: CLIP heuristic
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir="models"
            ).to(self._device).eval()
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32", cache_dir="models"
            )
            self._backend = "clip"
            logger.info(f"CLIPVQA (CLIP heuristic) initialised on {self._device}")
            return
        except (ImportError, Exception):
            pass

        self._backend = "heuristic"
        logger.info("CLIPVQA (heuristic) — install clipvqa or transformers for better accuracy")

    def process(self, sample: Sample) -> Sample:
        try:
            if self._backend == "native":
                score = float(self._model.predict(str(sample.path)))
            elif self._backend == "clip":
                score = self._process_clip(sample)
            else:
                score = self._process_heuristic(sample)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.clipvqa_score = score

        except Exception as e:
            logger.warning(f"CLIPVQA failed for {sample.path}: {e}")

        return sample

    def _process_clip(self, sample: Sample) -> Optional[float]:
        """CLIP features + self-attention temporal pooling approximation."""
        import torch
        from PIL import Image

        frames = self._extract_frames(sample)
        if not frames:
            return None

        quality_texts = [
            "a perfectly clear high quality image",
            "a very blurry low quality image",
        ]

        embeddings = []
        quality_scores = []

        with torch.no_grad():
            for frame in frames:
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self._clip_processor(
                    text=quality_texts, images=pil_img,
                    return_tensors="pt", padding=True,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._clip_model(**inputs)

                # Quality score from text similarity
                logits = outputs.logits_per_image.softmax(dim=-1)
                quality_scores.append(float(logits[0, 0].cpu()))

                # Image embedding for temporal attention
                img_embed = outputs.image_embeds[0].cpu().numpy()
                embeddings.append(img_embed)

        # Self-attention-like temporal weighting
        if len(embeddings) > 1:
            embed_matrix = np.array(embeddings)
            # Compute attention weights via dot product similarity
            norms = np.linalg.norm(embed_matrix, axis=1, keepdims=True)
            normalized = embed_matrix / (norms + 1e-8)
            attention = normalized @ normalized.T
            weights = np.mean(attention, axis=1)
            weights = np.exp(weights) / np.sum(np.exp(weights))
            score = float(np.dot(weights, quality_scores))
        else:
            score = quality_scores[0] if quality_scores else None

        return score

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: multi-scale quality with temporal attention-like weighting."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        per_frame = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 1.0)
            contrast = min(gray.std() / 65.0, 1.0)
            per_frame.append(0.6 * sharpness + 0.4 * contrast)

        # Attention-like: weight frames by their distinctiveness
        if len(per_frame) > 1:
            arr = np.array(per_frame)
            weights = np.exp(arr) / np.sum(np.exp(arr))
            score = float(np.dot(weights, arr))
        else:
            score = per_frame[0]

        return float(np.clip(score, 0.0, 1.0))

    def _extract_frames(self, sample: Sample):
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
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

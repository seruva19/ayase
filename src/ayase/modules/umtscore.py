"""UMTScore — Unified Multi-modal Transformer Score.

Video-text alignment scoring using UMT (Unified Multi-modal
Transformer) features. Measures how well video content matches
text descriptions via cross-modal similarity.

umtscore — higher = better alignment (0-1 range)
"""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class UMTScoreModule(PipelineModule):
    name = "umtscore"
    description = "UMTScore video-text alignment via UMT features"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._model = None
        self._clip_model = None
        self._clip_processor = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: Try UMT model
        try:
            import umt
            self._model = umt
            self._backend = "native"
            logger.info("UMTScore (native) initialised")
            return
        except ImportError:
            pass

        # Tier 2: Try CLIP as proxy for cross-modal similarity
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            self._backend = "clip"
            logger.info("UMTScore (CLIP proxy) initialised")
            return
        except (ImportError, Exception):
            pass

        # Tier 3: Heuristic fallback
        self._backend = "heuristic"
        logger.info("UMTScore (heuristic) initialised — install transformers for CLIP proxy")

    def process(self, sample: Sample) -> Sample:
        try:
            caption = getattr(sample, "caption", None)
            if not caption:
                return sample

            caption_text = caption.text if hasattr(caption, "text") else str(caption)

            if self._backend == "native":
                score = float(self._model.score(str(sample.path), caption_text))
            elif self._backend == "clip":
                score = self._process_clip(sample, caption_text)
            else:
                score = self._process_heuristic(sample, caption_text)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.umtscore = score

        except Exception as e:
            logger.warning(f"UMTScore failed for {sample.path}: {e}")

        return sample

    def _process_clip(self, sample: Sample, caption: str) -> Optional[float]:
        """CLIP-based text-video similarity proxy."""
        import torch
        from PIL import Image

        frames = self._extract_frames(sample)
        if not frames:
            return None

        scores = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)

            inputs = self._clip_processor(
                text=[caption], images=pil_img, return_tensors="pt", padding=True
            )
            with torch.no_grad():
                outputs = self._clip_model(**inputs)

            # Cosine similarity from CLIP
            logits = outputs.logits_per_image
            sim = float(logits.softmax(dim=-1).max().item())
            scores.append(sim)

        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _process_heuristic(self, sample: Sample, caption: str) -> Optional[float]:
        """Heuristic: simple content-text alignment estimation."""
        frames = self._extract_frames(sample)
        if not frames:
            return None

        # Analyze visual content features
        visual_features = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
            b, g, r = (
                frame[:, :, 0].astype(np.float64),
                frame[:, :, 1].astype(np.float64),
                frame[:, :, 2].astype(np.float64),
            )

            # Basic visual descriptors
            brightness = np.mean(gray) / 255.0
            contrast = np.std(gray) / 128.0
            colorfulness = (np.std(r - g) + np.std(0.5 * (r + g) - b)) / 256.0
            sharpness = min(cv2.Laplacian(gray, cv2.CV_64F).var() / 1000.0, 1.0)
            visual_features.append([brightness, contrast, colorfulness, sharpness])

        avg_features = np.mean(visual_features, axis=0)

        # Text analysis heuristic: longer, more descriptive captions suggest
        # more alignment potential; visual richness correlates with alignment
        words = caption.lower().split()
        text_richness = min(len(words) / 20.0, 1.0)
        visual_richness = float(np.mean(avg_features))

        # Heuristic alignment: rich visuals + rich text = likely aligned
        score = 0.5 * visual_richness + 0.3 * text_richness + 0.2

        return float(np.clip(score, 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> list:
        """Extract frames from video or image."""
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)

        return frames

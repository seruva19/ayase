"""VQAScore text-visual alignment module.

Uses VQA model to score "Does this figure show {text}?" probability.
ECCV 2024, outperforms CLIPScore on compositional text prompts.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VQAScoreModule(PipelineModule):
    name = "vqa_score"
    description = "VQAScore text-visual alignment via VQA probability (0-1, higher=better)"
    default_config = {
        "model": "clip-flant5-xxl",
        "subsample": 4,
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None

    def setup(self) -> None:
        try:
            from t2v_metrics import VQAScore as VQAScoreMetric

            model_name = self.config.get("model", "clip-flant5-xxl")
            self._model = VQAScoreMetric(model=model_name)
            self._ml_available = True
            logger.info("VQAScore model loaded: %s", model_name)
        except ImportError:
            # Fallback: use CLIP similarity as approximation
            try:
                import torch
                import clip

                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._clip_model, self._clip_preprocess = clip.load("ViT-B/32", device=device)
                self._device = device
                self._ml_available = True
                self._model = None  # signals CLIP fallback
                logger.info("VQAScore fallback: using CLIP on %s", device)
            except (ImportError, Exception) as e:
                logger.warning("VQAScore unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        if sample.caption is None or not sample.caption.text:
            return sample

        try:
            if self._model is not None:
                score = self._compute_vqascore(sample)
            else:
                score = self._compute_clip_fallback(sample)

            if score is not None:
                sample.quality_metrics.vqa_score_alignment = float(score)
        except Exception as e:
            logger.warning("VQAScore processing failed: %s", e)
        return sample

    def _compute_vqascore(self, sample: Sample) -> Optional[float]:
        """Compute VQAScore using t2v_metrics."""
        from PIL import Image

        text = sample.caption.text
        subsample = self.config.get("subsample", 4)

        if sample.is_video:
            import cv2

            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]

            scores = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    score = self._model(images=[pil_img], texts=[text]).item()
                    scores.append(score)
            cap.release()
            return float(np.mean(scores)) if scores else None
        else:
            img = Image.open(str(sample.path)).convert("RGB")
            return self._model(images=[img], texts=[text]).item()

    def _compute_clip_fallback(self, sample: Sample) -> Optional[float]:
        """Fallback: compute CLIP similarity."""
        import cv2
        import torch
        from PIL import Image

        text = sample.caption.text
        subsample = self.config.get("subsample", 4)

        text_tok = clip_tokenize = __import__("clip").tokenize([text]).to(self._device)

        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
            cap.release()
        else:
            frames.append(Image.open(str(sample.path)).convert("RGB"))

        if not frames:
            return None

        scores = []
        with torch.no_grad():
            text_features = self._clip_model.encode_text(text_tok)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            for img in frames:
                img_t = self._clip_preprocess(img).unsqueeze(0).to(self._device)
                img_features = self._clip_model.encode_image(img_t)
                img_features = img_features / img_features.norm(dim=-1, keepdim=True)
                sim = (img_features @ text_features.T).item()
                scores.append(max(0.0, sim))

        return float(np.mean(scores)) if scores else None

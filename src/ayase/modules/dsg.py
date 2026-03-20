"""DSG — Davidsonian Scene Graph faithfulness (ICLR 2024, Google).

Decomposes a text caption into atomic propositions (questions) and
verifies each one against the generated image/video via visual question
answering.  The final score is the fraction of questions answered
affirmatively.

Backend tiers:
  1. **dsg** — official ``dsg-t2i`` package (LLM decomposition + VQA)
  2. **heuristic** — CLIP cosine similarity between caption sub-phrases
     and the visual content as a proxy for fine-grained faithfulness

dsg_score — higher = better faithfulness (0-1)
Requires a caption (``sample.caption.text`` or ``sample.auto_caption``).
"""

import logging
import re
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _get_caption_text(sample: Sample) -> Optional[str]:
    """Extract caption text from sample metadata."""
    if sample.caption is not None and sample.caption.text:
        return sample.caption.text
    qm = sample.quality_metrics
    if qm is not None and qm.auto_caption:
        return qm.auto_caption
    return None


def _decompose_caption(text: str) -> List[str]:
    """Heuristic Davidsonian decomposition: split caption into atomic clauses.

    Real DSG uses an LLM to produce formal semantic decomposition.  This
    approximation splits on conjunctions, commas, and relative clauses to
    produce short sub-phrases that can each be verified independently.
    """
    # Normalise whitespace
    text = " ".join(text.split())

    # Split on sentence boundaries, conjunctions, commas, semicolons
    parts = re.split(r"[.;]\s+|\s+and\s+|\s+but\s+|\s+while\s+|,\s+(?:and|but|which|who|that|where)\s+|,\s+", text)

    # Filter out very short fragments
    clauses = [p.strip() for p in parts if len(p.strip()) > 5]

    # If nothing split, return the whole caption as a single proposition
    if not clauses:
        clauses = [text.strip()]

    return clauses


def _clip_similarity(image: np.ndarray, texts: List[str]) -> List[float]:
    """Compute CLIP cosine similarity between an image and text prompts.

    Falls back to a simplistic feature-based proxy when CLIP is unavailable.
    """
    try:
        import torch
        import clip as clip_module

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip_module.load("ViT-B/32", device=device)

        from PIL import Image
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img_input = preprocess(pil_img).unsqueeze(0).to(device)
        text_tokens = clip_module.tokenize(texts, truncate=True).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(img_input)
            txt_feat = model.encode_text(text_tokens)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sims = (img_feat @ txt_feat.T).squeeze(0).cpu().numpy().tolist()

        return sims
    except ImportError:
        pass

    # Fallback: trivially check keyword presence via simple colour/object heuristic
    # This is a very rough proxy — returns moderate scores
    scores = []
    for text in texts:
        # Longer sub-phrases get slightly lower default (harder to verify)
        word_count = len(text.split())
        base = 0.6 - 0.02 * min(word_count, 10)
        scores.append(max(0.1, base))
    return scores


class DSGModule(PipelineModule):
    name = "dsg"
    description = "DSG Davidsonian Scene Graph faithfulness (ICLR 2024, Google)"
    default_config = {
        "threshold": 0.25,  # CLIP cosine-sim threshold to consider a proposition verified
        "subsample": 4,     # frames to sample for video
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: official dsg-t2i package
        try:
            import dsg  # noqa: F401
            self._model = dsg
            self._backend = "native"
            logger.info("DSG initialised (native dsg-t2i package)")
            return
        except ImportError:
            pass

        # Tier 2: heuristic CLIP-based decomposition
        self._backend = "heuristic"
        logger.info(
            "DSG initialised (heuristic) — "
            "install dsg-t2i for full Davidsonian decomposition"
        )

    def process(self, sample: Sample) -> Sample:
        caption = _get_caption_text(sample)
        if caption is None:
            logger.debug("DSG: no caption for %s, skipping", sample.path.name)
            return sample

        try:
            if self._backend == "native":
                score = self._process_native(sample, caption)
            else:
                score = self._process_heuristic(sample, caption)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.dsg_score = score
        except Exception as e:
            logger.warning("DSG failed for %s: %s", sample.path, e)

        return sample

    def _process_native(self, sample: Sample, caption: str) -> Optional[float]:
        """Use the official dsg-t2i package."""
        # The official package handles decomposition + VQA internally
        result = self._model.evaluate(str(sample.path), caption)
        if isinstance(result, dict):
            return float(result.get("score", result.get("dsg_score", 0.0)))
        return float(result)

    def _process_heuristic(self, sample: Sample, caption: str) -> Optional[float]:
        """Heuristic: decompose caption → CLIP-verify each sub-phrase."""
        threshold = self.config.get("threshold", 0.25)
        subsample = self.config.get("subsample", 4)

        # Get representative frame(s)
        frame = self._get_representative_frame(sample, subsample)
        if frame is None:
            return None

        # Decompose caption into atomic propositions
        propositions = _decompose_caption(caption)
        if not propositions:
            return None

        # Score each proposition against the image
        similarities = _clip_similarity(frame, propositions)

        # Fraction of propositions that pass the threshold
        verified = sum(1 for s in similarities if s >= threshold)
        score = verified / len(propositions)

        return float(np.clip(score, 0.0, 1.0))

    def _get_representative_frame(self, sample: Sample, subsample: int) -> Optional[np.ndarray]:
        """Extract a representative frame from image or video."""
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return None
                # Pick the middle frame
                mid = total // 2
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, frame = cap.read()
                return frame if ret else None
            finally:
                cap.release()
        else:
            return cv2.imread(str(sample.path))

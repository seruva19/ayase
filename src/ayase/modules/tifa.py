"""TIFA — Text-to-Image Faithfulness Assessment (ICCV 2023).

Measures how faithfully a generated image matches its text prompt by:
    1. Generating yes/no questions from the caption (rule-based, no LLM)
    2. Answering them via VQA on the generated image
    3. Scoring = fraction of correct answers

Requires ``sample.caption.text`` — skips if no caption is available.
Also checks for a sidecar ``.txt`` file next to the sample.

Tiered backends:
    1. Rule-based question generation + ViLT VQA (dandelin/vilt-b32-finetuned-vqa)
    2. CLIP text-image cosine similarity proxy
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import CaptionMetadata, QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Common colors for question generation
_COLORS = {
    "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "black", "white", "brown", "gray", "grey", "golden", "silver",
}

# Words to skip as nouns (stop words / too generic)
_SKIP_NOUNS = {
    "a", "an", "the", "it", "this", "that", "there", "here",
    "image", "photo", "picture", "scene", "view", "background",
    "something", "thing", "one", "ones", "way", "lot", "kind",
}


def _generate_questions(caption: str, max_questions: int = 8) -> List[Tuple[str, str]]:
    """Generate (question, expected_answer) pairs from a caption.

    Returns simple yes/no questions about objects, colors, and counts
    mentioned in the caption text.  Pure rule-based — no LLM needed.
    """
    questions: List[Tuple[str, str]] = []
    caption_lower = caption.lower()
    words = re.findall(r"[a-z]+", caption_lower)

    # --- Color questions ---
    found_colors = [c for c in _COLORS if c in words]
    for color in found_colors[:2]:
        questions.append((f"Is there something {color} in the image?", "yes"))

    # --- Noun/object questions ---
    # Simple POS-free heuristic: take nouns as words after determiners/adjectives
    # or standalone content words that are likely nouns.
    _NON_NOUNS = {
        "is", "are", "was", "were", "has", "have", "had", "be",
        "with", "from", "into", "onto", "upon", "over", "under",
        "and", "but", "for", "nor", "yet", "not", "can", "will",
        "very", "much", "more", "most", "also", "just", "only",
    }
    _ADJECTIVE_SUFFIXES = ("ing", "ly", "ed", "ful", "ous", "ive", "ish", "able", "ible", "less")
    _COMMON_ADJECTIVES = {
        "big", "small", "large", "tall", "short", "long", "new", "old",
        "good", "bad", "high", "low", "dark", "bright", "beautiful",
        "nice", "pretty", "ugly", "happy", "sad", "fast", "slow",
        "soft", "hard", "warm", "cold", "hot", "cool", "young",
    }

    def _is_likely_noun(w: str) -> bool:
        """Return True if the word is likely a noun (not a verb/adjective)."""
        if w in _NON_NOUNS or w in _COMMON_ADJECTIVES:
            return False
        if w.endswith(_ADJECTIVE_SUFFIXES):
            return False
        return True

    candidate_nouns = []
    for i, w in enumerate(words):
        if w in _SKIP_NOUNS or len(w) < 3:
            continue
        if w in _COLORS:
            continue
        # Words following a/an/the are likely nouns
        if i > 0 and words[i - 1] in ("a", "an", "the"):
            if _is_likely_noun(w):
                candidate_nouns.append(w)
        # Standalone content words that pass the noun filter
        elif _is_likely_noun(w):
            candidate_nouns.append(w)

    seen = set()
    for noun in candidate_nouns:
        if noun in seen:
            continue
        seen.add(noun)
        questions.append((f"Is there a {noun} in the image?", "yes"))
        if len(questions) >= max_questions:
            break

    # --- Count questions ---
    count_pattern = re.findall(r"(\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(\w+)", caption_lower)
    for num, obj in count_pattern[:2]:
        questions.append((f"Are there {num} {obj} in the image?", "yes"))
        if len(questions) >= max_questions:
            break

    return questions[:max_questions]


class TIFAModule(PipelineModule):
    name = "tifa"
    description = "TIFA text-to-image faithfulness via VQA question answering (ICCV 2023)"
    default_config = {
        "vqa_model": "dandelin/vilt-b32-finetuned-vqa",
        "num_questions": 8,
        "subsample": 4,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.vqa_model = self.config.get("vqa_model", "dandelin/vilt-b32-finetuned-vqa")
        self.num_questions = self.config.get("num_questions", 8)
        self.subsample = self.config.get("subsample", 4)
        self._backend = None  # "vilt" | "clip"
        self._ml_available = False
        self._model = None
        self._processor = None
        self._clip_model = None
        self._clip_processor = None
        self._device = "cpu"

    def setup(self):
        if self.test_mode:
            return

        # Tier 1: ViLT VQA
        try:
            import torch
            from transformers import ViltForQuestionAnswering, ViltProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            models_dir = self.config.get("models_dir", "models")
            self._processor = ViltProcessor.from_pretrained(self.vqa_model, cache_dir=models_dir)
            self._model = ViltForQuestionAnswering.from_pretrained(
                self.vqa_model, cache_dir=models_dir
            ).to(self._device)
            self._backend = "vilt"
            self._ml_available = True
            logger.info(f"TIFA: using ViLT VQA backend on {self._device}.")
            return
        except Exception:
            pass

        # Tier 2: CLIP proxy
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            models_dir = self.config.get("models_dir", "models")
            clip_name = "openai/clip-vit-base-patch32"
            self._clip_model = CLIPModel.from_pretrained(clip_name, cache_dir=models_dir).to(
                self._device
            )
            self._clip_processor = CLIPProcessor.from_pretrained(clip_name, cache_dir=models_dir)
            self._backend = "clip"
            self._ml_available = True
            logger.info("TIFA: using CLIP similarity proxy.")
            return
        except Exception:
            pass

        logger.warning("TIFA: no ML backend available.")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        caption_text = self._get_caption(sample)
        if not caption_text:
            return sample

        try:
            if self._backend == "vilt":
                score = self._compute_vilt(sample, caption_text)
            else:
                score = self._compute_clip(sample, caption_text)

            if score is None:
                return sample

            score = float(np.clip(score, 0.0, 1.0))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.tifa_score = score

        except Exception as e:
            logger.warning(f"TIFA failed for {sample.path}: {e}")

        return sample

    # -- Caption extraction -----------------------------------------------------

    def _get_caption(self, sample: Sample) -> Optional[str]:
        if sample.caption and sample.caption.text:
            return sample.caption.text

        # Check sidecar .txt file
        txt_path = sample.path.with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
            if text:
                return text

        return None

    # -- Backend implementations ------------------------------------------------

    def _compute_vilt(self, sample: Sample, caption: str) -> Optional[float]:
        from PIL import Image

        frames = self._load_frames(sample)
        if not frames:
            return None

        questions = _generate_questions(caption, self.num_questions)
        if not questions:
            return None

        correct_total = 0
        count_total = 0

        for frame in frames:
            pil_img = Image.fromarray(frame)
            for question, expected in questions:
                try:
                    encoding = self._processor(pil_img, question, return_tensors="pt").to(
                        self._device
                    )
                    outputs = self._model(**encoding)
                    idx = outputs.logits.argmax(-1).item()
                    answer = self._model.config.id2label[idx].lower()
                    if answer == expected:
                        correct_total += 1
                    count_total += 1
                except Exception:
                    continue

        if count_total == 0:
            return None
        return correct_total / count_total

    def _compute_clip(self, sample: Sample, caption: str) -> Optional[float]:
        import torch
        from PIL import Image

        frames = self._load_frames(sample)
        if not frames:
            return None

        similarities = []
        for frame in frames:
            pil_img = Image.fromarray(frame)
            inputs = self._clip_processor(
                text=[caption], images=pil_img, return_tensors="pt", padding=True
            ).to(self._device)
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
            # Cosine similarity from CLIP logits
            sim = outputs.logits_per_image.item() / 100.0  # normalize to ~0-1
            similarities.append(float(np.clip(sim, 0.0, 1.0)))

        return float(np.mean(similarities)) if similarities else None

    # -- Frame loading ----------------------------------------------------------

    def _load_frames(self, sample: Sample):
        frames = []
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    return frames
                n = min(self.subsample, total)
                indices = np.linspace(0, total - 1, n, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            else:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
        return frames

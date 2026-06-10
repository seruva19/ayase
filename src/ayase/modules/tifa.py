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

import numpy as np

from ayase.image import arrays_to_pil, sample_frames
from ayase.models import CaptionMetadata, QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import (
    cached_clip_image_feature_groups,
    cached_clip_image_features,
    cached_clip_text_features,
    media_state_key,
)

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
        "clip_model": "openai/clip-vit-base-patch32",
        "num_questions": 8,
        "subsample": 4,
    }
    metric_groups = {
        "tifa_score": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.vqa_model = self.config.get("vqa_model", "dandelin/vilt-b32-finetuned-vqa")
        self.clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
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
            from ayase.config import resolve_model_path
            from ayase.runtime import (
                from_pretrained_with_attention,
                resolve_torch_device,
                shared_runtime_resource,
            )

            models_dir = self.config.get("models_dir", "models")
            self._device = resolve_torch_device(self.config.get("device", "auto"))
            resolved = resolve_model_path(self.clip_model_name, models_dir)

            def load_clip():
                model = from_pretrained_with_attention(
                    CLIPModel,
                    resolved,
                    self.config,
                    device=self._device,
                ).to(self._device).eval()
                processor = CLIPProcessor.from_pretrained(resolved)
                return model, processor

            self._clip_model, self._clip_processor = shared_runtime_resource(
                self,
                (
                    "hf_clip",
                    resolved,
                    self._device,
                    str(self.config.get("attention_backend", "auto")),
                    "default",
                ),
                load_clip,
            )
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

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        if not self._ml_available:
            return samples
        if self._backend != "clip":
            return super().process_batch(samples)

        try:
            prepared = []
            image_groups = []
            cache_keys = []
            for sample in samples:
                caption_text = self._get_caption(sample)
                if not caption_text:
                    continue
                frames = self._load_frames(sample)
                if not frames:
                    continue
                prepared.append((sample, caption_text))
                image_groups.append(arrays_to_pil(frames))
                cache_keys.append((self.subsample, media_state_key(sample.path)))

            if not prepared:
                return samples

            feature_groups = cached_clip_image_feature_groups(
                self,
                self._clip_model,
                self._clip_processor,
                image_groups,
                model_key=self.clip_model_name,
                device=self._device,
                cache_keys=cache_keys,
            )
            for (sample, caption_text), image_features in zip(prepared, feature_groups):
                score = self._score_clip_features(caption_text, image_features)
                if score is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.tifa_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("TIFA batch failed: %s", e)
        return samples

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
        frames = self._load_frames(sample)
        if not frames:
            return None

        image_features = cached_clip_image_features(
            self,
            self._clip_model,
            self._clip_processor,
            arrays_to_pil(frames),
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=(self.subsample, media_state_key(sample.path)),
        )
        return self._score_clip_features(caption, image_features)

    def _score_clip_features(self, caption: str, image_features) -> Optional[float]:
        if image_features is None or image_features.size(0) == 0:
            return None

        text_features = cached_clip_text_features(
            self,
            self._clip_model,
            self._clip_processor,
            [caption],
            model_key=self.clip_model_name,
            device=self._device,
            cache_key=("tifa_caption", caption),
        )
        scale = getattr(self._clip_model, "logit_scale", None)
        if scale is not None:
            logits = (image_features @ text_features.T) * scale.exp()
        else:
            logits = image_features @ text_features.T
        similarities = (logits.squeeze(-1) / 100.0).detach().float().cpu().numpy()
        similarities = np.clip(similarities, 0.0, 1.0)
        return float(np.mean(similarities)) if len(similarities) else None

    # -- Frame loading ----------------------------------------------------------

    def _load_frames(self, sample: Sample):
        try:
            return sample_frames(sample.path, max_frames=self.subsample, color="rgb")
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
        return []

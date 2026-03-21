"""Captioning module — generates BLIP captions and computes BLEU score.

Implements EvalCrafter metric #11 (blip_bleu): sample multiple frames, generate
BLIP captions for each, compute BLEU-1 through BLEU-4 against the original
prompt, take max per n-gram level, average the 4 maxes.
"""

import logging
import math
from collections import Counter
from typing import List, Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CaptioningModule(PipelineModule):
    name = "captioning"
    description = "Generates captions using BLIP + computes BLEU score (EvalCrafter blip_bleu)"
    default_config = {
        "model_name": "Salesforce/blip-image-captioning-base",
        "num_frames": 5,  # EvalCrafter samples 5 frames
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "Salesforce/blip-image-captioning-base")
        self.num_frames = self.config.get("num_frames", 5)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch
            from ayase.config import resolve_model_path

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)
            is_blip2 = "blip2" in self.model_name.lower() or "blip-2" in self.model_name.lower()

            if is_blip2:
                from transformers import AutoProcessor, Blip2ForConditionalGeneration

                logger.info(f"Loading BLIP-2 ({self.model_name}) on {self._device}...")
                self._processor = AutoProcessor.from_pretrained(resolved)
                self._model = Blip2ForConditionalGeneration.from_pretrained(
                    resolved, torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                ).to(self._device)
            else:
                from transformers import BlipProcessor, BlipForConditionalGeneration

                logger.info(f"Loading BLIP ({self.model_name}) on {self._device}...")
                self._processor = BlipProcessor.from_pretrained(resolved)
                self._model = BlipForConditionalGeneration.from_pretrained(
                    resolved, use_safetensors=True
                ).to(self._device)

            self._ml_available = True

        except Exception as e:
            logger.warning(f"Failed to setup Captioning: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            import torch
            import cv2
            from PIL import Image
            from ayase.models import CaptionMetadata
            import numpy as np

            frames = self._load_frames(sample)
            if not frames:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            # Generate captions for all sampled frames
            generated_captions: List[str] = []
            with torch.no_grad():
                for frame_bgr in frames:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(frame_rgb)
                    inputs = self._processor(pil_image, return_tensors="pt").to(self._device)
                    out = self._model.generate(**inputs, max_new_tokens=50)
                    text = self._processor.decode(out[0], skip_special_tokens=True)
                    if text:
                        generated_captions.append(text)

            if not generated_captions:
                return sample

            # Use the best (longest) caption as the representative auto-caption
            best_caption = max(generated_captions, key=len)

            if sample.caption is None:
                # No existing caption — set the generated one
                sample.caption = CaptionMetadata(
                    text=best_caption,
                    length=len(best_caption),
                    source_file=None,
                )
            else:
                # Existing caption — compute BLEU score (EvalCrafter blip_bleu)
                reference = sample.caption.text
                bleu = self._compute_blip_bleu(reference, generated_captions)
                sample.quality_metrics.blip_bleu = bleu

                if best_caption.strip().lower() != reference.strip().lower():
                    sample.quality_metrics.auto_caption = best_caption
                    if bleu < 0.1:
                        sample.validation_issues.append(
                            ValidationIssue(
                                severity=ValidationSeverity.WARNING,
                                message=f"Very low caption BLEU ({bleu:.3f}): generated captions diverge from original.",
                                details={
                                    "existing_caption": reference,
                                    "generated_caption": best_caption,
                                    "blip_bleu": bleu,
                                },
                                recommendation="Video content may not match the caption text.",
                            )
                        )

            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Generated Caption: {best_caption}",
                    details={
                        "generated_caption": best_caption,
                        "num_frames_captioned": len(generated_captions),
                    },
                )
            )

        except Exception as e:
            logger.warning(f"Caption generation failed: {e}")

        return sample

    # ------------------------------------------------------------------ #
    #  BLEU computation (EvalCrafter blip_bleu algorithm)                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_blip_bleu(reference: str, hypotheses: List[str]) -> float:
        """Compute blip_bleu following EvalCrafter:

        For each BLEU-n (n=1..4), take the MAX score across all hypotheses.
        Then average the 4 max scores.
        """
        ref_tokens = reference.lower().split()
        if not ref_tokens:
            return 0.0

        max_per_n = []  # one entry per n in [1, 2, 3, 4]
        for n in range(1, 5):
            best = 0.0
            for hyp in hypotheses:
                score = CaptioningModule._bleu_n(ref_tokens, hyp.lower().split(), n)
                if score > best:
                    best = score
            max_per_n.append(best)

        return float(sum(max_per_n) / 4.0)

    @staticmethod
    def _bleu_n(reference: List[str], hypothesis: List[str], n: int) -> float:
        """Compute BLEU-n with brevity penalty for a single reference/hypothesis pair."""
        if len(hypothesis) == 0:
            return 0.0

        # Brevity penalty
        bp = 1.0
        if len(hypothesis) < len(reference):
            bp = math.exp(1.0 - len(reference) / len(hypothesis))

        # Modified n-gram precision for each k from 1..n
        precisions = []
        for k in range(1, n + 1):
            ref_ngrams = CaptioningModule._get_ngrams(reference, k)
            hyp_ngrams = CaptioningModule._get_ngrams(hypothesis, k)
            if not hyp_ngrams:
                return 0.0
            clipped = 0
            for gram, count in hyp_ngrams.items():
                clipped += min(count, ref_ngrams.get(gram, 0))
            precision = clipped / sum(hyp_ngrams.values())
            if precision == 0:
                return 0.0
            precisions.append(math.log(precision))

        # Geometric mean of precisions
        log_avg = sum(precisions) / len(precisions)
        return bp * math.exp(log_avg)

    @staticmethod
    def _get_ngrams(tokens: List[str], n: int) -> Counter:
        """Extract n-gram counts from a token list."""
        return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    # ------------------------------------------------------------------ #
    #  Frame loading                                                      #
    # ------------------------------------------------------------------ #

    def _load_frames(self, sample: Sample) -> List:
        """Load multiple uniformly-spaced frames (EvalCrafter samples 5)."""
        try:
            import cv2
            import numpy as np

            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    return []
                n = min(self.num_frames, total)
                indices = np.linspace(0, total - 1, n, dtype=int)
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                cap.release()
                return frames
            else:
                img = cv2.imread(str(sample.path))
                return [img] if img is not None else []
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
            return []

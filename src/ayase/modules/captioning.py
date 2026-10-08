"""Generate per-frame BLIP captions and compare them with a sample caption.

For each image/video sample, the BLIP-2 model (EvalCrafter uses
``Salesforce/blip2-opt-2.7b``) captions five uniformly sampled frames. If
``sample.caption`` is missing, the longest generated caption is installed as
the sample caption and no BLEU score is set. Otherwise, ``blip_bleu`` is the
[0, 1] average of the best frame-level BLEU-1, BLEU-2, BLEU-3, and BLEU-4
scores against ``sample.caption.text`` — computed with the pycocoevalcap
``Bleu`` scorer on raw text, exactly as in EvalCrafter (higher means more
lexical overlap); ``auto_caption`` stores the longest differing output. When
pycocoevalcap is not installed, ``blip_bleu`` is left unset rather than
approximated by a different BLEU formulation.
Sources: https://github.com/evalcrafter/EvalCrafter and
https://github.com/Salesforce/BLIP
"""

import logging
from typing import List, Optional

from ayase.image import sample_frames
from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CaptioningModule(PipelineModule):
    name = "captioning"
    provenance = {
        "auto_caption": "utility",
        "blip_bleu": "published",
    }
    sources = {
        "blip_bleu": "EvalCrafter BLIP-BLEU — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/Scores_with_CLIP/Scores_with_CLIP.py",
    }
    description = "Generates captions using BLIP-2 + computes BLEU score (EvalCrafter blip_bleu)"
    default_config = {
        "model_name": "Salesforce/blip2-opt-2.7b",
        "num_frames": 5,  # EvalCrafter samples 5 frames
    }
    models = [
        {
            "id": "pycocoevalcap",
            "type": "pip_package",
            "install": "pip install pycocoevalcap",
            "task": "Official BLEU scorer for blip_bleu (required for the score, not for captioning)",
        },
    ]
    metric_groups = {
        "auto_caption": "text",
        "blip_bleu": "alignment",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "Salesforce/blip2-opt-2.7b")
        self.num_frames = self.config.get("num_frames", 5)
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None
        self._bleu_available = False

    def setup(self) -> None:
        try:
            import torch
            from ayase.config import resolve_model_path
            from ayase.runtime import from_pretrained_with_attention, resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(self.model_name, models_dir)
            is_blip2 = "blip2" in self.model_name.lower() or "blip-2" in self.model_name.lower()

            if is_blip2:
                from transformers import AutoProcessor, Blip2ForConditionalGeneration

                logger.info(f"Loading BLIP-2 ({self.model_name}) on {self._device}...")
                self._processor = AutoProcessor.from_pretrained(resolved)
                self._model = from_pretrained_with_attention(
                    Blip2ForConditionalGeneration,
                    resolved,
                    self.config,
                    device=self._device,
                    torch_dtype=torch.float16 if self._device == "cuda" else torch.float32,
                ).to(self._device)
            else:
                from transformers import BlipProcessor, BlipForConditionalGeneration

                logger.info(f"Loading BLIP ({self.model_name}) on {self._device}...")
                self._processor = BlipProcessor.from_pretrained(resolved)
                self._model = from_pretrained_with_attention(
                    BlipForConditionalGeneration,
                    resolved,
                    self.config,
                    device=self._device,
                    use_safetensors=True,
                ).to(self._device)

            self._ml_available = True
            self._backend = "blip2" if is_blip2 else "blip"

            try:
                from pycocoevalcap.bleu.bleu import Bleu  # noqa: F401

                self._bleu_available = True
            except ImportError:
                logger.warning(
                    "pycocoevalcap not installed — blip_bleu disabled "
                    "(install: pip install pycocoevalcap); captioning still works."
                )

        except Exception as e:
            self._backend = "unavailable"
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
                    # EvalCrafter uses the model's default generation config.
                    out = self._model.generate(**inputs)
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
                if bleu is not None:
                    sample.quality_metrics.blip_bleu = bleu

                if best_caption.strip().lower() != reference.strip().lower():
                    sample.quality_metrics.auto_caption = best_caption
                    if bleu is not None and bleu < 0.1:
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

    def _compute_blip_bleu(self, reference: str, hypotheses: List[str]) -> Optional[float]:
        """EvalCrafter blip_bleu via the official pycocoevalcap ``Bleu`` scorer.

        For each BLEU-n (n=1..4) take the MAX score across generated captions,
        then average the four maxima. Text is passed raw — the reference
        implementation does not lowercase or retokenize. Returns ``None`` when
        pycocoevalcap is unavailable instead of substituting a different BLEU.
        """
        if not self._bleu_available:
            return None
        if not reference.strip() or not hypotheses:
            return 0.0

        from pycocoevalcap.bleu.bleu import Bleu

        max_per_n = []  # one entry per n in [1, 2, 3, 4]
        for n in range(1, 5):
            scorer = Bleu(n=n)
            best = 0.0
            for hyp in hypotheses:
                if not hyp.strip():
                    continue
                score, _ = scorer.compute_score(
                    {0: [reference]}, {0: [hyp]}
                )
                # pycocoevalcap returns cumulative BLEU orders: Bleu(n=3)
                # yields [BLEU-1, BLEU-2, BLEU-3].  This loop measures the
                # requested order, so select its final component rather than
                # coercing the component list (or accidentally reusing BLEU-1).
                if isinstance(score, (list, tuple)):
                    if len(score) < n:
                        logger.warning(
                            "pycocoevalcap returned %d BLEU components for n=%d",
                            len(score),
                            n,
                        )
                        return None
                    score = score[n - 1]
                score = float(score)
                if score > best:
                    best = score
            max_per_n.append(best)

        return float(sum(max_per_n) / 4.0)

    # ------------------------------------------------------------------ #
    #  Frame loading                                                      #
    # ------------------------------------------------------------------ #

    def _load_frames(self, sample: Sample) -> List:
        """Load multiple uniformly-spaced frames (EvalCrafter samples 5)."""
        try:
            return sample_frames(sample.path, max_frames=self.num_frames, color="bgr")
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
            return []

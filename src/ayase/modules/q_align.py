"""Q-Align (Quality and Aesthetic Alignment) module.

State-of-the-art unified quality and aesthetic assessment using a
vision-language model (ICML 2024, q-future/one-align).

The model processes an image/frame with a quality-related prompt and
returns a score derived from the probability distribution over quality
level tokens (excellent / good / fair / poor / bad).

qalign_quality   — technical quality score (1-5, higher=better)
qalign_aesthetic — aesthetic quality score (1-5, higher=better)

use ``dtype: float16`` / ``dtype: bfloat16`` to reduce VRAM.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import cv2
import numpy as np

if TYPE_CHECKING:
    import torch

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Quality level tokens -> numeric scores (standard Q-Align mapping)
QUALITY_LEVELS = {
    "excellent": 5.0,
    "good": 4.0,
    "fair": 3.0,
    "poor": 2.0,
    "bad": 1.0,
}


class QAlignModule(PipelineModule):
    name = "q_align"
    description = "Q-Align unified quality + aesthetic assessment (ICML 2024)"
    default_config = {
        "model_name": "q-future/one-align",
        "dtype": "float16",  # float16 | bfloat16 | float32
        "device": "auto",
        "subsample": 8,  # Every Nth video frame
        "max_frames": 16,  # Max frames to score per video
        "warning_threshold": 2.5,  # Warn if score < 2.5 / 5
        "trust_remote_code": True,
        "model_revision": None,
    }
    metric_groups = {
        "qalign_aesthetic": "aesthetic",
        "qalign_quality": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "q-future/one-align")
        self.dtype_str = self.config.get("dtype", "float16")
        self.device_config = self.config.get("device", "auto")
        self.subsample = self.config.get("subsample", 8)
        self.max_frames = self.config.get("max_frames", 16)
        self.warning_threshold = self.config.get("warning_threshold", 2.5)

        self.device = None
        self._ml_available = False
        self._backend = "unavailable"
        self._model = None
        self._processor = None
        self._quality_token_ids = {}  # type: Dict[int, float]

    def _resolve_dtype(self):
        import torch
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        return mapping.get(self.dtype_str, torch.float16)

    def setup(self) -> None:
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            self.device = torch.device(resolve_torch_device(self.device_config))
            dtype = self._resolve_dtype()

            logger.info(
                f"Loading Q-Align (vendored mPLUG-Owl2) from {self.model_name} on "
                f"{self.device} (dtype={dtype}) — this may take a while (~7 GB)..."
            )

            # Import the vendored mPLUG-Owl2 architecture. Importing registers the
            # ``mplug_owl2`` config/model with the transformers Auto* factories, so the
            # weights load with trust_remote_code=False (no remote code download, no
            # icecream dependency) while the checkpoint itself still comes from HF.
            import ayase.vendor.q_align  # noqa: F401
            from ayase.vendor.q_align.modeling_mplug_owl2 import MPLUGOwl2LlamaForCausalLM

            self._model = MPLUGOwl2LlamaForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=False,
                device_map="auto" if self.device.type == "cuda" else None,
            )
            if self.device.type != "cuda":
                self._model = self._model.to(self.device)
            self._model.eval()

            # The model builds its own tokenizer / CLIP image processor /
            # preferential level-token ids in __init__, so the native .score()
            # API is ready to use.
            self._ml_available = True
            self._backend = "qalign"
            logger.info("Q-Align initialised (vendored, native .score())")

        except ImportError as e:
            self._backend = "unavailable"
            logger.warning("Q-Align unavailable (torch/transformers missing): %s", e)
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup Q-Align: {e}")

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_from_logits(self, logits: torch.Tensor) -> Optional[float]:
        """Compute expected quality score from next-token logits.

        Uses softmax over the quality-level tokens (excellent=5 … bad=1)
        and returns the probability-weighted expected value.
        """
        if not self._quality_token_ids:
            return None

        token_ids = list(self._quality_token_ids.keys())
        scores = torch.tensor(
            [self._quality_token_ids[tid] for tid in token_ids],
            dtype=torch.float32,
        )

        quality_logits = logits[token_ids].float()
        probs = torch.softmax(quality_logits, dim=0)

        return float((probs * scores).sum().item())

    def _score_from_text(self, text: str) -> Optional[float]:
        """Fallback: parse a numeric or keyword score from generated text."""
        text_lower = text.lower().strip()

        # Try to extract a number first
        match = re.search(r"(\d+(?:\.\d+)?)", text_lower)
        if match:
            val = float(match.group(1))
            if 1.0 <= val <= 5.0:
                return val

        # Try keyword matching
        for keyword, score in QUALITY_LEVELS.items():
            if keyword in text_lower:
                return score

        return None

    def _assess(self, image, prompt: str) -> Optional[float]:
        """Run the model with a quality prompt and return the score.

        ``image`` is a PIL RGB image (the processor consumes PIL directly, so
        no temporary file round-trip is needed).
        """
        try:
            import torch

            img = image

            # Build input using processor
            inputs = self._processor(
                text=prompt, images=img, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Logit-based scoring (preferred — no generation needed)
            logits = outputs.logits[0, -1, :]
            score = self._score_from_logits(logits)

            if score is not None:
                return score

            # Fallback: generate text and parse
            with torch.no_grad():
                gen_inputs = self._processor(
                    text=prompt, images=img, return_tensors="pt"
                )
                gen_inputs = {k: v.to(self.device) for k, v in gen_inputs.items()
                              if isinstance(v, torch.Tensor)}
                output_ids = self._model.generate(
                    **gen_inputs,
                    max_new_tokens=10,
                    do_sample=False,
                )
                tokenizer = (
                    self._processor.tokenizer
                    if hasattr(self._processor, "tokenizer")
                    else self._processor
                )
                generated = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                return self._score_from_text(generated)

        except Exception as e:
            logger.debug(f"Q-Align assessment failed: {e}")
            return None

    def _pil_score(self, pil_img, task_: str) -> Optional[float]:
        """Native Q-Align score (1-5) for one PIL image on the given task."""
        try:
            import torch
            with torch.no_grad():
                out = self._model.score([pil_img], task_=task_, input_="image")
            return float(out.reshape(-1)[0].item())
        except Exception as e:
            logger.debug("Q-Align .score(%s) failed: %s", task_, e)
            return None

    def score_frames(self, frames_bgr: List[np.ndarray], task_: str = "quality") -> Optional[float]:
        """Native Q-Align *video* score (1-5) over a list of BGR frames.

        Used both here and by the VMBench Motion Smoothness metric, which scores
        Q-Align over sliding windows of frames.
        """
        if not self._ml_available or not frames_bgr:
            return None
        from PIL import Image

        pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
        try:
            import torch
            with torch.no_grad():
                out = self._model.score([pil_frames], task_=task_, input_="video")
            return float(out.reshape(-1)[0].item())
        except Exception as e:
            logger.debug("Q-Align video .score(%s) failed: %s", task_, e)
            return None

    def _score_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """Score a single BGR frame for quality and aesthetics."""
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        return self._pil_score(img, "quality"), self._pil_score(img, "aesthetics")

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if sample.is_video:
                quality, aesthetic = self._process_video(sample.path)
            else:
                quality, aesthetic = self._process_image(sample.path)

            if quality is None and aesthetic is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            if quality is not None:
                sample.quality_metrics.qalign_quality = quality
            if aesthetic is not None:
                sample.quality_metrics.qalign_aesthetic = aesthetic

            # Warn if scores are low
            combined = None
            if quality is not None and aesthetic is not None:
                combined = (quality + aesthetic) / 2.0
            elif quality is not None:
                combined = quality

            if combined is not None and combined < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low Q-Align score: {combined:.2f}/5",
                        details={
                            "qalign_quality": quality,
                            "qalign_aesthetic": aesthetic,
                        },
                        recommendation=(
                            "Q-Align vision-language assessment indicates "
                            "below-threshold quality or aesthetics."
                        ),
                    )
                )

            logger.debug(
                f"Q-Align for {sample.path.name}: "
                f"quality={quality} aesthetic={aesthetic}"
            )

        except Exception as e:
            logger.warning(f"Q-Align failed for {sample.path}: {e}")

        return sample

    def _process_image(
        self, path: Path
    ) -> Tuple[Optional[float], Optional[float]]:
        frame = cv2.imread(str(path))
        if frame is None:
            return None, None
        return self._score_frame(frame)

    def _process_video(
        self, path: Path
    ) -> Tuple[Optional[float], Optional[float]]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return None, None

        quality_scores: List[float] = []
        aesthetic_scores: List[float] = []
        idx = 0
        scored = 0

        try:
            while scored < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if idx % self.subsample == 0:
                    q, a = self._score_frame(frame)
                    if q is not None:
                        quality_scores.append(q)
                    if a is not None:
                        aesthetic_scores.append(a)
                    scored += 1
                idx += 1
        finally:
            cap.release()

        quality = float(np.mean(quality_scores)) if quality_scores else None
        aesthetic = float(np.mean(aesthetic_scores)) if aesthetic_scores else None
        return quality, aesthetic

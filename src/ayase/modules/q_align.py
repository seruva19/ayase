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
import tempfile
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
            from transformers import AutoModelForCausalLM, AutoProcessor

            if self.device_config == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(self.device_config)

            dtype = self._resolve_dtype()

            logger.info(
                f"Loading Q-Align model ({self.model_name}) on {self.device} "
                f"(dtype={dtype}) — this may take a while (~7 GB)..."
            )

            models_dir = self.config.get("models_dir", None)
            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)

            # Load processor (handles image + text tokenisation)
            try:
                self._processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=trc,
                    revision=rev,
                    cache_dir=models_dir,
                )
            except Exception as e:
                logger.warning(f"Q-Align processor failed, trying tokenizer: {e}")
                from transformers import AutoTokenizer
                self._processor = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=trc,
                    revision=rev,
                    cache_dir=models_dir,
                )

            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=trc,
                revision=rev,
                torch_dtype=dtype,
                cache_dir=models_dir,
                device_map="auto" if self.device.type == "cuda" else None,
            )

            if self.device.type != "cuda":
                self._model = self._model.to(self.device)

            self._model.eval()

            # Pre-resolve token IDs for quality level words so we can
            # read logits directly instead of generating text.
            tokenizer = (
                self._processor.tokenizer
                if hasattr(self._processor, "tokenizer")
                else self._processor
            )
            for level, score in QUALITY_LEVELS.items():
                ids = tokenizer.encode(level, add_special_tokens=False)
                if ids:
                    self._quality_token_ids[ids[0]] = score

            self._ml_available = True
            logger.info(f"Q-Align initialised ({len(self._quality_token_ids)} level tokens found)")

        except ImportError:
            logger.warning(
                "transformers not installed or Q-Align model unavailable. "
                "Install with: pip install transformers accelerate"
            )
        except Exception as e:
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

    def _assess(self, image_path: str, prompt: str) -> Optional[float]:
        """Run the model with a quality prompt and return the score."""
        try:
            from PIL import Image

            img = Image.open(image_path).convert("RGB")

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

    def _score_frame(
        self, frame_bgr: np.ndarray
    ) -> Tuple[Optional[float], Optional[float]]:
        """Score a single BGR frame for quality and aesthetics."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
            cv2.imwrite(tmp_path, frame_bgr)

        try:
            quality = self._assess(
                tmp_path,
                "USER: <image>\nRate the quality of this image.\nASSISTANT: The quality is",
            )
            aesthetic = self._assess(
                tmp_path,
                "USER: <image>\nRate the aesthetic quality of this image.\n"
                "ASSISTANT: The aesthetic quality is",
            )
            return quality, aesthetic
        finally:
            Path(tmp_path).unlink(missing_ok=True)

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

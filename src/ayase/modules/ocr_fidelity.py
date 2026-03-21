"""OCR Fidelity — measures whether text requested in the caption appears in video frames.

Based on the EvalCrafter OCR score methodology:
1. Extract expected text from the caption (quoted strings like "Hello", 'World')
2. Run PaddleOCR on sampled video frames to recognize rendered text
3. Compute Normalized Edit Distance (NED) between expected and recognized text
4. Score = (1 - NED) * 100 → higher means text was rendered more accurately

This is fundamentally different from the ``text_detection`` module, which only
measures text *area coverage*.  This module checks text *accuracy* — whether the
video actually renders the words the prompt asked for.

References:
    - EvalCrafter (Liu et al., 2023) — T2V benchmark OCR score
    - PaddleOCR (Baidu, 2020) — open-source OCR engine
"""

import logging
import re
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _extract_quoted_text(caption: str) -> List[str]:
    """Extract text enclosed in quotes from a caption.

    Supports double quotes, single quotes, guillemets, and backticks.
    Returns a list of non-empty extracted strings.
    """
    patterns = [
        r'"([^"]+)"',     # "text"
        r"'([^']+)'",     # 'text'
        r"\u00ab([^\u00bb]+)\u00bb",  # «text»
        r"\u201c([^\u201d]+)\u201d",  # \u201ctext\u201d
        r"`([^`]+)`",     # `text`
    ]
    found: List[str] = []
    for pat in patterns:
        found.extend(re.findall(pat, caption))
    return [t.strip() for t in found if t.strip()]


def _normalized_edit_distance(reference: str, hypothesis: str) -> float:
    """Compute Normalized Edit Distance (NED) between two strings.

    Returns a value in [0, 1] where 0 means identical strings.
    Uses the standard Levenshtein distance normalized by max length.
    """
    if not reference and not hypothesis:
        return 0.0
    try:
        from Levenshtein import distance as lev_distance

        d = lev_distance(reference, hypothesis)
    except ImportError:
        # Fallback: simple DP Levenshtein
        d = _levenshtein_dp(reference, hypothesis)
    return d / max(len(reference), len(hypothesis))


def _character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate (CER) between two strings.

    CER = Levenshtein(ref, hyp) / len(ref).
    Returns a value in [0, ∞) where 0 means identical strings.
    Clamped to [0, 1] for scoring purposes.
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0
    try:
        from Levenshtein import distance as lev_distance

        d = lev_distance(reference, hypothesis)
    except ImportError:
        d = _levenshtein_dp(reference, hypothesis)
    return min(1.0, d / len(reference))


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate (WER) between two strings.

    WER = Levenshtein(ref_words, hyp_words) / len(ref_words).
    Returns a value in [0, ∞) where 0 means identical word sequences.
    Clamped to [0, 1] for scoring purposes.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    try:
        from Levenshtein import distance as lev_distance

        d = lev_distance(ref_words, hyp_words)
    except (ImportError, TypeError):
        d = _levenshtein_dp(ref_words, hyp_words)
    return min(1.0, d / len(ref_words))


def _levenshtein_dp(s, t) -> int:
    """Minimal Levenshtein distance implementation (no external deps). Works on strings or lists."""
    m, n = len(s), len(t)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]


class OCRFidelityModule(PipelineModule):
    name = "ocr_fidelity"
    description = "Checks whether text requested in the caption actually appears in video frames (EvalCrafter OCR)"
    default_config = {
        "num_frames": 8,
        "lang": "en",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.num_frames = self.config.get("num_frames", 8)
        self.lang = self.config.get("lang", "en")
        self._ocr = None
        self._ocr_available = False

    def setup(self) -> None:
        try:
            # PaddleOCR 2.x uses np.sctypes which was removed in NumPy 2.0
            import numpy as _np
            if not hasattr(_np, "sctypes"):
                _np.sctypes = {
                    "int": [_np.int8, _np.int16, _np.int32, _np.int64],
                    "uint": [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
                    "float": [_np.float16, _np.float32, _np.float64],
                    "complex": [_np.complex64, _np.complex128],
                    "others": [bool, object, bytes, str, _np.void],
                }

            from paddleocr import PaddleOCR
            import paddleocr

            logger.info("Loading PaddleOCR for OCR Fidelity...")
            # PaddleOCR >=3.x removed the show_log argument
            ocr_version = getattr(paddleocr, "__version__", "0.0.0")
            major = int(ocr_version.split(".")[0])
            if major >= 3:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
            else:
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang, show_log=False)
            self._ocr_available = True
        except ImportError:
            logger.warning("PaddleOCR not installed. OCR Fidelity disabled.")
        except Exception as e:
            logger.error(f"Failed to init PaddleOCR: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ocr_available:
            return sample

        # Prefer explicit expected_text from config (set by downstream caller)
        explicit = self.config.get("expected_text")
        if explicit:
            expected_texts = [explicit] if isinstance(explicit, str) else list(explicit)
        else:
            # Fallback: extract quoted text from caption
            caption_text = self._get_caption(sample)
            if not caption_text:
                return sample
            expected_texts = _extract_quoted_text(caption_text)

        if not expected_texts:
            return sample

        expected = " ".join(expected_texts).lower()

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            # Run OCR on each frame, collect all recognized text
            all_recognized: List[str] = []
            for frame in frames:
                result = self._ocr.ocr(frame, cls=True)
                if result and result[0]:
                    for line in result[0]:
                        txt = line[1][0]
                        if txt:
                            all_recognized.append(txt)

            recognized = " ".join(all_recognized).lower()

            ned = _normalized_edit_distance(expected, recognized)
            cer = _character_error_rate(expected, recognized)
            wer = _word_error_rate(expected, recognized)
            score = max(0.0, (1.0 - ned)) * 100.0

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.ocr_fidelity = round(score, 2)
            sample.quality_metrics.ocr_score = round(score, 2)
            sample.quality_metrics.ocr_cer = round(cer, 4)
            sample.quality_metrics.ocr_wer = round(wer, 4)

            if score < 30.0:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low OCR fidelity: {score:.1f}% — text in video doesn't match caption",
                        details={
                            "expected_text": expected,
                            "recognized_text": recognized[:200],
                            "ocr_fidelity": score,
                            "cer": round(cer, 4),
                            "wer": round(wer, 4),
                        },
                        recommendation="Video may not be rendering the requested text correctly.",
                    )
                )

        except Exception as e:
            logger.warning(f"OCR Fidelity failed for {sample.path}: {e}")

        return sample

    def _get_caption(self, sample: Sample) -> Optional[str]:
        if sample.caption and sample.caption.text:
            return sample.caption.text
        txt_path = sample.path.with_suffix(".txt")
        if txt_path.exists():
            try:
                return txt_path.read_text(encoding="utf-8").strip()
            except Exception:
                pass
        return None

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        frames: List[np.ndarray] = []
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    return frames
                n = min(self.num_frames, total)
                indices = np.linspace(0, total - 1, n, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                cap.release()
            else:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    frames.append(img)
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
        return frames


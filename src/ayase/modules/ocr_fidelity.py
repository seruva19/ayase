"""OCR Fidelity — EvalCrafter OCR score for text rendered in video frames.

Protocol (EvalCrafter ``metrics/ocr_score.py``):
1. Expected text comes from the caller (``expected_text`` config) or is
   extracted from quoted caption strings — the benchmark takes it from its
   metadata.
2. PaddleOCR runs on **every** frame; recognized lines are concatenated raw
   (no separator, no case folding).
3. Per frame: NED = lev(gt, pred)/max(len(pred), len(gt)) (ICDAR 2019),
   CER = CharacTER's shift-aware character edit rate,
   WER = word-level lev/len(gt_words). Missing OCR text is scored as a
   deletion of the expected text, rather than a zero-error observation.
4. ``ocr_fidelity`` == ``ocr_score`` = mean over frames of (NED+CER+WER)/3 —
   this is an *error* measure: **lower is better**. ``ocr_cer``/``ocr_wer``
   report the two component means.

This is fundamentally different from the ``text_detection`` module, which only
measures text *area coverage*. This module checks text *accuracy* — whether the
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


def _character_error_rate(hypothesis: str, reference: str) -> float:
    """EvalCrafter CER — ``cer.calculate_cer(pred_words, gt_words)``.

    The ``cer`` package implements CharacTER: a word-shift pass on the
    hypothesis followed by character-level Levenshtein between the joined
    strings. When the package is unavailable the fallback computes ordinary
    character edit distance normalized by the reference character length,
    clipped to 1.0, without the word-shift optimization.
    """
    if not hypothesis or not reference:
        return 0.0 if hypothesis == reference else 1.0
    try:
        from cer import calculate_cer

        return float(calculate_cer(hypothesis.split(" "), reference.split(" ")))
    except ImportError:
        pass
    hyp_chars = " ".join(hypothesis.split(" "))
    ref_chars = " ".join(reference.split(" "))
    return min(1.0, _levenshtein_dp(hyp_chars, ref_chars) / len(ref_chars))


def _word_error_rate(reference: str, hypothesis: str) -> float:
    """EvalCrafter WER — ``fastwer.score_sent(pred, gt)/100``.

    Word-level Levenshtein normalized by the *reference* word count, returned
    as a fraction. No clipping.
    """
    try:
        import fastwer

        return float(fastwer.score_sent(hypothesis, reference) / 100)
    except ImportError:
        pass
    ref_words = reference.split(" ")
    hyp_words = hypothesis.split(" ")
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return _levenshtein_dp(ref_words, hyp_words) / len(ref_words)


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
    implementation_status = "implemented"
    implementation_notes = "PaddleOCR text is compared with the reference; empty recognition counts as deletion error and the fallback CER uses reference length. See field deviations for the adapted OCR protocol."
    implementation_review = "Source inspection; confidence medium; pretrained inference not validated."
    provenance = {"*": "published", "ocr_fidelity": "adapted", "ocr_score": "adapted", "ocr_cer": "adapted"}
    sources = {
        "ocr_cer": "EvalCrafter OCR-Score (Liu et al., CVPR 2024) — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/ocr_score.py",
        "ocr_fidelity": "EvalCrafter OCR-Score (Liu et al., CVPR 2024) — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/ocr_score.py",
        "ocr_score": "EvalCrafter OCR-Score (Liu et al., CVPR 2024) — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/ocr_score.py",
        "ocr_wer": "EvalCrafter OCR-Score (Liu et al., CVPR 2024) — https://github.com/evalcrafter/EvalCrafter/blob/master/metrics/ocr_score.py",
    }
    description = "EvalCrafter OCR score — text rendering accuracy vs expected text (error measure, lower=better)"
    default_config = {
        "num_frames": 0,
        "lang": "en",
        "text_recognition_model_name": None,
    }
    metric_groups = {
        "ocr_cer": "text",
        "ocr_fidelity": "text",
        "ocr_score": "text",
        "ocr_wer": "text",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.num_frames = self.config.get("num_frames", 0)
        self.lang = self.config.get("lang", "en")
        self.text_recognition_model_name = self.config.get("text_recognition_model_name")
        self._ocr = None
        self._ocr_api = None
        self._ocr_available = False
        self._backend = None

    def setup(self) -> None:
        try:
            import os
            os.environ.setdefault("FLAGS_use_mkldnn", "0")

            from ayase._compat import ensure_paddle_gpu
            ensure_paddle_gpu()

            from paddleocr import PaddleOCR

            logger.info("Loading PaddleOCR for OCR Fidelity...")
            if hasattr(PaddleOCR, "predict"):
                # PaddleOCR 3.x pipeline API
                kw = {}
                if self.text_recognition_model_name:
                    kw["text_recognition_model_name"] = self.text_recognition_model_name
                else:
                    kw["lang"] = self.lang
                self._ocr = PaddleOCR(**kw)
                self._ocr_api = "v3"
            else:
                # PaddleOCR 2.x API — the version EvalCrafter runs upstream
                self._ocr = PaddleOCR(use_angle_cls=True, lang=self.lang)
                self._ocr_api = "v2"
            self._ocr_available = True
            self._backend = "paddleocr"
        except ImportError:
            self._backend = "unavailable"
            logger.warning("PaddleOCR not installed. OCR Fidelity disabled.")
        except Exception as e:
            self._backend = "unavailable"
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

        # EvalCrafter keeps the ground-truth text as-is (no case folding).
        expected = " ".join(expected_texts)

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            neds: List[float] = []
            cers: List[float] = []
            wers: List[float] = []
            last_recognized = ""
            for frame in frames:
                frame_texts: List[str] = []
                if self._ocr_api == "v3":
                    result = self._ocr.predict(frame)
                    if result and result[0]:
                        texts = result[0].get("rec_texts") or []
                        frame_texts.extend(t for t in texts if t)
                else:
                    result = self._ocr.ocr(frame, cls=True)
                    if result and result[0]:
                        for line in result[0]:
                            if line and line[1] and line[1][0]:
                                frame_texts.append(line[1][0])
                # EvalCrafter concatenates recognized lines raw — no
                # separator, no case folding.
                recognized = "".join(frame_texts)
                last_recognized = recognized
                neds.append(_normalized_edit_distance(expected, recognized))
                cers.append(_character_error_rate(recognized, expected))
                wers.append(_word_error_rate(expected, recognized))

            cer = float(np.mean(cers)) if cers else None
            wer = float(np.mean(wers)) if wers else None
            score = float(np.mean([(n + c + w) / 3.0 for n, c, w in zip(neds, cers, wers)]))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.ocr_fidelity = score
            sample.quality_metrics.ocr_score = score
            sample.quality_metrics.ocr_cer = cer
            sample.quality_metrics.ocr_wer = wer

            if score > 0.5:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low OCR fidelity: error score {score:.2f} — text in video doesn't match caption",
                        details={
                            "expected_text": expected,
                            "recognized_text": last_recognized[:200],
                            "ocr_fidelity": score,
                            "cer": cer,
                            "wer": wer,
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
                # num_frames <= 0 follows the EvalCrafter protocol: every frame.
                n = total if self.num_frames <= 0 else min(self.num_frames, total)
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
    deviations = {"ocr_fidelity": "Current root-corrected loop scores empty recognized strings rather than the author empty-detection zero-error special case; fallback _character_error_rate computes reference-normalized clipped ordinary character edit distance without published CharacTER word-shift optimization (lines76-99). Composite averages this adapted CER with NED/WER.", "ocr_score": "Current root-corrected loop scores empty recognized strings rather than the author empty-detection zero-error special case; fallback _character_error_rate computes reference-normalized clipped ordinary character edit distance without published CharacTER word-shift optimization (lines76-99). Composite averages this adapted CER with NED/WER.", "ocr_cer": "Current root-corrected loop scores empty recognized strings rather than the author empty-detection zero-error special case; fallback _character_error_rate computes reference-normalized clipped ordinary character edit distance without published CharacTER word-shift optimization (lines76-99). Composite averages this adapted CER with NED/WER."}


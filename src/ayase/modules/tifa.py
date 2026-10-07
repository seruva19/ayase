"""TIFA — Text-to-Image Faithfulness Assessment (Hu et al., ICCV 2023).

Official pipeline, vendored from https://github.com/Yushi-Hu/tifa:
    1. Multiple-choice questions generated from the caption — the released
       fine-tuned LLaMA-2 question generator
       (``tifa-benchmark/llama2_tifa_question_generation``) by default, or the
       OpenAI ``gpt-3.5-turbo`` prompt when ``question_generator="openai"``.
    2. Question filtering through the official UnifiedQA model
       (``allenai/unifiedqa-v2-t5-large-1363200``).
    3. Multiple-choice VQA on the image — official mPLUG-large via modelscope
       (``damo/mplug_visual-question-answering_coco_large_en``); the free-form
       answer is snapped to the nearest choice by the official SBERT matcher.
    4. ``tifa_score`` = mean over questions of ``mc_answer == answer``.

Requires ``sample.caption.text`` — skips if no caption is available.
TIFA is defined for images; for video inputs Ayase reports the mean of the
per-frame official scores over uniformly sampled frames (own extension).
"""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TIFAModule(PipelineModule):
    name = "tifa"
    provenance = "adapted"
    sources = {
        "tifa_score": "TIFA, Hu et al. ICCV 2023 — https://github.com/Yushi-Hu/tifa",
    }
    deviations = {
        "tifa_score": "TIFA is defined for images; for video it is the mean of the official tifa_score over uniformly sampled frames",
    }
    description = "TIFA text-to-image faithfulness via LLM-generated MC questions + UnifiedQA filter + VQA (ICCV 2023)"
    default_config = {
        "question_generator": "llama2",  # "llama2" | "openai"
        "vqa_model": "mplug-large",
        "subsample": 4,
        "filter_questions": True,
    }
    metric_groups = {
        "tifa_score": "alignment",
    }
    models = [
        {
            "id": "tifa-benchmark/llama2_tifa_question_generation",
            "type": "huggingface",
            "task": "TIFA question generation (fine-tuned LLaMA-2)",
        },
        {
            "id": "allenai/unifiedqa-v2-t5-large-1363200",
            "type": "huggingface",
            "task": "TIFA question filter",
        },
        {
            "id": "damo/mplug_visual-question-answering_coco_large_en",
            "type": "other",
            "task": "mPLUG-large VQA via modelscope",
        },
        {
            "id": "sentence-transformers/all-mpnet-base-v2",
            "type": "huggingface",
            "task": "SBERT multiple-choice answer matching",
        },
        {
            "id": "word2number",
            "type": "pip_package",
            "install": "pip install word2number",
            "task": "TIFA question filter numeric normalization",
        },
        {
            "id": "modelscope",
            "type": "pip_package",
            "install": "pip install modelscope",
            "task": "mPLUG VQA runtime",
        },
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._ml_available = False
        self._vqa = None
        self._qa_filter = None
        self._qg_pipeline = None
        self._tifa_score_single = None
        self._get_llama2_qas = None
        self._get_gpt_qas = None
        self._filter_qas = None

    def setup(self):
        if self.test_mode:
            return

        try:
            from ayase.third_party.tifascore import (
                UnifiedQAModel,
                VQAModel,
                filter_question_and_answers,
                get_llama2_pipeline,
                get_llama2_question_and_answers,
                tifa_score_single,
            )

            self._tifa_score_single = tifa_score_single
            self._get_llama2_qas = get_llama2_question_and_answers
            self._filter_qas = filter_question_and_answers
            try:
                from ayase.third_party.tifascore import get_question_and_answers

                self._get_gpt_qas = get_question_and_answers
            except ImportError:
                self._get_gpt_qas = None

            vqa_name = self.config.get("vqa_model", "mplug-large")
            self._vqa = VQAModel(vqa_name)
            self._qa_filter = (
                UnifiedQAModel() if self.config.get("filter_questions", True) else None
            )

            generator = self.config.get("question_generator", "llama2")
            if generator == "llama2":
                self._qg_pipeline = get_llama2_pipeline()
            elif generator == "openai" and self._get_gpt_qas is None:
                logger.warning("openai question generation unavailable; TIFA disabled")
                return

            self._backend = "official"
            self._ml_available = True
            logger.info("TIFA: official pipeline (vqa=%s, qg=%s).", vqa_name, generator)
        except ImportError as e:
            self._backend = "unavailable"
            logger.info("TIFA dependencies not installed; metric unavailable (%s).", e)
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("TIFA backend unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        caption_text = self._get_caption(sample)
        if not caption_text:
            return sample

        try:
            score = self._compute_tifa(sample, caption_text)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.tifa_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning(f"TIFA failed for {sample.path}: {e}")

        return sample

    # -- Caption extraction -----------------------------------------------------

    def _get_caption(self, sample: Sample) -> Optional[str]:
        if sample.caption and sample.caption.text:
            return sample.caption.text

        txt_path = sample.path.with_suffix(".txt")
        if txt_path.exists():
            text = txt_path.read_text(encoding="utf-8").strip()
            if text:
                return text

        return None

    # -- Official pipeline -------------------------------------------------------

    def _question_answers(self, caption: str) -> Optional[list]:
        """Generate + filter question/answer pairs via the official models."""
        generator = self.config.get("question_generator", "llama2")
        if generator == "openai":
            qas = self._get_gpt_qas(caption) if self._get_gpt_qas else None
        else:
            qas = self._get_llama2_qas(self._qg_pipeline, caption)
        if not qas:
            return None
        if self._qa_filter is not None:
            qas = self._filter_qas(self._qa_filter, qas)
        return qas or None

    def _compute_tifa(self, sample: Sample, caption: str) -> Optional[float]:
        qas = self._question_answers(caption)
        if not qas:
            return None

        if not sample.is_video:
            result = self._tifa_score_single(self._vqa, qas, str(sample.path))
            score = result.get("tifa_score")
            return float(score) if score is not None else None

        # Video extension: mean of the official per-image score over frames.
        frames = self._load_frames(sample)
        if not frames:
            return None

        scores = []
        for frame in frames:
            path = self._write_temp_frame(frame)
            if path is None:
                continue
            try:
                result = self._tifa_score_single(self._vqa, qas, str(path))
                if result.get("tifa_score") is not None:
                    scores.append(float(result["tifa_score"]))
            finally:
                path.unlink(missing_ok=True)
        return float(np.mean(scores)) if scores else None

    @staticmethod
    def _write_temp_frame(frame: np.ndarray) -> Optional[Path]:
        try:
            import cv2

            fd, tmp = tempfile.mkstemp(suffix=".png")
            try:
                import os

                os.close(fd)
            except OSError:
                pass
            cv2.imwrite(tmp, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            return Path(tmp)
        except Exception:
            return None

    def _load_frames(self, sample: Sample):
        try:
            return sample_frames(
                sample.path, max_frames=self.config.get("subsample", 4), color="rgb"
            )
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
        return []

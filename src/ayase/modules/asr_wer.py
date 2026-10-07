"""Compute per-sample word error rate between speech and expected text.

Expected text is selected in this order: ``expected_text`` config (string or
list), ``sample.caption.text``, then a same-stem ``.txt`` file. The hypothesis
comes from explicit ``transcript`` config, ``transcript_path``/``.asr.txt``, or
shared faster-whisper/OpenAI Whisper ASR (default model ``large-v3``). The
optional ``language`` config is passed to ASR; otherwise the backend detects it.
WER is word-level Levenshtein distance over Whisper-normalized text
(``whisper.normalizers`` — EnglishTextNormalizer for English,
BasicTextNormalizer otherwise), unbounded and better when lower. It is not
aggregated across the dataset.
Sources: https://github.com/SYSTRAN/faster-whisper and https://github.com/openai/whisper
"""

import logging

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.modules.asr_cer import _expected_text
from ayase.modules.asr_transcribe import _asr_normalizer, transcribe_sample, word_error_rate

logger = logging.getLogger(__name__)


class ASRWERModule(PipelineModule):
    name = "asr_wer"
    provenance = "published"
    sources = {
        "asr_wer": "WER (word-level Levenshtein / reference length), Whisper eval normalization — https://github.com/openai/whisper",
    }
    description = "ASR word error rate against expected speech text"
    default_config = {
        "model_name": "large-v3",
        "device": "auto",
        "language": None,
        "expected_text": None,
        "transcript": None,
    }
    metric_info = {
        "asr_wer": "ASR word error rate versus expected text (unbounded, lower=better)",
    }
    metric_groups = {
        "asr_wer": "audio",
    }

    def process(self, sample: Sample) -> Sample:
        expected = _expected_text(sample, self.config)
        if not expected:
            return sample
        try:
            transcript = transcribe_sample(sample.path, self.config)
            if not transcript:
                return sample
            wer = word_error_rate(
                expected, transcript, _asr_normalizer(self.config.get("language"))
            )
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.asr_wer = round(float(wer), 4)
        except Exception as e:
            logger.warning("ASR WER failed for %s: %s", sample.path, e)
        return sample

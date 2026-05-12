"""ASR word error rate for speech/text fidelity.

Uses the shared ASR transcript cache and compares recognized speech against
expected text from config, sample caption, or sidecar text. Lower WER is better.
"""

import logging

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.modules.asr_cer import _expected_text
from ayase.modules.asr_transcribe import transcribe_sample, word_error_rate

logger = logging.getLogger(__name__)


class ASRWERModule(PipelineModule):
    name = "asr_wer"
    description = "ASR word error rate against expected speech text"
    default_config = {
        "model_name": "large-v3",
        "device": "auto",
        "language": None,
        "expected_text": None,
        "transcript": None,
    }
    metric_info = {
        "asr_wer": "ASR word error rate versus expected text (0-1, lower=better)",
    }

    def process(self, sample: Sample) -> Sample:
        expected = _expected_text(sample, self.config)
        if not expected:
            return sample
        try:
            transcript = transcribe_sample(sample.path, self.config)
            if not transcript:
                return sample
            wer = word_error_rate(expected, transcript)
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.asr_wer = round(float(wer), 4)
        except Exception as e:
            logger.warning("ASR WER failed for %s: %s", sample.path, e)
        return sample

"""ASR character error rate for speech/text fidelity.

Compares a shared cached Whisper transcript against expected text from module
config, sample caption, or sidecar text. Lower CER is better.
"""

import logging
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.modules.asr_transcribe import char_error_rate, transcribe_sample

logger = logging.getLogger(__name__)


class ASRCERModule(PipelineModule):
    name = "asr_cer"
    description = "ASR character error rate against expected speech text"
    default_config = {
        "model_name": "large-v3",
        "device": "auto",
        "language": None,
        "expected_text": None,
        "transcript": None,
    }
    metric_info = {
        "asr_cer": "ASR character error rate versus expected text (0-1, lower=better)",
    }
    metric_groups = {
        "asr_cer": "audio",
    }

    def process(self, sample: Sample) -> Sample:
        expected = _expected_text(sample, self.config)
        if not expected:
            return sample
        try:
            transcript = transcribe_sample(sample.path, self.config)
            if not transcript:
                return sample
            cer = char_error_rate(expected, transcript)
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.asr_cer = round(float(cer), 4)
        except Exception as e:
            logger.warning("ASR CER failed for %s: %s", sample.path, e)
        return sample


def _expected_text(sample: Sample, config: dict) -> Optional[str]:
    explicit = config.get("expected_text")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if isinstance(explicit, list):
        text = " ".join(str(part) for part in explicit if str(part).strip()).strip()
        if text:
            return text
    if sample.caption and sample.caption.text:
        return sample.caption.text
    sidecar = sample.path.with_suffix(".txt")
    try:
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
    except Exception:
        logger.debug("Failed to read expected-text sidecar %s", sidecar)
    return None

"""Declare TTSDS2 as an unavailable dataset-level TTS distribution metric.

Published TTSDS2 compares synthetic speech with separate real-speech and noise
datasets across feature distributions. The official ``ttsds`` package exposes
``BenchmarkSuite`` over datasets; it has no supported single-file ``score`` or
``evaluate`` API. Ayase's per-sample pipeline therefore emits no value.

TTSDS2 scores are reported on a 0-100 scale, higher is better.
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class TTSDS2Module(PipelineModule):
    name = "tts_system_dist"
    provenance = "published"
    requires_external_backend = True
    sources = {
        "tts_system_dist_score": "TTSDS2, Minixhofer et al. 2025 — https://arxiv.org/abs/2506.19441",
    }
    description = "TTSDS2 dataset distribution score (external backend required)"
    default_config = {"enabled": False}
    models = [{
        "id": "ttsds", "type": "pip_package", "install": "pip install ttsds",
        "task": "Official dataset-level TTSDS2 BenchmarkSuite",
        "notes": "Requires synthetic, real-reference, and noise datasets; not a per-file scorer",
    }]
    metric_info = {
        "tts_system_dist_score": "TTSDS2 aggregate distribution score (0-100, higher=better)",
    }
    metric_groups = {"tts_system_dist_score": "audio"}

    def __init__(self, config=None):
        super().__init__(config)
        self.enabled = self.config.get("enabled", False)
        self._backend = None

    def setup(self) -> None:
        if self.enabled:
            logger.warning(
                "TTSDS2 is unavailable in Ayase's per-sample pipeline: the official "
                "ttsds.BenchmarkSuite requires synthetic, reference, and noise datasets"
            )

    def process(self, sample: Sample) -> Sample:
        return sample

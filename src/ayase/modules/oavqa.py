"""OAVQA -- Omnidirectional Audio-Visual QA (2024).

Quality assessment for omnidirectional (360-degree) content with both audio and
visual components, fusing trained audio and visual streams for a final quality
prediction.

Ayase does not ship (and cannot currently locate a loadable checkpoint for) the
real OAVQA weights. Per the project's no-heuristic policy a named-metric field
must be produced by the real model or left ``None`` -- it must never be
fabricated from an ImageNet ResNet + MelSpectrogram fed through
randomly-initialised, untrained fusion/quality heads (which is all the previous
implementation did). Until a genuine OAVQA checkpoint is wired in,
``oavqa_score`` is left unset.

oavqa_score -- higher = better quality (0-1); real model only
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class OAVQAModule(PipelineModule):
    name = "oavqa"
    provisional = True  # no turnkey real backend in a standard install
    description = "OAVQA omnidirectional audio-visual QA (2024; real model only, disabled if unavailable)"
    default_config = {
        "subsample": 8,
        "n_mels": 64,
        "audio_sr": 16000,
    }
    metric_groups = {
        "oavqa_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.n_mels = self.config.get("n_mels", 64)
        self.audio_sr = self.config.get("audio_sr", 16000)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        # No public, loadable OAVQA checkpoint is wired in. The previous
        # implementation ran an ImageNet ResNet + MelSpectrogram through
        # untrained, random-init fusion/quality heads, so its scores were
        # meaningless. Disable rather than fabricate.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "OAVQA: no real OAVQA model/weights available; module disabled "
            "(oavqa_score left unset). Wire a genuine OAVQA checkpoint to enable."
        )

    def process(self, sample: Sample) -> Sample:
        # Real backend unavailable -> leave oavqa_score None (graceful).
        return sample

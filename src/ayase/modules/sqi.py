"""SQI — Streaming Quality Index (Duanmu et al., 2016).

SQI predicts continuous streaming Quality-of-Experience by combining
presentation quality with the effect of rebuffering/stalling events observed
during a playback session. Those stalling events are a property of a streaming
*session*, not of a stored media file, so SQI cannot be computed from a file on
disk.

A previous revision fabricated ``sqi_score`` from a weighted sum of
resolution/bitrate/fps metadata with ``stalling_factor = 1.0`` assumed. That is
a static-metadata heuristic, not SQI, so it has been removed. Without streaming
session telemetry the metric is reported as unavailable.

REVIVAL NOTES (provisional — no turnkey backend)
Metric: SQI streaming QoE index (Duanmu et al., JSTSP 2016).
Category: IMPOSSIBLE.
Why provisional: Combines FR-VQA with live STALL / REBUFFERING telemetry from a streaming session;
  those inputs do not exist for a stored file.
To revive: Not revivable here -- the required streaming session telemetry cannot be derived from a
  media file on disk. Permanent; keep provisional or delete.
Source: SQI, Duanmu et al., JSTSP 2016.
"""
import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SQIModule(PipelineModule):
    name = "sqi"
    provisional = True  # no turnkey real backend in a standard install
    description = "SQI streaming quality index (2016)"
    default_config = {}
    metric_groups = {
        "sqi_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        self._backend = "unavailable"
        logger.info(
            "SQI: requires streaming-session telemetry (stalling/rebuffering "
            "events) that is not derivable from a stored file; metric reported "
            "as unavailable."
        )

    def process(self, sample: Sample) -> Sample:
        # SQI needs playback stalling data absent from a static file -> skip
        # instead of emitting a metadata-derived proxy.
        return sample

"""MD-VQA — Multi-Dimensional Quality Assessment for UGC Live Videos.

CVPR 2023 — evaluates semantic, distortion, and motion aspects separately for
UGC live streaming videos using trained CLIP/ResNet feature heads.

GitHub: https://github.com/zzc-1998/MD-VQA

Ayase does not ship (and cannot currently locate a loadable checkpoint for) the
real MD-VQA weights. Per the project's no-heuristic policy a named-metric field
must be produced by the real model or left ``None`` — it must never be
fabricated from an ImageNet/CLIP backbone bolted to randomly-initialised,
untrained quality heads (which is all the previous implementation did). Until a
genuine MD-VQA checkpoint is wired in, the outputs are left unset.

mdvqa_semantic — semantic content quality (higher = better, 0-1); real model only
mdvqa_distortion — distortion quality (higher = better, 0-1); real model only
mdvqa_motion — motion quality (higher = better, 0-1); real model only
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MDVQAModule(PipelineModule):
    name = "mdvqa"
    description = "MD-VQA multi-dimensional UGC live VQA (CVPR 2023; real model only, disabled if unavailable)"
    default_config = {
        "subsample": 8,
        "frame_size": 224,
    }
    metric_groups = {
        "mdvqa_distortion": "nr_quality",
        "mdvqa_motion": "nr_quality",
        "mdvqa_semantic": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.frame_size = self.config.get("frame_size", 224)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        # No public, loadable MD-VQA checkpoint is wired in. The previous
        # implementation ran pretrained backbones through untrained, random-init
        # semantic/distortion/motion heads, so its scores were meaningless.
        # Disable rather than fabricate.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "MD-VQA: no real MD-VQA model/weights available; module disabled "
            "(mdvqa_semantic/distortion/motion left unset). Wire a genuine "
            "MD-VQA checkpoint to enable."
        )

    def process(self, sample: Sample) -> Sample:
        # Real backend unavailable -> leave mdvqa_* metrics None (graceful).
        return sample

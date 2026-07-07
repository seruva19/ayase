"""MM-PCQA -- Multi-Modal Point Cloud QA (IJCAI 2023).

Multi-modal quality assessment for point cloud content, combining rendered 2D
projection features with 3D patch features through a trained fusion/regression
network.

Ayase does not ship (and cannot currently locate a loadable checkpoint for) the
real MM-PCQA weights. Per the project's no-heuristic policy a named-metric field
must be produced by the real model or left ``None`` -- it must never be
fabricated from an ImageNet ResNet backbone bolted to randomly-initialised,
untrained quality/attention heads (which is all the previous implementation
did). Until a genuine MM-PCQA checkpoint is wired in, ``mm_pcqa_score`` is left
unset.

mm_pcqa_score -- higher = better quality (0-1); real model only

REVIVAL NOTES (provisional -- no turnkey backend)
Metric: MM-PCQA (IJCAI 2023).
Category: EXTERNAL.
Why provisional: Code is complete but needs an external weight AND point-cloud (.ply) input, which
  ayase (video/image validator) does not feed.
To revive: Mirror WPC.pth (Baidu pan.baidu.com/s/1SuDsQxSRGJ5jePjhTPatHQ, code `pcqa`; or OneDrive
  1drv.ms/f/s!AjaDoj_-yWggygWzjplEICwa2G9k) AND add .ply input plumbing. Low fit for this project.
Source: MM-PCQA, IJCAI 2023.
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MMPCQAModule(PipelineModule):
    name = "mm_pcqa"
    provisional = True  # no turnkey real backend in a standard install
    description = "MM-PCQA multi-modal point cloud QA (IJCAI 2023; real model only, disabled if unavailable)"
    default_config = {
        "n_views": 6,
        "render_size": 224,
    }
    metric_groups = {
        "mm_pcqa_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.n_views = self.config.get("n_views", 6)
        self.render_size = self.config.get("render_size", 224)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        # No public, loadable MM-PCQA checkpoint is wired in. The previous
        # implementation ran an ImageNet ResNet over rendered views through
        # untrained, random-init quality/attention heads, so its scores were
        # meaningless. Disable rather than fabricate.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "MM-PCQA: no real MM-PCQA model/weights available; module disabled "
            "(mm_pcqa_score left unset). Wire a genuine MM-PCQA checkpoint to enable."
        )

    def process(self, sample: Sample) -> Sample:
        # Real backend unavailable -> leave mm_pcqa_score None (graceful).
        return sample

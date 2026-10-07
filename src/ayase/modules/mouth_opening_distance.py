"""MOD — Mouth Opening Distance (external backend).

DiffPoseTalk (Sun et al., CVPR 2024, arXiv:2310.00434) evaluates lip motion
with MOD: the per-frame distance between upper- and lower-lip mesh vertices,
compared against the ground-truth sequence of the same utterance.

Computing lip vertex positions needs a FLAME-based 3D face decoder
(license-gated, not bundled), so the module stays registered but marked
``requires_external_backend`` until a licensed decoder is wired in.

mod -- mean mouth-opening distance vs ground truth (lower=better).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MouthOpeningDistanceModule(PipelineModule):
    name = "mouth_opening_distance"
    description = "MOD: mouth-opening distance vs GT (DiffPoseTalk, external backend)"
    provenance = "published"
    sources = {
        "mod": "MOD, DiffPoseTalk (Sun et al., arXiv:2310.00434) — https://github.com/DiffPoseTalk/DiffPoseTalk",
    }
    requires_external_backend = True  # FLAME vertex decoder is license-gated
    requires_reference = True
    default_config = {"fps": 30}
    metric_info = {
        "mod": "Mean mouth-opening vertex distance vs ground truth (lower=better)",
    }
    metric_groups = {"mod": "face"}

    def setup(self) -> None:
        logger.warning(
            "mouth_opening_distance unavailable: needs a FLAME vertex decoder "
            "(license-gated, not bundled); mod left unset."
        )

    def process(self, sample):
        return sample

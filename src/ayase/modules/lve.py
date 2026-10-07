"""LVE — Lip Vertex Error (external backend).

MeshTalk (Richard et al., SIGGRAPH 2021, arXiv:2104.08223) evaluates lip
synchrony by the mean Euclidean error of mouth-region mesh vertices between
generated and ground-truth 3D face sequences.

Computing vertex positions needs a 3D face decoder (FLAME/face-model assets,
license-gated, not bundled), so the module stays registered but marked
``requires_external_backend`` until a licensed decoder is wired in.

lve -- mean lip-vertex error vs ground truth (lower=better).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LVEModule(PipelineModule):
    name = "lve"
    description = "LVE: lip-vertex error vs GT (MeshTalk, external backend)"
    provenance = "published"
    sources = {
        "lve": "LVE, MeshTalk (Richard et al., arXiv:2104.08223) — https://github.com/facebookresearch/meshtalk",
    }
    requires_external_backend = True  # 3D face model assets are license-gated
    requires_reference = True
    default_config = {"fps": 30}
    metric_info = {
        "lve": "Mean lip-region vertex error vs ground truth (lower=better)",
    }
    metric_groups = {"lve": "face"}

    def setup(self) -> None:
        logger.warning(
            "lve unavailable: needs a 3D face vertex decoder (license-gated, "
            "not bundled); lve left unset."
        )

    def process(self, sample):
        return sample

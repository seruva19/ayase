"""Visual Chronometer physical-FPS estimation: registered but not implemented.

The authors provide a demo checkpoint with limited-use wording, while their
strongest predictor is not released. A compatible Ayase backend is not yet
implemented. Optical-flow magnitude is not substituted for physical FPS.
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VisualChronometerModule(PipelineModule):
    name = "visual_chronometer"
    description = "Visual Chronometer physical FPS (inference backend unimplemented)"
    input_type = "vid"
    implementation_status = "unimplemented"
    implementation_notes = (
        "A public demo checkpoint exists, but its demo-use wording and source terms remain "
        "unresolved; the strongest model is unreleased. No compatible Ayase predictor or "
        "independently trained replacement is implemented, so physical FPS is not produced."
    )
    requires_external_backend = True
    provenance = "utility"
    sources = {
        "visual_chronometer_phyfps": "Visual Chronometer — https://arxiv.org/abs/2603.14375"
    }
    metric_info = {
        "visual_chronometer_phyfps": (
            "Reserved physical frame rate in FPS; descriptive timing estimate, "
            "unimplemented and always unset"
        )
    }
    metric_groups = {"visual_chronometer_phyfps": "motion"}
    models = [
        {
            "id": "xiangbog/Visual_Chronometer",
            "type": "huggingface",
            "task": "Public VC-Common 10-60 FPS demo checkpoint; not the strongest model",
            "auto_download": False,
            "notes": (
                "Demo-use wording: https://github.com/taco-group/Pulse-of-Motion#model-weights; "
                "no download or inference is attempted by this unimplemented entry."
            ),
        }
    ]

    def setup(self) -> None:
        logger.info("Visual Chronometer is unimplemented; no physical FPS is produced")

    def process(self, sample: Sample) -> Sample:
        return sample

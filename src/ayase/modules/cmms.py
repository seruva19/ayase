"""CMMS learned token-space image quality: registered but not implemented.

The TiTok-S-128 encoder is public, but its output does not constitute CMMS.
The compatible trained regression head is missing from the released assets.
No surrogate or randomly initialized regressor writes a CMMS score.
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CMMSModule(PipelineModule):
    name = "cmms"
    description = "CMMS learned token-space image quality (trained head unavailable)"
    input_type = "img"
    implementation_status = "unimplemented"
    implementation_notes = (
        "The released TiTok-S-128 encoder is available, but the trained CMMS regression "
        "head is not. This entry reserves the published metric and never produces a score."
    )
    requires_external_backend = True
    provenance = "published"
    sources = {"cmms_score": "CMMS — https://arxiv.org/abs/2603.08064"}
    metric_info = {
        "cmms_score": "Reserved CMMS learned quality score; unimplemented and always unset"
    }
    metric_groups = {"cmms_score": "nr_quality"}
    models = [
        {
            "id": "yucornetto/tokenizer_titok_s128_imagenet",
            "type": "huggingface",
            "task": "Published TiTok-S-128 image tokenizer; insufficient without the CMMS head",
        },
        {
            "id": "cmms_best.pt",
            "type": "local",
            "task": "Unreleased trained CMMS regression head",
            "auto_download": False,
            "notes": "No verified public checkpoint; entry is unimplemented, not a manual-install path.",
        },
    ]

    def setup(self) -> None:
        logger.info("CMMS is unimplemented: the trained regression head is unavailable")

    def process(self, sample: Sample) -> Sample:
        return sample

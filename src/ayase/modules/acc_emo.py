"""Acc_emo — target-emotion accuracy (external backend).

Emotion-conditioned talking-head evals (EAMM, arXiv:2205.15278; EAT,
arXiv:2309.04946) measure whether the generated face expresses the intended
emotion: a pretrained emotion recogniser (Emotion-FAN, EmoNet) classifies each
frame and the score is the accuracy against the target label carried in the
sample metadata (``emotion_label``).

EmoNet weights ship under CC BY-NC-ND and Emotion-FAN is not bundled, so the
module stays registered but marked ``requires_external_backend`` until a
licensed emotion recogniser is wired in.

acc_emo -- fraction of frames matching the target emotion (0-1, higher=better).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AccEmoModule(PipelineModule):
    name = "acc_emo"
    description = "Target-emotion accuracy via Emotion-FAN/EmoNet (external backend)"
    provenance = "published"
    sources = {
        "acc_emo": "Acc_emo, EAMM (https://arxiv.org/abs/2205.15278); EAT (https://arxiv.org/abs/2309.04946); Emotion-FAN/EmoNet backends",
    }
    requires_external_backend = True  # EmoNet weights are CC BY-NC-ND
    requires_reference = True
    default_config = {"fps": 15}
    metric_info = {
        "acc_emo": "Fraction of frames classified as the target emotion (0-1, higher=better)",
    }
    metric_groups = {"acc_emo": "face"}

    def setup(self) -> None:
        logger.warning(
            "acc_emo unavailable: no licensed emotion recogniser bundled "
            "(EmoNet is CC BY-NC-ND; Emotion-FAN not bundled); acc_emo left "
            "unset."
        )

    def process(self, sample):
        return sample

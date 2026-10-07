"""WER of lip reading — articulation intelligibility (external backend).

Talking-head evals measure whether generated lip motion is intelligible by
lip-reading the generated video with a visual speech recognition model and
comparing the transcript against the ground-truth utterance (TalkLip,
arXiv:2303.17480; AV-HuBERT, arXiv:2201.02184; Auto-AVSR, arXiv:2303.14307).

AV-HuBERT weights ship under a research-only license and Auto-AVSR requires
a separate trained frontend; neither is bundled in a standard install, so the
module stays registered but marked ``requires_external_backend`` until a
licensed lip-reading backend is wired in.

lip_reading_wer -- word error rate of the lip-read transcript (0-1, lower=better).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LipReadingWERModule(PipelineModule):
    name = "lip_reading_wer"
    description = "WER of lip-read transcript vs utterance (AV-HuBERT/Auto-AVSR)"
    provenance = "published"
    sources = {
        "lip_reading_wer": "lip-reading WER, TalkLip (arXiv:2303.17480); AV-HuBERT (arXiv:2201.02184); Auto-AVSR (arXiv:2303.14307) — https://github.com/facebookresearch/av_hubert",
    }
    requires_external_backend = True  # AV-HuBERT is research-only licensed
    requires_reference = True
    default_config = {"fps": 25}
    metric_info = {
        "lip_reading_wer": "WER of lip-read transcript vs ground-truth caption (0-1, lower=better)",
    }
    metric_groups = {"lip_reading_wer": "face"}

    def setup(self) -> None:
        logger.warning(
            "lip_reading_wer unavailable: no licensed lip-reading backend "
            "(AV-HuBERT weights are research-only; Auto-AVSR not bundled); "
            "lip_reading_wer left unset."
        )

    def process(self, sample):
        return sample

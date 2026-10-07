"""POI-Forensics — person-of-interest audio-visual identity (external backend).

POI-Forensics (Cozzolino et al., CVPRW 2023, arXiv:2204.03083) verifies
identity by comparing audio-visual behavioural patterns of a person of
interest against reference recordings — the sibling pipeline of ID-Reveal
with an additional audio modality.

Upstream ships only a demo with no confirmed released weights, so the module
stays registered but marked ``requires_external_backend`` until the upstream
model is wired in. The video-only sibling metric is ``id_reveal``.

poi_forensics_score -- identity distance to reference recordings (lower=better).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class POIForensicsModule(PipelineModule):
    name = "poi_forensics"
    description = "POI-Forensics audio-visual identity distance (external backend)"
    provenance = "published"
    sources = {
        "poi_forensics_score": "POI-Forensics, Cozzolino et al., CVPRW 2023 (arXiv:2204.03083) — https://github.com/grip-unina/poi-forensics",
    }
    requires_external_backend = True  # upstream ships demo only, weights unconfirmed
    requires_reference = True
    default_config = {"fps": 25}
    metric_info = {
        "poi_forensics_score": "Audio-visual behavioural identity distance vs references (lower=better)",
    }
    metric_groups = {"poi_forensics_score": "face"}

    def setup(self) -> None:
        logger.warning(
            "poi_forensics unavailable: upstream ships only a demo and no "
            "released weights; poi_forensics_score left unset."
        )

    def process(self, sample):
        return sample

"""AED/APD with the published Deep3DFaceRecon extractor (external backend).

PIRenderer's evaluation extracts per-frame 3DMM coefficients with the trained
model of Deng et al. (Deep3DFaceRecon, CVPR 2019) on BFM09, then AED is the
mean distance of the expression coefficients and APD of the pose coefficients
between generated and driver videos.

The Deep3DFaceRecon model requires the Basel Face Model 2009 assets
(``BFM09_model_info.mat`` etc.), which are license-gated and cannot be
downloaded automatically in a standard install. The module therefore stays
registered but marked ``requires_external_backend`` until a user provides a
Deep3DFaceRecon installation; the turnkey TDDFA-based equivalent lives in
``aed_apd`` (adapted provenance — different coefficient parameterization).

aed / apd -- lower = closer to the driver; left None when unavailable.
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AedApdDeep3DModule(PipelineModule):
    name = "aed_apd_deep3d"
    description = "AED/APD via published Deep3DFaceRecon extractor (external backend)"
    provenance = "published"
    sources = {
        "aed": "AED, PIRenderer (Ren et al., arXiv:2109.08379) + Deep3DFaceRecon (Deng et al., arXiv:1903.08527) — https://github.com/RenYurui/PIRender",
        "apd": "APD, PIRenderer (Ren et al., arXiv:2109.08379) + Deep3DFaceRecon (Deng et al., arXiv:1903.08527) — https://github.com/RenYurui/PIRender",
    }
    requires_external_backend = True  # BFM09 assets are license-gated
    requires_reference = True
    default_config = {"fps": 25}
    metric_info = {
        "aed": "Mean distance of Deep3DFaceRecon expression coefficients vs driver (lower=closer)",
        "apd": "Mean distance of Deep3DFaceRecon pose coefficients vs driver (lower=closer)",
    }
    metric_groups = {"aed": "face", "apd": "face"}

    def setup(self) -> None:
        logger.warning(
            "aed_apd_deep3d unavailable: the published protocol needs "
            "Deep3DFaceRecon with BFM09 assets (license-gated, not bundled). "
            "Use the 'aed_apd' module for the adapted TDDFA-based variant."
        )

    def process(self, sample):
        return sample

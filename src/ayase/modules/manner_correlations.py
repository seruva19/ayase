"""Manner correlations — per-person facial-mannerism signature (external backend).

Agarwal, Farid et al. (CVPRW 2019) protect persons of interest by modelling
their idiosyncratic mannerisms: per-frame facial action units, head-pose
angles and lip descriptors are correlated pairwise (190 correlations) and a
per-person one-class SVM distinguishes genuine recordings from fakes.

The upstream protocol ships no released code or trained model — it is a
described procedure — so the module stays registered but marked
``requires_external_backend`` until the correlation+classifier pipeline is
wired in.

manner_correlation -- authenticity score of the mannerism signature (higher=more genuine).
"""

import logging

from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MannerCorrelationsModule(PipelineModule):
    name = "manner_correlations"
    description = "Per-person mannerism correlations + one-class SVM (Agarwal 2019, external backend)"
    provenance = "published"
    sources = {
        "manner_correlation": "manner correlations, Agarwal, Farid et al., CVPRW 2019 (https://openaccess.thecvf.com/content_CVPRW_2019/html/Media_Forensics/Agarwal_Protecting_World_Leaders_Against_Deep_Fakes_CVPRW_2019_paper.html)",
    }
    requires_external_backend = True  # procedure only — no published code
    requires_reference = True
    default_config = {"fps": 25}
    metric_info = {
        "manner_correlation": "Mannerism-signature authenticity score vs reference person (higher=genuine)",
    }
    metric_groups = {"manner_correlation": "face"}

    def setup(self) -> None:
        logger.warning(
            "manner_correlations unavailable: the source publishes a "
            "procedure (190 AU/pose/lip correlations + per-person OC-SVM) "
            "but no code; manner_correlation left unset."
        )

    def process(self, sample):
        return sample

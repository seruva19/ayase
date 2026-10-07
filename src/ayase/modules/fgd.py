"""FGD — Frechet Gesture Distance (external backend).

Yoon et al. (TOG 2020) evaluate co-speech gesture generation with a Frechet
distance between the generated and reference distributions of features from
an autoencoder trained on TED-style gesture motion. For the BEAT dataset the
sanctioned implementation is the PantoMatrix/EMAGE autoencoder checkpoint on
SMPL-X joint rotations.

The trained gesture autoencoder is not bundled (checkpoint must be fetched
from the PantoMatrix release assets), so the module stays registered but
marked ``requires_external_backend`` until the encoder is wired in.

fgd (dataset-level) -- lower = closer to the reference gesture set.
"""

import logging

from ayase.base_modules import BatchMetricModule
from ayase.models import Sample

logger = logging.getLogger(__name__)


class FGDModule(BatchMetricModule):
    name = "fgd"
    description = "Frechet Gesture Distance on autoencoder latents (Yoon et al. 2020)"
    provenance = "published"
    sources = {
        "fgd": "FGD, Yoon et al., TOG 2020 (arXiv:2009.02119); PantoMatrix/EMAGE encoder — https://github.com/PantoMatrix/PantoMatrix",
    }
    requires_external_backend = True  # gesture autoencoder checkpoint not bundled
    requires_reference = True
    default_config = {"fps": 15}
    metric_info = {
        "fgd": "Frechet distance on gesture-autoencoder latents vs reference set (lower=closer)",
    }
    metric_groups = {"fgd": "motion"}

    def setup(self) -> None:
        logger.warning(
            "fgd unavailable: the published FGD encoder (Yoon et al. 2020 "
            "gesture autoencoder / PantoMatrix checkpoint) is not bundled; "
            "fgd left unset."
        )

    def extract_features(self, sample: Sample):
        return None

    def compute_distribution_metric(self, features, reference_features=None):
        return None

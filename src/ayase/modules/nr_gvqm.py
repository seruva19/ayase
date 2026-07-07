"""NR-GVQM — No-Reference Gaming Video Quality Metric (ISM 2018).

The published NR-GVQM extracts nine frame-level NSS/signal-processing features
and maps them to perceptual MOS via a regression model *trained on gaming VQA
datasets*.

Ayase implements the nine features faithfully, but the trained regression model
is not shipped and cannot currently be located in a loadable form. The previous
implementation combined the features with hand-picked constant weights (falsely
annotated as "calibrated") and a set of invented per-feature quality mappings,
so the ``nr_gvqm_score`` it produced was NOT the published metric and was not
calibrated to MOS.

Per the project's no-heuristic policy a named-metric field must be produced by
the real model or left ``None`` — it must never be fabricated from an untrained
/ hand-tuned regression. Until the genuine NR-GVQM regression weights are wired
in, ``nr_gvqm_score`` is left unset.

nr_gvqm_score — higher = better quality (0-1); real trained model only.

REVIVAL NOTES (provisional — no turnkey backend)
Metric: NR-GVQM (ISM 2018).
Category: REDUNDANT.
Why provisional: It is an SVR that regresses 9 frame features to predict VMAF, which ayase already
  computes for real -- so it only approximates something ayase has.
To revive: Not worth reviving -- redundant with ayase's real VMAF. Remove or keep provisional.
Source: NR-GVQM, ISM 2018.
"""

import logging

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class NRGVQMModule(PipelineModule):
    name = "nr_gvqm"
    provisional = True  # no turnkey real backend in a standard install
    description = "NR-GVQM no-reference gaming video quality (ISM 2018; real model only, disabled if unavailable)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "nr_gvqm_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        # The nine NR-GVQM features are well defined, but the paper's trained
        # regression model is not available. Rather than fabricate a score from
        # hand-picked constant weights, disable the module.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "NR-GVQM: trained regression weights unavailable; module disabled "
            "(nr_gvqm_score left unset). Wire the genuine NR-GVQM regression to enable."
        )

    def process(self, sample: Sample) -> Sample:
        # Real backend unavailable -> leave nr_gvqm_score None (graceful).
        return sample

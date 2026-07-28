"""RankDVQA — Ranking-based Deep VQA (WACV 2024).

Full-reference deep VQA trained with ranking-inspired hybrid training without
human MOS labels.

Only the real RankDVQA network produces ``rankdvqa_score``. Earlier revisions
computed a patch-level SSIM as a stand-in — SSIM is not RankDVQA, so that proxy
has been removed. No trained RankDVQA weights are wired up here, so the score is
left ``None``.

GitHub: https://chenfeng-bristol.github.io/RankDVQA/

rankdvqa_score — higher = better quality

REVIVAL NOTES (requires_external_backend — no turnkey backend)
Metric: RankDVQA (WACV 2024).
Category: TRAINING-ONLY.
Why requires_external_backend: Repo is training-code-only, no .pth; but labels are VMAF-generated (NO human MOS
  needed → fully reproducible).
To revive: Reimplement the ranking-based hybrid FR transformer; build the self-labelled DB with
  VMAF-generated labels; train and validate you reproduce the paper's SRCC/PLCC before flipping
  requires_external_backend=False. Effort L; ~ a better-VMAF.
Source: https://chenfeng-bristol.github.io/RankDVQA/
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class RankDVQAModule(ReferenceBasedModule):
    name = "rankdvqa"
    requires_external_backend = True  # no turnkey real backend in a standard install
    description = "RankDVQA ranking-based FR VQA (real model only)"
    metric_field = "rankdvqa_score"
    default_config = {"subsample": 8}
    metric_groups = {
        "rankdvqa_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        logger.warning(
            "RankDVQA: real trained model unavailable; rankdvqa_score left unset."
        )

    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        # No trained RankDVQA weights available; do not fabricate a proxy score.
        return None

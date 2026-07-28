"""RankDVQA — Ranking-based Deep VQA (WACV 2024).

Full-reference deep VQA trained with ranking-inspired hybrid training without
human MOS labels.

Only the real RankDVQA network produces ``rankdvqa_score``. No proxy metric is substituted for RankDVQA. A trained RankDVQA backend is
required; without one, the score is left ``None``.

GitHub: https://chenfeng-bristol.github.io/RankDVQA/

rankdvqa_score — higher = better quality

Backend requirements
Metric: RankDVQA (WACV 2024).
Unavailable because: Repo is training-code-only, no .pth; but labels are VMAF-generated (NO human MOS
  needed → fully reproducible).
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

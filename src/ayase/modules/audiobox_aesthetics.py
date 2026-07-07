"""Audiobox Aesthetics — Meta Audio Aesthetics (Tjandra et al., 2025).

Predicts four aesthetic axes for audio:
  * PQ (Production Quality)
  * CE (Content Enjoyment)
  * PC (Production Complexity)
  * CU (Content Usefulness)

pip install audiobox_aesthetics

Outputs (in ``QualityMetrics``):
  audiobox_production — PQ
  audiobox_enjoyment  — CE
  audiobox_pc         — PC
  audiobox_cu         — CU

Requires the ``audiobox_aesthetics`` package. When it is unavailable the four
metrics are left ``None`` — no spectral heuristic is substituted.
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class AudioboxAestheticsModule(PipelineModule):
    name = "audiobox_aesthetics"
    description = "Meta Audiobox Aesthetics audio quality (2025)"
    default_config = {
        "sample_rate": 16000,
    }
    metric_groups = {
        "audiobox_cu": "audio",
        "audiobox_enjoyment": "audio",
        "audiobox_pc": "audio",
        "audiobox_production": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = None
        self.sample_rate = self.config.get("sample_rate", 16000)

    def setup(self) -> None:
        # Tier 1: audiobox_aesthetics package (real model, all 4 axes)
        try:
            from audiobox_aesthetics.infer import initialize_predictor
            ckpt = self.config.get("ckpt")  # None → package default
            self._model = initialize_predictor(ckpt=ckpt)
            self._ml_available = True
            self._backend = "audiobox"
            logger.info("Audiobox Aesthetics initialised (audiobox_aesthetics package)")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"audiobox_aesthetics init failed: {e}")

        # No real Audiobox Aesthetics backend available -> metrics stay None.
        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "Audiobox Aesthetics unavailable: install the `audiobox_aesthetics` package"
        )

    def _score_audiobox(self, path: Path) -> Optional[dict]:
        """Score using audiobox_aesthetics package, returning all 4 axes."""
        try:
            # AesPredictor.forward expects a list of dicts keyed by data_col
            # ("path" by default).
            rows = self._model.forward([{self._model.data_col: str(path)}])
            if not rows:
                return None
            row = rows[0]
            return {
                "production": float(row.get("PQ")) if row.get("PQ") is not None else None,
                "enjoyment": float(row.get("CE")) if row.get("CE") is not None else None,
                "pc": float(row.get("PC")) if row.get("PC") is not None else None,
                "cu": float(row.get("CU")) if row.get("CU") is not None else None,
            }
        except Exception as e:
            logger.debug(f"Audiobox package scoring failed: {e}")
            return None

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "audiobox":
            return sample
        try:
            scores = self._score_audiobox(sample.path)
            if scores is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.audiobox_production = scores.get("production")
            sample.quality_metrics.audiobox_enjoyment = scores.get("enjoyment")
            sample.quality_metrics.audiobox_pc = scores.get("pc")
            sample.quality_metrics.audiobox_cu = scores.get("cu")
        except Exception as e:
            logger.error(f"Audiobox Aesthetics failed: {e}")
        return sample

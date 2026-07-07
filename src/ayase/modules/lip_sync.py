"""Lip Sync Error — LSE-D / LSE-C (SyncNet / Wav2Lip, 2020).

Measures audio-visual lip synchronisation quality:
  LSE-D (Lip Sync Error - Distance): lower = better sync
  LSE-C (Lip Sync Error - Confidence): higher = better sync

These are SyncNet-defined metrics. Only a real SyncNet backend produces them;
an audio/mouth-energy cross-correlation heuristic is NOT LSE-D/LSE-C and is not
used as a stand-in. When SyncNet is unavailable the scores are left ``None``.

pip install syncnet (or wav2lip)
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LipSyncModule(PipelineModule):
    name = "lip_sync"
    description = "LSE-D/LSE-C lip sync error (SyncNet/Wav2Lip; real model only)"
    default_config = {
        "subsample": 16,
        "sample_rate": 16000,
    }
    metric_groups = {
        "lse_c": "temporal",
        "lse_d": "temporal",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._ml_available = False
        self._backend = None
        self.subsample = self.config.get("subsample", 16)
        self.sample_rate = self.config.get("sample_rate", 16000)

    def setup(self) -> None:
        # Real SyncNet package only.
        try:
            import syncnet

            self._model = syncnet
            self._ml_available = True
            self._backend = "syncnet"
            logger.info("Lip Sync module initialised (syncnet package)")
            return
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"syncnet init failed: {e}")

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning(
            "Lip Sync: SyncNet not installed (pip install syncnet); "
            "LSE-D/LSE-C left unset."
        )

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        if not self._ml_available or self._backend != "syncnet":
            return sample

        try:
            scores = self._score_syncnet(sample.path)
            if scores is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.lse_d = scores.get("lse_d")
            sample.quality_metrics.lse_c = scores.get("lse_c")
            logger.debug(
                f"Lip Sync for {sample.path.name}: "
                f"LSE-D={scores.get('lse_d')} LSE-C={scores.get('lse_c')}"
            )
        except Exception as e:
            logger.error(f"Lip Sync failed: {e}")
        return sample

    def _score_syncnet(self, path: Path) -> Optional[dict]:
        try:
            result = self._model.evaluate(str(path))
            return {
                "lse_d": float(result.get("lse_d", result.get("distance"))),
                "lse_c": float(result.get("lse_c", result.get("confidence"))),
            }
        except Exception as e:
            logger.debug(f"SyncNet scoring failed: {e}")
            return None

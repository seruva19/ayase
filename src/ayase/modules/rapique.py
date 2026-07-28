"""RAPIQUE — Rapid and Accurate Video Quality Prediction of UGC.

IEEE OJSP 2021 — combines bandpass natural scene statistics (NSS) with deep
CNN semantic features fed to a trained SVR for MOS prediction.

Only the real RAPIQUE predictor produces ``rapique_score``. Earlier revisions
extracted NSS + ResNet-50 features and fed them to a randomly-initialised
regression head — that head is untrained, so its output is meaningless and has
been removed. The score is produced solely by the trained pyiqa RAPIQUE metric
when available; otherwise it is left ``None``.

GitHub: https://github.com/vztu/RAPIQUE

rapique_score — higher = better quality

REVIVAL NOTES (requires_external_backend — no turnkey backend)
Metric: RAPIQUE (IEEE OJSP 2021).
Category: CLASSICAL-PORT.
Why requires_external_backend: pyiqa does NOT expose a "rapique" metric (create_metric('rapique') always fails →
  the module's only path is unavailable); the trained SVR is not shipped.
To revive: Reimplement NSS + ResNet-50 features and retrain the SVR on public KoNViD-1k / YT-UGC;
  validate before flipping requires_external_backend=False. Effort L.
Source: https://github.com/vztu/RAPIQUE
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class RAPIQUEModule(PipelineModule):
    name = "rapique"
    requires_external_backend = True  # no turnkey real backend in a standard install
    description = "RAPIQUE rapid NR-VQA (real pyiqa RAPIQUE metric only)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "rapique_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._ml_available = False
        self._backend = None
        self._metric = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._metric = pyiqa.create_metric("rapique", device=self._device)
            self._ml_available = True
            self._backend = "rapique"
            logger.info("RAPIQUE loaded via pyiqa on %s", self._device)
            return
        except (ImportError, Exception) as e:
            logger.info("RAPIQUE pyiqa metric unavailable: %s", e)

        self._backend = "unavailable"
        self._ml_available = False
        logger.warning("RAPIQUE: real trained model unavailable (pyiqa rapique); rapique_score left unset.")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._metric is None:
            return sample

        try:
            import torch

            frames = sample_frames(sample.path, max_frames=self.subsample, color="rgb")
            if not frames:
                return sample

            scores = []
            for frame in frames:
                arr = np.ascontiguousarray(frame, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)
                with torch.no_grad():
                    scores.append(float(self._metric(tensor).item()))

            if scores:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.rapique_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("RAPIQUE failed for %s: %s", sample.path, e)

        return sample

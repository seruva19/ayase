"""KAD Kernel Audio Distance.

Dataset-level audio distribution metric. The published KAD (Kernel Audio
Distance) is defined over learned audio embeddings (the ``kadtk`` toolkit).
Ayase does not wire a real embedding backend, so it does not emit a ``kad``
value in the pipeline: a kernel distance over hand-crafted spectral features is
only a proxy for KAD and is therefore not written into the named metric.

The generic MMD math (``compute_distribution_metric`` / ``_mmd2``) is retained
as a reusable utility so a real embedding backend can plug in later; it does not
run over spectral proxy features in the pipeline.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.models import Sample

logger = logging.getLogger(__name__)


class KADModule(BatchMetricModule):
    name = "kad"
    description = "Kernel Audio Distance for audio generation (real kadtk backend only)"
    default_config = {
        "sample_rate": 16000,
        "kernel": "rbf",
        "sigma": None,
    }
    models = [
        {
            "id": "fadtk/kadtk",
            "type": "other",
            "task": "Kernel Audio Distance backend (learned audio embeddings)",
        },
    ]
    metric_info = {
        "kad": "Kernel Audio Distance between generated and reference audio sets (lower=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.kernel = self.config.get("kernel", "rbf")
        self.sigma = self.config.get("sigma", None)
        self._backend = None

    def setup(self) -> None:
        # A real KAD requires the kadtk toolkit and its learned audio embedding
        # model. That integration is not wired, so the module emits no `kad`.
        self._backend = "unavailable"
        logger.warning(
            "KAD: real Kernel Audio Distance backend (kadtk embeddings) not wired; "
            "spectral-feature proxy removed, kad left unset."
        )

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        # Do not accumulate proxy features -> no fabricated `kad` in the pipeline.
        return None

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
        """Maximum Mean Discrepancy over provided feature sets.

        Pure math utility: computes MMD^2 between ``features`` and
        ``reference_features`` (or a self-split when no reference is given). Only
        meaningful when fed genuine audio embeddings by a real backend.
        """
        gen = np.stack(features).astype(np.float64)
        if reference_features:
            ref = np.stack(reference_features).astype(np.float64)
        else:
            mid = len(gen) // 2
            if mid < 1:
                return float("inf")
            ref = gen[:mid]
            gen = gen[mid:]
        if len(gen) < 1 or len(ref) < 1:
            return float("inf")
        return float(max(_mmd2(gen, ref, self.kernel, self.sigma), 0.0))


def _mmd2(x: np.ndarray, y: np.ndarray, kernel: str, sigma: Optional[float]) -> float:
    kxx = _kernel(x, x, kernel, sigma)
    kyy = _kernel(y, y, kernel, sigma)
    kxy = _kernel(x, y, kernel, sigma)
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean())


def _kernel(x: np.ndarray, y: np.ndarray, kernel: str, sigma: Optional[float]) -> np.ndarray:
    if kernel == "linear":
        return x @ y.T
    d2 = _squared_distances(x, y)
    if sigma is None:
        positive = d2[d2 > 0]
        sigma = float(np.sqrt(np.median(positive))) if positive.size else 1.0
    gamma = 1.0 / (2.0 * max(float(sigma) ** 2, 1e-12))
    return np.exp(-gamma * d2)


def _squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)

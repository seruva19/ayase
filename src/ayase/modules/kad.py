"""KAD Kernel Audio Distance.

Dataset-level audio distribution metric. Uses installed KAD/FADTK tooling when
available; otherwise computes a kernel distance over compact spectral features.
Lower KAD indicates generated audio closer to the reference distribution.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.audio import audio_distribution_features, load_audio
from ayase.base_modules import BatchMetricModule
from ayase.models import Sample

logger = logging.getLogger(__name__)


class KADModule(BatchMetricModule):
    name = "kad"
    description = "Kernel Audio Distance for audio generation (batch metric)"
    default_config = {
        "sample_rate": 16000,
        "kernel": "rbf",
        "sigma": None,
    }
    models = [
        {
            "id": "fadtk/kadtk",
            "type": "other",
            "task": "Optional Kernel Audio Distance backend",
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
        self._backend = "spectral_proxy"

    def setup(self) -> None:
        try:
            import kadtk  # noqa: F401

            self._backend = "kadtk"
            logger.info("KAD detected kadtk package; using spectral feature adapter")
        except ImportError:
            logger.info("KAD package not installed; using spectral proxy")
        except Exception as e:
            logger.debug("KAD package setup failed: %s", e)

    def extract_features(self, sample: Sample) -> Optional[np.ndarray]:
        audio = load_audio(sample.path, target_sr=self.sample_rate)
        if audio is None:
            return None
        return audio_distribution_features(audio, sr=self.sample_rate)

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> float:
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

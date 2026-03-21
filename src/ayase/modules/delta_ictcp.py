"""Delta ICtCp module.

Delta ICtCp measures perceptual color difference in the ICtCp color space
defined by BT.2100 PQ. This color space is designed for HDR content and
provides better perceptual uniformity than traditional color spaces.

Range: 0+ (lower = more similar, 0 = identical).

Full-reference metric.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


def _pq_inverse_eotf(signal: np.ndarray) -> np.ndarray:
    """Apply ST.2084 PQ EOTF (signal → linear light, normalized to [0,1]).

    Converts PQ-encoded values to linear light for ICtCp computation.
    """
    signal = np.clip(signal, 0.0, 1.0)
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875

    Vm2 = np.power(signal, 1.0 / m2)
    num = np.maximum(Vm2 - c1, 0.0)
    den = np.maximum(c2 - c3 * Vm2, 1e-10)
    return np.power(num / den, 1.0 / m1)


def _linear_to_pq(L: np.ndarray) -> np.ndarray:
    """Apply PQ (Perceptual Quantizer) EOTF inverse (BT.2100)."""
    L = np.clip(L, 0, 1)
    m1 = 0.1593017578125
    m2 = 78.84375
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    Lm1 = np.power(L, m1)
    return np.power((c1 + c2 * Lm1) / (1 + c3 * Lm1), m2)


def _rgb_to_ictcp(rgb_linear: np.ndarray) -> np.ndarray:
    """Convert linear RGB (BT.2020) to ICtCp via PQ.

    Simplified pipeline: RGB -> LMS -> PQ -> ICtCp
    """
    # BT.2020 RGB to LMS (exact BT.2100 coefficients)
    r, g, b = rgb_linear[:, :, 0], rgb_linear[:, :, 1], rgb_linear[:, :, 2]
    L = 0.412109375 * r + 0.523925781 * g + 0.063964844 * b
    M = 0.166748047 * r + 0.720458984 * g + 0.112792969 * b
    S = 0.024047852 * r + 0.075439453 * g + 0.900512695 * b

    # Apply PQ
    L_pq = _linear_to_pq(L)
    M_pq = _linear_to_pq(M)
    S_pq = _linear_to_pq(S)

    # LMS_PQ to ICtCp
    intensity = 0.5 * L_pq + 0.5 * M_pq
    Ct = 1.6137 * L_pq - 3.3234 * M_pq + 1.7097 * S_pq
    Cp = 4.3781 * L_pq - 4.2455 * M_pq - 0.1325 * S_pq

    return np.stack([intensity, Ct, Cp], axis=2)


class DeltaICtCpModule(ReferenceBasedModule):
    name = "delta_ictcp"
    description = "Delta ICtCp HDR perceptual color difference (lower=better)"
    default_config = {
        "subsample": 5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._ml_available = True  # Pure numpy

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        try:
            ref = cv2.imread(str(reference_path))
            dist = cv2.imread(str(sample_path))
            if ref is None or dist is None:
                return None

            h = min(ref.shape[0], dist.shape[0])
            w = min(ref.shape[1], dist.shape[1])
            ref = cv2.resize(ref, (w, h))
            dist = cv2.resize(dist, (w, h))

            return self._compute_delta(ref, dist)
        except Exception as e:
            logger.debug(f"Delta ICtCp failed: {e}")
            return None

    def _compute_delta(self, ref_bgr: np.ndarray, dist_bgr: np.ndarray) -> float:
        # Convert BGR to RGB, normalize to [0,1]
        ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        dist_rgb = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Apply PQ inverse EOTF for linearization (BT.2100 PQ content)
        # For SDR content this also works as a reasonable approximation
        ref_linear = _pq_inverse_eotf(ref_rgb)
        dist_linear = _pq_inverse_eotf(dist_rgb)

        # Convert to ICtCp
        ref_ictcp = _rgb_to_ictcp(ref_linear)
        dist_ictcp = _rgb_to_ictcp(dist_linear)

        # Compute Delta ICtCp (Euclidean distance)
        diff = ref_ictcp - dist_ictcp
        delta = np.sqrt(np.sum(diff ** 2, axis=2))
        return float(np.mean(delta))

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        reference = Path(reference) if not isinstance(reference, Path) else reference
        if not reference.exists():
            return sample

        try:
            if sample.is_video:
                score = self._process_video(sample.path, reference)
            else:
                score = self.compute_reference_score(sample.path, reference)

            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.delta_ictcp = score
                logger.debug(f"Delta ICtCp for {sample.path.name}: {score:.4f}")
        except Exception as e:
            logger.error(f"Delta ICtCp failed: {e}")
        return sample

    def _process_video(self, path, ref_path) -> Optional[float]:
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        scores = []
        idx = 0
        while True:
            r1, ref_f = ref_cap.read()
            r2, dist_f = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                h = min(ref_f.shape[0], dist_f.shape[0])
                w = min(ref_f.shape[1], dist_f.shape[1])
                s = self._compute_delta(cv2.resize(ref_f, (w, h)), cv2.resize(dist_f, (w, h)))
                scores.append(s)
            idx += 1
        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None

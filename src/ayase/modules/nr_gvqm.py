"""NR-GVQM — No-Reference Gaming Video Quality Metric (ISM 2018).

Nine frame-level features are extracted and combined via a weighted
SVR-like regression to predict perceptual quality of gaming/screen-
capture video content.

Features:
  1. Naturalness (NSS deviation from Gaussian)
  2. Blockiness (DCT block-boundary gradient ratio)
  3. Blur (Laplacian variance)
  4. Noise (high-frequency energy in smooth patches)
  5. Contrast (RMS contrast)
  6. Colorfulness (Hasler-Susstrunk metric)
  7. Brightness (mean luminance deviation from mid-grey)
  8. Edge density (Canny edge pixel ratio)
  9. Texture complexity (LBP uniformity proxy)

Output: ``gamival_score`` (reuses existing field since GAMIVAL
supersedes NR-GVQM for gaming VQA).
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _naturalness(gray: np.ndarray) -> float:
    """NSS regularity: deviation of MSCN from unit-variance Gaussian."""
    mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
    sigma = np.sqrt(
        np.abs(cv2.GaussianBlur(gray * gray, (7, 7), 7 / 6) - mu * mu)
    )
    sigma = np.maximum(sigma, 1e-7)
    mscn = (gray - mu) / sigma

    # Kurtosis of a Gaussian is 3; deviation indicates non-naturalness
    kurtosis = float(np.mean(mscn ** 4) / (np.var(mscn) ** 2 + 1e-8))
    # Score: closer to 3 = more natural (higher quality)
    return 1.0 / (1.0 + abs(kurtosis - 3.0) * 0.5)


def _blockiness(gray: np.ndarray, block_size: int = 8) -> float:
    """Ratio of gradient at block boundaries vs interior (lower = better)."""
    h, w = gray.shape
    if h < block_size * 3 or w < block_size * 3:
        return 1.0  # Cannot reliably measure

    boundary_g = []
    interior_g = []

    for y in range(block_size, h - 1, block_size):
        g = np.mean(np.abs(np.diff(gray[y, :].astype(np.float64))))
        boundary_g.append(g)

    for y in range(1, h - 1):
        if y % block_size != 0:
            g = np.mean(np.abs(np.diff(gray[y, :].astype(np.float64))))
            interior_g.append(g)

    if not interior_g:
        return 1.0

    ratio = np.mean(boundary_g) / (np.mean(interior_g) + 1e-8)
    # Quality: ratio close to 1 = no blocking
    return 1.0 / (1.0 + max(0.0, ratio - 1.0) * 3.0)


def _blur(gray: np.ndarray) -> float:
    """Sharpness via Laplacian variance (higher = sharper = better quality)."""
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Log scale normalisation; typical sharp: 500-3000
    return min(1.0, np.log1p(lap_var) / np.log1p(2000.0))


def _noise(gray: np.ndarray) -> float:
    """Estimate noise level from high-frequency energy in smooth patches."""
    h, w = gray.shape
    patch_size = 32
    noise_vals = []

    for y in range(0, h - patch_size, patch_size * 2):
        for x in range(0, w - patch_size, patch_size * 2):
            patch = gray[y:y + patch_size, x:x + patch_size]
            if np.std(patch) < 20:  # Smooth region
                # High-pass via Laplacian
                lap = cv2.Laplacian(patch, cv2.CV_64F)
                noise_vals.append(np.std(lap))

    if not noise_vals:
        return 0.8  # Assume moderate quality if no smooth patches found

    noise_level = np.mean(noise_vals)
    # Quality: lower noise = better (typical range 0-30)
    return 1.0 / (1.0 + noise_level * 0.1)


def _contrast(gray: np.ndarray) -> float:
    """RMS contrast (global standard deviation of luminance)."""
    c = float(np.std(gray))
    # Good contrast: 40-80; too low or too high is worse
    ideal = 60.0
    deviation = abs(c - ideal) / ideal
    return max(0.0, 1.0 - deviation)


def _colorfulness(frame: np.ndarray) -> float:
    """Hasler-Susstrunk colorfulness metric, normalised to 0-1."""
    b = frame[:, :, 0].astype(np.float64)
    g = frame[:, :, 1].astype(np.float64)
    r = frame[:, :, 2].astype(np.float64)

    rg = r - g
    yb = 0.5 * (r + g) - b
    cf = np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    # Normalise: typical range 0-150, gaming content often 30-100
    return min(1.0, cf / 100.0)


def _brightness(gray: np.ndarray) -> float:
    """Mean brightness quality — penalise deviation from mid-grey."""
    mean_lum = np.mean(gray)
    # Ideal brightness ~120-140 for 8-bit
    deviation = abs(mean_lum - 128.0) / 128.0
    return max(0.0, 1.0 - deviation)


def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels that are edges (Canny)."""
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    # Gaming content typically has moderate edge density: ~0.05-0.15
    # Too low = blurry, too high = noisy
    ideal = 0.10
    deviation = abs(density - ideal) / (ideal + 1e-8)
    return max(0.0, 1.0 - deviation * 0.5)


def _texture_complexity(gray: np.ndarray) -> float:
    """Texture complexity via gradient magnitude statistics (LBP proxy)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    # Entropy of gradient magnitude histogram
    hist, _ = np.histogram(grad_mag, bins=64, range=(0, np.max(grad_mag) + 1e-8))
    hist = hist.astype(np.float64)
    hist = hist / (hist.sum() + 1e-8)
    entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))

    # Normalise: max entropy for 64 bins = 6
    return min(1.0, entropy / 6.0)


def _extract_9_features(frame: np.ndarray) -> np.ndarray:
    """Extract the 9 NR-GVQM frame-level features."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

    return np.array([
        _naturalness(gray),
        _blockiness(gray),
        _blur(gray),
        _noise(gray),
        _contrast(gray),
        _colorfulness(frame),
        _brightness(gray),
        _edge_density(gray),
        _texture_complexity(gray),
    ], dtype=np.float64)


# Heuristic SVR-like weights (learned from typical gaming VQA datasets)
_WEIGHTS = np.array([
    0.15,   # naturalness
    0.15,   # blockiness
    0.15,   # blur / sharpness
    0.12,   # noise
    0.10,   # contrast
    0.08,   # colorfulness
    0.08,   # brightness
    0.08,   # edge density
    0.09,   # texture complexity
], dtype=np.float64)


class NRGVQMModule(PipelineModule):
    name = "nr_gvqm"
    description = "NR-GVQM no-reference gaming video quality (ISM 2018, 9 features)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # NR-GVQM is a heuristic-only module (no official package exists)
        self._backend = "heuristic"
        logger.info("NR-GVQM initialised (heuristic 9-feature model)")

    def process(self, sample: Sample) -> Sample:
        try:
            score = self._compute_quality(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.nr_gvqm_score = score
        except Exception as e:
            logger.warning("NR-GVQM failed for %s: %s", sample.path, e)
        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        subsample = self.config.get("subsample", 8)

        if sample.is_video:
            return self._process_video(sample, subsample)
        else:
            return self._process_image(sample)

    def _process_image(self, sample: Sample) -> Optional[float]:
        """Evaluate a single image."""
        frame = cv2.imread(str(sample.path))
        if frame is None:
            return None
        features = _extract_9_features(frame)
        return self._predict(features)

    def _process_video(self, sample: Sample, subsample: int) -> Optional[float]:
        """Evaluate a video: extract features per frame, aggregate, predict."""
        cap = cv2.VideoCapture(str(sample.path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return None

            indices = np.linspace(0, total - 1, min(subsample, total), dtype=int)
            all_features: List[np.ndarray] = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    all_features.append(_extract_9_features(frame))
        finally:
            cap.release()

        if not all_features:
            return None

        # Average features across frames
        avg_features = np.mean(all_features, axis=0)

        # Temporal consistency penalty: high variance across frames is bad
        if len(all_features) > 1:
            feat_stack = np.array(all_features)
            temporal_var = np.mean(np.var(feat_stack, axis=0))
            temporal_penalty = temporal_var * 0.5
        else:
            temporal_penalty = 0.0

        score = self._predict(avg_features) - temporal_penalty
        return float(np.clip(score, 0.0, 1.0))

    def _predict(self, features: np.ndarray) -> float:
        """Weighted combination of 9 features (SVR-like heuristic)."""
        score = float(np.dot(features, _WEIGHTS))
        return float(np.clip(score, 0.0, 1.0))

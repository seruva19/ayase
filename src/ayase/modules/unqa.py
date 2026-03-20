"""UNQA — Unified No-Reference Quality Assessment (2024).

A unified framework that shares feature representations across audio,
image, and video modalities for quality prediction.  The key insight is
that perceptual quality degradation manifests as predictable statistical
deviations in natural-content features regardless of modality.

Backend tiers:
  1. **unqa** — official ``unqa`` package with pre-trained multi-modal model
  2. **heuristic** — shared NSS + spectral features across modalities,
     mapped to a quality score via weighted combination

Output: stores to ``confidence_score`` as a proxy (unified quality
confidence across modalities).
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _compute_nss_stats(gray: np.ndarray) -> dict:
    """Compute natural scene statistics features from a grayscale frame."""
    h, w = gray.shape
    mu = cv2.GaussianBlur(gray, (7, 7), 7 / 6)
    sigma = np.sqrt(
        np.abs(cv2.GaussianBlur(gray * gray, (7, 7), 7 / 6) - mu * mu)
    )
    sigma = np.maximum(sigma, 1e-7)
    mscn = (gray - mu) / sigma

    # Generalised Gaussian shape parameter proxy
    mean_abs = np.mean(np.abs(mscn))
    variance = np.var(mscn)
    kurtosis = float(np.mean(mscn ** 4) / (variance ** 2 + 1e-8))
    skewness = float(np.mean(mscn ** 3) / (np.std(mscn) ** 3 + 1e-8))

    return {
        "mscn_mean": float(np.mean(mscn)),
        "mscn_var": float(variance),
        "mscn_mean_abs": float(mean_abs),
        "mscn_kurtosis": kurtosis,
        "mscn_skewness": skewness,
    }


def _compute_spectral_features(gray: np.ndarray) -> dict:
    """Compute frequency-domain features via DCT."""
    # Use a centre crop for speed
    h, w = gray.shape
    crop_h, crop_w = min(h, 256), min(w, 256)
    cy, cx = h // 2, w // 2
    block = gray[cy - crop_h // 2:cy + crop_h // 2,
                 cx - crop_w // 2:cx + crop_w // 2].astype(np.float32)

    dct = cv2.dct(block)
    mag = np.abs(dct)

    # Energy distribution: low-freq vs high-freq ratio
    quarter_h, quarter_w = crop_h // 4, crop_w // 4
    low_energy = np.sum(mag[:quarter_h, :quarter_w])
    high_energy = np.sum(mag[quarter_h:, quarter_w:])
    total_energy = np.sum(mag) + 1e-8
    lf_ratio = float(low_energy / total_energy)
    hf_ratio = float(high_energy / total_energy)

    return {
        "dct_mean": float(np.mean(mag)),
        "dct_std": float(np.std(mag)),
        "lf_ratio": lf_ratio,
        "hf_ratio": hf_ratio,
    }


def _compute_sharpness(gray: np.ndarray) -> float:
    """Laplacian-based sharpness measure."""
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.var(lap))


def _compute_contrast(gray: np.ndarray) -> float:
    """Global contrast via standard deviation."""
    return float(np.std(gray))


def _compute_colorfulness(frame: np.ndarray) -> float:
    """Hasler-Susstrunk colorfulness metric."""
    b, g, r = (
        frame[:, :, 0].astype(np.float64),
        frame[:, :, 1].astype(np.float64),
        frame[:, :, 2].astype(np.float64),
    )
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(
        np.sqrt(rg.var() + yb.var()) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    )


class UNQAModule(PipelineModule):
    name = "unqa"
    description = "UNQA unified no-reference quality for audio/image/video (2024)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: official unqa package
        try:
            import unqa  # noqa: F401
            self._model = unqa
            self._backend = "native"
            logger.info("UNQA initialised (native package)")
            return
        except ImportError:
            pass

        # Tier 2: heuristic multi-modal features
        self._backend = "heuristic"
        logger.info(
            "UNQA initialised (heuristic) — "
            "install unqa for the full unified model"
        )

    def process(self, sample: Sample) -> Sample:
        try:
            score = self._compute_quality(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.confidence_score = score
        except Exception as e:
            logger.warning("UNQA failed for %s: %s", sample.path, e)
        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        if self._backend == "native":
            return self._process_native(sample)
        return self._process_heuristic(sample)

    def _process_native(self, sample: Sample) -> Optional[float]:
        """Use the official UNQA model."""
        result = self._model.predict(str(sample.path))
        return float(result)

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: shared NSS + spectral features → quality score."""
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
        return self._score_frame(frame)

    def _process_video(self, sample: Sample, subsample: int) -> Optional[float]:
        """Evaluate a video by averaging frame scores + temporal consistency."""
        cap = cv2.VideoCapture(str(sample.path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return None

            indices = np.linspace(0, total - 1, min(subsample, total), dtype=int)
            frame_scores = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    s = self._score_frame(frame)
                    if s is not None:
                        frame_scores.append(s)
        finally:
            cap.release()

        if not frame_scores:
            return None

        # Temporal consistency bonus: low variance across frames = more consistent quality
        mean_score = float(np.mean(frame_scores))
        if len(frame_scores) > 1:
            consistency = 1.0 / (1.0 + np.std(frame_scores) * 5.0)
        else:
            consistency = 0.8

        # Weighted combination: content quality + temporal consistency
        score = 0.85 * mean_score + 0.15 * consistency
        return float(np.clip(score, 0.0, 1.0))

    def _score_frame(self, frame: np.ndarray) -> Optional[float]:
        """Compute unified quality score for a single frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

        # Natural scene statistics
        nss = _compute_nss_stats(gray)

        # Spectral features
        spec = _compute_spectral_features(gray)

        # Spatial quality
        sharpness = _compute_sharpness(gray)
        contrast = _compute_contrast(gray)
        colorfulness = _compute_colorfulness(frame)

        # Map to quality score (higher = better)
        # Sharpness: log-scale, typical good values ~500-2000
        sharpness_q = min(1.0, np.log1p(sharpness) / np.log1p(2000.0))

        # Contrast: good images have contrast ~40-80
        contrast_q = min(1.0, contrast / 80.0)

        # Colorfulness: typical range 0-150, sweet spot ~40-100
        color_q = min(1.0, colorfulness / 100.0)

        # NSS regularity: natural images have kurtosis ~3
        kurtosis_dev = abs(nss["mscn_kurtosis"] - 3.0)
        nss_q = 1.0 / (1.0 + kurtosis_dev * 0.3)

        # Spectral balance: natural content has ~60-80% low-frequency energy
        spec_balance = 1.0 - abs(spec["lf_ratio"] - 0.7) * 2.0
        spec_q = max(0.0, min(1.0, spec_balance))

        # High-frequency content: too little = blurry, too much = noisy
        hf_q = 1.0 - abs(spec["hf_ratio"] - 0.15) * 4.0
        hf_q = max(0.0, min(1.0, hf_q))

        # Weighted combination
        score = (
            0.25 * sharpness_q
            + 0.15 * contrast_q
            + 0.10 * color_q
            + 0.20 * nss_q
            + 0.15 * spec_q
            + 0.15 * hf_q
        )

        return float(np.clip(score, 0.0, 1.0))

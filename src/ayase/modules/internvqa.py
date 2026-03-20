"""InternVQA — Lightweight Compressed Video Quality Assessment (2025).

A lightweight no-reference VQA model designed specifically for compressed
video, leveraging codec-aware features alongside spatial quality analysis.
The model recognises common compression artefacts (blocking, ringing,
banding, mosquito noise) and weights them according to their perceptual
impact.

Backend tiers:
  1. **internvqa** — official ``internvqa`` package with pre-trained model
  2. **heuristic** — codec-aware spatial features (DCT blocking, Sobel
     ringing, banding detection) combined via weighted regression

Output: stores to ``dover_score`` as a proxy (skips if already set).
"""

import logging
from typing import Optional, List

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _detect_blocking(gray: np.ndarray, block_size: int = 8) -> float:
    """Detect DCT block boundary artefacts.

    Measures the average gradient magnitude along 8x8 block boundaries
    compared to the interior.  Higher ratio = more blocking.
    """
    h, w = gray.shape
    if h < block_size * 2 or w < block_size * 2:
        return 0.0

    # Horizontal block boundaries
    boundary_grads = []
    interior_grads = []
    for y in range(block_size, h - 1, block_size):
        grad = np.abs(gray[y, :].astype(np.float64) - gray[y - 1, :].astype(np.float64))
        boundary_grads.append(np.mean(grad))
    for y in range(1, h - 1):
        if y % block_size != 0:
            grad = np.abs(gray[y, :].astype(np.float64) - gray[y - 1, :].astype(np.float64))
            interior_grads.append(np.mean(grad))

    if not interior_grads or not boundary_grads:
        return 0.0

    boundary_mean = np.mean(boundary_grads)
    interior_mean = np.mean(interior_grads) + 1e-8

    # Blocking index: ratio > 1 indicates blocking artefacts
    blocking_ratio = boundary_mean / interior_mean
    return float(max(0.0, blocking_ratio - 1.0))


def _detect_ringing(gray: np.ndarray) -> float:
    """Detect ringing/Gibbs artefacts near strong edges.

    Ringing manifests as oscillations near sharp edges.  We detect edges
    with Canny and measure variance in a narrow band around them.
    """
    edges = cv2.Canny(gray.astype(np.uint8), 100, 200)
    if np.sum(edges) == 0:
        return 0.0

    # Dilate edges to get a narrow band
    kernel = np.ones((5, 5), np.uint8)
    band = cv2.dilate(edges, kernel, iterations=1)
    band_mask = (band > 0) & (edges == 0)

    if np.sum(band_mask) == 0:
        return 0.0

    # Measure oscillation in the band via local variance
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    ring_energy = np.std(lap[band_mask])
    global_energy = np.std(lap) + 1e-8

    return float(max(0.0, ring_energy / global_energy - 1.0))


def _detect_banding(gray: np.ndarray) -> float:
    """Detect banding artefacts in smooth gradient regions.

    Banding appears as visible steps in what should be smooth gradients.
    We measure this by looking at the histogram in low-variance patches.
    """
    h, w = gray.shape
    patch_size = 32
    banding_scores = []

    for y in range(0, h - patch_size, patch_size):
        for x in range(0, w - patch_size, patch_size):
            patch = gray[y:y + patch_size, x:x + patch_size]
            if np.std(patch) < 15:  # smooth region
                # Count number of unique intensity levels
                unique = len(np.unique(patch))
                # Fewer unique values in a smooth region = more banding
                expected = min(patch_size * patch_size, 30)
                banding = 1.0 - unique / expected
                banding_scores.append(max(0.0, banding))

    if not banding_scores:
        return 0.0

    return float(np.mean(banding_scores))


def _get_codec_penalty(sample: Sample) -> float:
    """Estimate a codec quality penalty from video metadata.

    Lower bitrate relative to resolution = worse expected quality.
    """
    vm = sample.video_metadata
    if vm is None:
        return 0.0

    if vm.bitrate is None or vm.bitrate <= 0:
        return 0.0

    pixels = (vm.width or 1920) * (vm.height or 1080)
    fps = vm.fps or 30.0

    # Bits per pixel per frame
    bpp = vm.bitrate / (pixels * fps + 1e-8)

    # Typical good quality: bpp > 0.1; bad: bpp < 0.02
    if bpp > 0.1:
        return 0.0
    elif bpp > 0.05:
        return 0.1
    elif bpp > 0.02:
        return 0.2
    else:
        return 0.3


class InternVQAModule(PipelineModule):
    name = "internvqa"
    description = "InternVQA lightweight compressed video quality (2025)"
    default_config = {
        "subsample": 8,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._backend = "heuristic"

    def setup(self) -> None:
        # Tier 1: official internvqa package
        try:
            import internvqa  # noqa: F401
            self._model = internvqa
            self._backend = "native"
            logger.info("InternVQA initialised (native package)")
            return
        except ImportError:
            pass

        # Tier 2: heuristic codec-aware quality
        self._backend = "heuristic"
        logger.info(
            "InternVQA initialised (heuristic) — "
            "install internvqa for the full model"
        )

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        # Skip if dover_score is already set
        if (sample.quality_metrics is not None
                and sample.quality_metrics.dover_score is not None):
            logger.debug(
                "InternVQA: dover_score already set for %s, skipping",
                sample.path.name,
            )
            return sample

        try:
            score = self._compute_quality(sample)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.internvqa_score = score
        except Exception as e:
            logger.warning("InternVQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_quality(self, sample: Sample) -> Optional[float]:
        if self._backend == "native":
            return self._process_native(sample)
        return self._process_heuristic(sample)

    def _process_native(self, sample: Sample) -> Optional[float]:
        """Use the official InternVQA model."""
        result = self._model.predict(str(sample.path))
        return float(result)

    def _process_heuristic(self, sample: Sample) -> Optional[float]:
        """Heuristic: codec-aware spatial features → quality score."""
        subsample = self.config.get("subsample", 8)

        cap = cv2.VideoCapture(str(sample.path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return None

            indices = np.linspace(0, total - 1, min(subsample, total), dtype=int)
            frame_scores: List[float] = []

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)

                # Compression artefact detection
                blocking = _detect_blocking(gray)
                ringing = _detect_ringing(gray)
                banding = _detect_banding(gray)

                # Spatial quality features
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
                contrast = float(np.std(gray))

                # Quality mapping (1.0 = perfect, 0.0 = severe artefacts)
                block_q = 1.0 / (1.0 + blocking * 2.0)
                ring_q = 1.0 / (1.0 + ringing * 1.5)
                band_q = 1.0 / (1.0 + banding * 3.0)
                sharp_q = min(1.0, np.log1p(sharpness) / np.log1p(1500.0))
                contrast_q = min(1.0, contrast / 70.0)

                frame_q = (
                    0.25 * block_q
                    + 0.20 * ring_q
                    + 0.15 * band_q
                    + 0.25 * sharp_q
                    + 0.15 * contrast_q
                )
                frame_scores.append(frame_q)
        finally:
            cap.release()

        if not frame_scores:
            return None

        spatial_q = float(np.mean(frame_scores))

        # Apply codec-aware penalty
        codec_penalty = _get_codec_penalty(sample)
        score = spatial_q - codec_penalty

        return float(np.clip(score, 0.0, 1.0))

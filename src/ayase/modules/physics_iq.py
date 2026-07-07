"""Physics-IQ full-reference physical-understanding metric.

Faithful port of the Physics-IQ benchmark evaluation protocol
(``google-deepmind/physics-IQ-benchmark``, ICCV 2025). A *generated*
continuation video is compared against the *real* future ("continuation")
video via binary motion masks derived from frame differencing.

Metrics (all reference-based — the sample is the generated video, the
reference is the real continuation):

  Spatial IoU           IoU of the time-collapsed motion masks       (0-1, higher=better)
  Spatiotemporal IoU    mean per-frame IoU of the motion masks       (0-1, higher=better)
  Weighted Spatial IoU  IoU of the per-pixel motion-frequency maps   (0-1, higher=better)
  MSE                   mean per-frame RGB MSE on [0,1] frames        (lower=better)
  Physics-IQ score      combined score                               (0-100, higher=better)

The binary motion masks, the three IoU formulas and the MSE are ported
verbatim from the official code:

  * ``physiq/binary_mask_generator.py`` — ``generate_mask`` (grayscale +
    GaussianBlur, ``cv2.accumulateWeighted`` running-average background,
    ``cv2.absdiff`` + fixed ``threshold``, MORPH_OPEN then MORPH_CLOSE;
    the first frame's mask is all zeros).
  * ``physiq/calculate_and_write_metrics_to_csv.py`` — ``load_view``
    (downscale to a quarter resolution, mask binarise ``> 127``),
    ``spatiotemporal_iou_per_frame``, ``spatial_binary_masks``,
    ``weighted_spatial_mask``, ``compute_weighted_spatial_iou`` and
    ``mse_per_frame``.

The combined score follows ``physiq/calculate_iq_score.py``:

    score = mean( ST/var_ST, spatial/var_spatial, weighted/var_weighted )
            - (mse - var_mse)
    score = clamp(round(score * 100, 2), 0, 100)

DEVIATION (documented): the official ``var_*`` terms are the *physical
variance* — the metric between two *independent real takes* (take-1 vs
take-2) of the same scenario. That normalisation requires a second real
recording, which a single-reference module does not have. Here the
normalisation is set to its neutral identity (``var_ST = var_spatial =
var_weighted = 1`` and ``var_mse = 0``), so the score reduces to
``clamp(round((mean(ST, spatial, weighted) - mse) * 100, 2), 0, 100)``.
The four sub-metrics themselves are exact.

This is a deterministic algorithmic metric — there is no learned model,
so a faithful port *is* the real backend (``self._backend = "port"``).
There is no heuristic fallback: a missing / non-video / too-short /
undecodable reference leaves every ``physics_iq_*`` field ``None``.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.base_modules import ReferenceBasedModule
from ayase.models import QualityMetrics, Sample

logger = logging.getLogger(__name__)


class PhysicsIQModule(ReferenceBasedModule):
    name = "physics_iq"
    description = (
        "Physics-IQ physical-understanding protocol (motion-mask IoU + MSE vs real continuation)"
    )
    default_config = {
        # cv2.threshold cutoff on the running-average frame difference
        # (official ``threshold_value`` in binary_mask_generator.generate_mask).
        "motion_threshold": 10,
        # cv2.accumulateWeighted running-average weight (official 0.3).
        "accumulate_alpha": 0.3,
        # GaussianBlur kernel size, (k, k) (official 5).
        "gaussian_kernel": 5,
        # Morphology structuring-element size for OPEN then CLOSE (official 5).
        "morph_kernel": 5,
        # Binarise the downscaled mask with ``> threshold`` (official 127).
        "mask_binarize_threshold": 127,
        # target_size = (W // factor, H // factor); official downscales by 4.
        "downscale_factor": 4,
        # Frame-count cap (fps alignment): 0 = use all matched frames.
        # The official code considers ``fps * 5`` frames from the start.
        "max_frames": 0,
        # Minimum matched frames required to compute motion (need >= 2).
        "min_frames": 2,
        # Optional explicit reference path when sample.reference_path is unset.
        "reference_path": None,
    }
    metric_info = {
        "physics_iq_score": "Combined Physics-IQ score (0-100, higher=better)",
        "physics_iq_spatial_iou": "IoU of time-collapsed motion masks vs real continuation (0-1)",
        "physics_iq_spatiotemporal_iou": "Mean per-frame motion-mask IoU vs real continuation (0-1)",
        "physics_iq_weighted_spatial_iou": "IoU of per-pixel motion-frequency maps vs real continuation (0-1)",
        "physics_iq_mse": "Mean per-frame RGB MSE vs real continuation (lower=better)",
    }
    metric_groups = {
        "physics_iq_score": "fr_quality",
        "physics_iq_spatial_iou": "fr_quality",
        "physics_iq_spatiotemporal_iou": "fr_quality",
        "physics_iq_weighted_spatial_iou": "fr_quality",
        "physics_iq_mse": "fr_quality",
    }

    _VIDEO_SUFFIXES = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v")

    def __init__(self, config=None):
        super().__init__(config)
        self.motion_threshold = int(self.config.get("motion_threshold", 10))
        self.accumulate_alpha = float(self.config.get("accumulate_alpha", 0.3))
        self.gaussian_kernel = int(self.config.get("gaussian_kernel", 5))
        self.morph_kernel = int(self.config.get("morph_kernel", 5))
        self.mask_binarize_threshold = int(self.config.get("mask_binarize_threshold", 127))
        self.downscale_factor = max(1, int(self.config.get("downscale_factor", 4)))
        self.max_frames = int(self.config.get("max_frames", 0))
        self.min_frames = max(2, int(self.config.get("min_frames", 2)))
        # Set to "port" only once a computation actually succeeds.
        self._backend = None

    def setup(self) -> None:
        # Pure cv2/numpy algorithm — no model to load. cv2 and numpy are
        # imported at module scope, so process() is fully self-contained even
        # when setup() is never called (e.g. from tests).
        # Record provenance: "unavailable" once configured, upgraded to "port"
        # the first time a computation actually succeeds. This ensures samples
        # that are skipped (no reference, non-video, too short) report
        # "unavailable" rather than the pre-setup None sentinel.
        self._backend = "unavailable"
        logger.info("Physics-IQ initialised (deterministic port; no model)")

    # ------------------------------------------------------------------
    # Official mask generation (binary_mask_generator.generate_mask) plus
    # the load_view downscale + binarise, computed in a single decode pass.
    # ------------------------------------------------------------------
    def _decode_masks_and_rgb(
        self, path: Path, target_size: Tuple[int, int]
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Decode a video, returning (rgb[N,h,w,3] float [0,1], masks[N,h,w] bool).

        Motion masks reproduce ``generate_mask``: grayscale + GaussianBlur, a
        ``cv2.accumulateWeighted`` running-average background, ``absdiff`` +
        fixed ``threshold``, then MORPH_OPEN followed by MORPH_CLOSE. The first
        frame's mask is all zeros (and seeds the background model). Each mask is
        then downscaled to ``target_size`` and binarised ``> mask threshold``
        (as in ``load_view``); each RGB frame is downscaled and normalised.
        """
        cap = cv2.VideoCapture(str(path))
        rgb_frames: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        avg_frame: Optional[np.ndarray] = None
        kernel = np.ones((self.morph_kernel, self.morph_kernel), np.uint8)
        gk = self.gaussian_kernel
        try:
            if not cap.isOpened():
                return None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # RGB frame for MSE: resize then normalise to [0,1] (load_view).
                rgb = cv2.resize(frame, target_size).astype(np.float64) / 255.0
                rgb_frames.append(rgb)

                # Full-resolution motion mask (generate_mask).
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (gk, gk), 0)
                if avg_frame is None:
                    # First frame: seed background, emit an all-zero mask.
                    avg_frame = gray.astype("float")
                    mask_full = np.zeros_like(gray, dtype=np.uint8)
                else:
                    cv2.accumulateWeighted(gray, avg_frame, self.accumulate_alpha)
                    avg_gray = cv2.convertScaleAbs(avg_frame)
                    frame_diff = cv2.absdiff(gray, avg_gray)
                    _, binary = cv2.threshold(
                        frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY
                    )
                    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
                    mask_full = binary

                # Downscale then binarise the mask (load_view: resize, > 127).
                mask_ds = cv2.resize(mask_full, target_size)
                masks.append(mask_ds > self.mask_binarize_threshold)

                if self.max_frames and len(rgb_frames) >= self.max_frames:
                    break
        finally:
            cap.release()

        if not rgb_frames:
            return None
        return np.stack(rgb_frames), np.stack(masks)

    def _first_frame_size(self, path: Path) -> Optional[Tuple[int, int]]:
        """Return the native (height, width) of the first frame, or None."""
        cap = cv2.VideoCapture(str(path))
        try:
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            if not ret:
                return None
            return frame.shape[0], frame.shape[1]
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Official metric primitives (calculate_and_write_metrics_to_csv).
    # ------------------------------------------------------------------
    @staticmethod
    def _spatiotemporal_iou_per_frame(mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """Per-frame IoU for two (n_frames, h, w) bool arrays.

        Union of 0 (no motion in either mask) yields an IoU of 1.0, matching the
        official ``spatiotemporal_iou_per_frame``.
        """
        spatial_axes = tuple(range(1, mask1.ndim))
        intersection = np.logical_and(mask1, mask2).sum(axis=spatial_axes)
        union = np.logical_or(mask1, mask2).sum(axis=spatial_axes)
        return np.where(union == 0, 1.0, intersection / np.maximum(union, 1))

    @staticmethod
    def _spatial_binary_masks(mask_frames: np.ndarray) -> np.ndarray:
        """Collapse the time dimension: True where any frame has a mask pixel."""
        return mask_frames.any(axis=0)

    @staticmethod
    def _weighted_spatial_mask(mask_frames: np.ndarray) -> np.ndarray:
        """Per-pixel fraction of frames that have a mask pixel."""
        return np.sum(mask_frames, axis=0, dtype=np.uint16) / len(mask_frames)

    @staticmethod
    def _compute_weighted_spatial_iou(weighted_1: np.ndarray, weighted_2: np.ndarray) -> float:
        """IoU between two per-pixel motion-frequency maps."""
        intersection = np.minimum(weighted_1, weighted_2)
        union = np.maximum(weighted_1, weighted_2)
        valid_pixels = union > 0
        if np.sum(valid_pixels) == 0:
            return 1.0
        return float(np.sum(intersection[valid_pixels]) / np.sum(union[valid_pixels]))

    @staticmethod
    def _mse_per_frame(video1: np.ndarray, video2: np.ndarray) -> np.ndarray:
        """Per-frame MSE for two (n_frames, h, w, c) arrays in [0,1]."""
        diff = video1.astype(np.float32) - video2.astype(np.float32)
        return (diff ** 2).mean(axis=(1, 2, 3))

    # ------------------------------------------------------------------
    def _compute(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[Tuple[float, float, float, float, float]]:
        """Return (spatial_iou, spatiotemporal_iou, weighted_iou, mse, score) or None."""
        size = self._first_frame_size(reference_path)
        if size is None:
            logger.debug("physics_iq: could not decode reference %s; skipping", reference_path)
            return None
        ref_h, ref_w = size
        target_size = (
            max(1, ref_w // self.downscale_factor),
            max(1, ref_h // self.downscale_factor),
        )

        ref = self._decode_masks_and_rgb(reference_path, target_size)
        gen = self._decode_masks_and_rgb(sample_path, target_size)
        if ref is None or gen is None:
            logger.debug(
                "physics_iq: undecodable video pair (%s vs %s); skipping",
                sample_path, reference_path,
            )
            return None

        ref_rgb, ref_masks = ref
        gen_rgb, gen_masks = gen

        # Align matched timestamps from the start (both at the same fps).
        n = min(len(ref_masks), len(gen_masks))
        if n < self.min_frames:
            logger.debug(
                "physics_iq: too few matched frames (%d < %d) for %s; skipping",
                n, self.min_frames, sample_path,
            )
            return None
        ref_rgb, ref_masks = ref_rgb[:n], ref_masks[:n]
        gen_rgb, gen_masks = gen_rgb[:n], gen_masks[:n]

        # Spatiotemporal IoU: mean per-frame IoU of the motion masks.
        st_iou = float(np.mean(self._spatiotemporal_iou_per_frame(ref_masks, gen_masks)))

        # Spatial IoU: IoU of the time-collapsed motion masks.
        spatial_ref = self._spatial_binary_masks(ref_masks)
        spatial_gen = self._spatial_binary_masks(gen_masks)
        spatial_iou = float(
            self._spatiotemporal_iou_per_frame(
                spatial_ref[np.newaxis], spatial_gen[np.newaxis]
            )[0]
        )

        # Weighted spatial IoU: IoU of the per-pixel motion-frequency maps.
        weighted_iou = self._compute_weighted_spatial_iou(
            self._weighted_spatial_mask(ref_masks),
            self._weighted_spatial_mask(gen_masks),
        )

        # MSE: mean per-frame RGB MSE on [0,1] frames.
        mse = float(np.mean(self._mse_per_frame(ref_rgb, gen_rgb)))

        # Combined score (calculate_iq_score) with neutral physical-variance
        # normalisation — see the module docstring DEVIATION note.
        combined = (st_iou + spatial_iou + weighted_iou) / 3.0 - mse
        score = round(max(min(combined * 100.0, 100.0), 0.0), 2)

        return spatial_iou, st_iou, weighted_iou, mse, score

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        """ReferenceBasedModule hook: return the combined Physics-IQ score."""
        result = self._compute(sample_path, reference_path)
        return None if result is None else result[4]

    def _resolve_reference(self, sample: Sample) -> Optional[Path]:
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            reference = self.config.get("reference_path")
        if reference is None:
            return None
        return reference if isinstance(reference, Path) else Path(reference)

    def process(self, sample: Sample) -> Sample:
        reference = self._resolve_reference(sample)
        if reference is None:
            logger.debug("physics_iq: no reference for %s; skipping", sample.path.name)
            return sample
        if not reference.exists():
            logger.debug("physics_iq: reference not found: %s", reference)
            return sample
        # Physics-IQ requires paired video continuations; it is undefined for images.
        if not sample.is_video or sample.path.suffix.lower() not in self._VIDEO_SUFFIXES:
            logger.debug("physics_iq: requires a video sample; skipping %s", sample.path.name)
            return sample

        try:
            result = self._compute(sample.path, reference)
        except Exception as e:
            logger.debug("physics_iq: computation failed for %s: %s", sample.path.name, e)
            return sample
        if result is None:
            return sample

        spatial_iou, st_iou, weighted_iou, mse, score = result
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.physics_iq_spatial_iou = spatial_iou
        sample.quality_metrics.physics_iq_spatiotemporal_iou = st_iou
        sample.quality_metrics.physics_iq_weighted_spatial_iou = weighted_iou
        sample.quality_metrics.physics_iq_mse = mse
        sample.quality_metrics.physics_iq_score = score
        self._backend = "port"

        logger.debug(
            "physics_iq for %s: score=%.2f sIoU=%.3f stIoU=%.3f wIoU=%.3f mse=%.4f",
            sample.path.name, score, spatial_iou, st_iou, weighted_iou, mse,
        )
        return sample

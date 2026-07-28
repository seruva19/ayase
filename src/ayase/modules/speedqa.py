"""SpEED-QA — Spatial Efficient Entropic Differencing for Quality Assessment.

Bampis, Gupta, Soundararajan and Bovik, "SpEED-QA: Spatial Efficient Entropic
Differencing for Image and Video Quality", IEEE Signal Processing Letters, vol.
24, no. 9, pp. 1333-1337, Sept. 2017.

This is a deterministic, reduced/full-reference distortion algorithm with NO
trained weights. It is a faithful numpy/scipy port of the upstream MATLAB
release (github.com/christosbampis/SpEED-QA_release), functions:

* ``est_params.m``               — GSM local-variance + conditional entropy
* ``Single_Scale_Video_SPEED.m`` — spatial + temporal entropic differencing
* ``Single_Scale_SpEED.m``       — spatial-only (single image)
* ``SpEED_Video_Demo.m``         — parameters and VideoSpEED = mean_s * mean_t

Pipeline: build a Gaussian-scale band (downsample by 0.5, ``down_size`` times),
mean-subtract with a Gaussian window, then per block estimate the local variance
``ss`` and the GSM conditional entropy ``ent`` (est_params). The entropy-scaled
coefficients are ``ent .* log2(1 + ss)``. Spatial SpEED is the mean absolute
difference of reference vs distorted entropy-scaled coefficients; temporal SpEED
applies the same entropic differencing to adjacent-frame differences. The video
index is ``mean(spatial) * mean(temporal)``.

``speedqa_score`` is the raw SpEED DISTORTION index: identical reference and
distorted signal give ~0, and it increases (monotonically) with distortion.
Requires a reference (``sample.reference_path``); with no reference it is left
``None``.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _gaussian_window(size: int = 7, sigma: float = 7.0 / 6.0) -> np.ndarray:
    """Reproduce MATLAB fspecial('gaussian', 7, 7/6), normalised to sum 1."""
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    h = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0.0
    s = h.sum()
    if s != 0:
        h /= s
    return h


def _est_params(y: np.ndarray, blk: int, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Port of est_params.m — GSM local variance ``ss`` and entropy ``ent``.

    ``y`` is a mean-subtracted band. Returns per-(blk x blk)-block maps of the
    local variance parameter and the conditional differential entropy.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    h, w = y.shape
    h2 = (h // blk) * blk
    w2 = (w // blk) * blk
    if h2 < blk or w2 < blk:
        return np.zeros((0, 0)), np.zeros((0, 0))
    y = y[:h2, :w2]

    # im2col sliding [blk blk] -> overlapping patches (rows = patches, cols = 25)
    sw = sliding_window_view(y, (blk, blk))
    temp = sw.reshape(-1, blk * blk)
    mcu = temp.mean(axis=0)
    tc = temp - mcu
    cu = (tc.T @ tc) / temp.shape[0]

    # eig: zero the negative eigenvalues, rescale to preserve the trace, rebuild
    evals, evecs = np.linalg.eigh(cu)
    pos = evals > 0
    epos = np.where(pos, evals, 0.0)
    tr = evals.sum()
    spos = epos.sum()
    scale = tr / spos if spos > 0 else 0.0
    L = epos * scale
    cu = (evecs * L) @ evecs.T

    # im2col distinct [blk blk] -> non-overlapping blocks (cols = blocks)
    nb_h, nb_w = h2 // blk, w2 // blk
    blocks = (
        y.reshape(nb_h, blk, nb_w, blk)
        .transpose(0, 2, 1, 3)
        .reshape(nb_h * nb_w, blk * blk)
    )
    temp2 = blocks.T  # (blk*blk, n_blocks)

    d = L[L > 0]
    if d.size > 0:
        # ss = sum((cu \ temp2) .* temp2) / (blk*blk) — MATLAB backslash via lstsq
        try:
            sol = np.linalg.solve(cu, temp2)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(cu, temp2, rcond=None)[0]
        ss = np.sum(sol * temp2, axis=0) / (blk * blk)
        ss = ss.reshape(nb_h, nb_w)
    else:
        ss = np.zeros((nb_h, nb_w))

    # differential entropy summed over the positive eigenvalues
    const = np.log(2.0 * np.pi * np.e)
    ent = np.zeros_like(ss)
    for du in d:
        ent = ent + np.log2(np.maximum(ss * du + sigma, 1e-12)) + const
    return ss, ent


def _entropy_scaled(band: np.ndarray, window: np.ndarray, blk: int, sigma: float):
    """Mean-subtract a band and return (ss, entropy-scaled coefficients)."""
    from scipy.ndimage import correlate

    mu = correlate(band, window, mode="nearest")
    ss, ent = _est_params(band - mu, blk, sigma)
    return ss, ent * np.log2(1.0 + ss)


def _downsample(img: np.ndarray, times: int) -> np.ndarray:
    """Downsample by 0.5, ``times`` times (MATLAB imresize(.,0.5))."""
    import cv2

    for _ in range(times):
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    return img


class SpEEDQAModule(PipelineModule):
    name = "speedqa"
    description = "SpEED-QA spatial+temporal entropic differencing (deterministic port; distortion index, higher=worse)"
    default_config = {
        "subsample": 8,
        "blk": 5,
        "sigma_nsq": 0.1,
        "down_size": 4,
        "gaussian_size": 7,
    }
    metric_groups = {
        "speedqa_score": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = int(self.config.get("subsample", 8))
        self.blk = int(self.config.get("blk", 5))
        self.sigma_nsq = float(self.config.get("sigma_nsq", 0.1))
        self.down_size = int(self.config.get("down_size", 4))
        self._window = _gaussian_window(
            int(self.config.get("gaussian_size", 7)), 7.0 / 6.0
        )
        # Pure numpy/scipy/cv2 — deterministic algorithmic port, always available.
        self._backend = "port"

    def setup(self) -> None:
        self._backend = "port"

    # ------------------------------------------------------------------ core

    def _adaptive_down_size(self, min_dim: int) -> int:
        """Cap downsampling so the coarsest band still holds >= 2 blocks."""
        need = 2 * self.blk
        n = 0
        d = min_dim
        while n < self.down_size and d // 2 >= need:
            d //= 2
            n += 1
        return n

    def _single_scale_video(
        self,
        ref: np.ndarray,
        ref_next: np.ndarray,
        dis: np.ndarray,
        dis_next: np.ndarray,
        times: int,
    ) -> Tuple[float, float]:
        """Port of Single_Scale_Video_SPEED.m -> (speed_s, speed_t)."""
        ref = _downsample(ref, times)
        ref_next = _downsample(ref_next, times)
        dis = _downsample(dis, times)
        dis_next = _downsample(dis_next, times)

        # Spatial SpEED
        ss_ref, spatial_ref = _entropy_scaled(ref, self._window, self.blk, self.sigma_nsq)
        ss_dis, spatial_dis = _entropy_scaled(dis, self._window, self.blk, self.sigma_nsq)
        speed_s = float(np.nanmean(np.abs(spatial_ref - spatial_dis)))

        # Temporal SpEED — entropic differencing on frame differences
        ref_diff = ref_next - ref
        dis_diff = dis_next - dis
        from scipy.ndimage import correlate

        mu_ref_diff = correlate(ref_diff, self._window, mode="nearest")
        mu_dis_diff = correlate(dis_diff, self._window, mode="nearest")
        ss_ref_diff, q_ref = _est_params(ref_diff - mu_ref_diff, self.blk, self.sigma_nsq)
        ss_dis_diff, q_dis = _est_params(dis_diff - mu_dis_diff, self.blk, self.sigma_nsq)
        temporal_ref = q_ref * np.log2(1.0 + ss_ref) * np.log2(1.0 + ss_ref_diff)
        temporal_dis = q_dis * np.log2(1.0 + ss_dis) * np.log2(1.0 + ss_dis_diff)
        speed_t = float(np.nanmean(np.abs(temporal_ref - temporal_dis)))
        return speed_s, speed_t

    def _single_scale_spatial(self, ref: np.ndarray, dis: np.ndarray, times: int) -> float:
        """Port of Single_Scale_SpEED.m -> spatial-only speed (single image)."""
        ref = _downsample(ref, times)
        dis = _downsample(dis, times)
        _, spatial_ref = _entropy_scaled(ref, self._window, self.blk, self.sigma_nsq)
        _, spatial_dis = _entropy_scaled(dis, self._window, self.blk, self.sigma_nsq)
        return float(np.nanmean(np.abs(spatial_ref - spatial_dis)))

    def compute_speed(
        self, ref_frames: List[np.ndarray], dis_frames: List[np.ndarray]
    ) -> Optional[float]:
        """Compute the SpEED distortion index from grayscale float frames."""
        n = min(len(ref_frames), len(dis_frames))
        if n == 0:
            return None
        min_dim = min(ref_frames[0].shape[0], ref_frames[0].shape[1])
        times = self._adaptive_down_size(min_dim)

        if n == 1:
            return self._single_scale_spatial(ref_frames[0], dis_frames[0], times)

        speed_s_list: List[float] = []
        speed_t_list: List[float] = []
        for i in range(n - 1):
            s, t = self._single_scale_video(
                ref_frames[i], ref_frames[i + 1], dis_frames[i], dis_frames[i + 1], times
            )
            speed_s_list.append(s)
            speed_t_list.append(t)
        mean_s = float(np.nanmean(speed_s_list))
        mean_t = float(np.nanmean(speed_t_list))
        video_speed = mean_s * mean_t
        if not np.isfinite(video_speed):
            return None
        return video_speed

    # ------------------------------------------------------------------ frames

    def _load_frames(self, path: str, is_video: bool) -> List[np.ndarray]:
        """Load grayscale float64 (0-255) luminance frames (read-only copies)."""
        import cv2

        frames: List[np.ndarray] = []
        if is_video:
            cap = cv2.VideoCapture(path)
            try:
                total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
                step = max(1, total // self.subsample) if total else 1
                indices = list(range(0, total, step))[: self.subsample] if total else []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64))
            finally:
                cap.release()
        else:
            frame = cv2.imread(path)
            if frame is not None:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64))
        return frames

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        if not isinstance(reference, Path):
            reference = Path(reference)
        if not reference.exists():
            return sample

        try:
            import cv2

            ref_frames = self._load_frames(str(reference), sample.is_video)
            dis_frames = self._load_frames(str(sample.path), sample.is_video)
            if not ref_frames or not dis_frames:
                return sample

            # Align distorted frames to reference geometry (copy — inputs read-only).
            n = min(len(ref_frames), len(dis_frames))
            ref_frames = ref_frames[:n]
            aligned: List[np.ndarray] = []
            for i in range(n):
                rh, rw = ref_frames[i].shape
                d = dis_frames[i]
                if d.shape != (rh, rw):
                    d = cv2.resize(d, (rw, rh), interpolation=cv2.INTER_AREA)
                aligned.append(d)

            score = self.compute_speed(ref_frames, aligned)
            if score is not None:
                sample.quality_metrics.speedqa_score = float(score)
        except Exception as e:
            logger.warning("SpEED-QA failed for %s: %s", sample.path, e)

        return sample

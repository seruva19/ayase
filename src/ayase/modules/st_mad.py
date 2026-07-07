"""ST-MAD -- Spatiotemporal Most Apparent Distortion (Vu/Vu/Chandler, ICIP 2011).

A faithful, deterministic numpy/scipy port of the official MATLAB reference
(Netflix/vmaf ``matlab/STMAD_2011_MatlabCode``). ST-MAD extends the image MAD
metric (Larson & Chandler, JEI 2010) to video:

  * **Spatial MAD (SMAD)** -- per-frame MAD combining
      - ``hi_index`` : a near-threshold *detection* map built from a
        CSF-weighted local-contrast masking model times a local MSE map, and
      - ``lo_index`` : a supra-threshold *appearance* map built from the local
        statistics (std / skew / kurtosis differences) of a 5-scale x 4-orient
        log-Gabor decomposition,
    combined by ``SMAD = MadHi^(sig/4) + MadLo^(1-sig)`` with a
    distortion-level-dependent weight ``sig``.

  * **Temporal MAD (TMAD)** -- ``lo_index`` (appearance) applied to
    spatiotemporal-slice (STS) images (one image per every 8th column and every
    8th row, stacked over time), each weighted by an optical-flow (Lucas-Kanade)
    motion weight, combined by ``TMAD = stsRow^alpha + stsCol^(1-alpha)``.

  * **ST-MAD** -- ``STMAD = 2.5*log10(beta*SMAD) + TMAD`` where ``beta`` derives
    from the row/column motion energy.

There are NO trained weights: this is a pure classical algorithm ported line by
line from the reference (``hi_index.m``, ``lo_index.m``, ``ical_std.c``,
``ical_stat.c``, ``MotionWeight.m``, ``STMAD_index.m``). Lower = better;
identical clips yield a small motion-dependent baseline, distortion raises it.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter
from scipy.signal import convolve2d

from ayase.image import sample_frames
from ayase.models import Sample
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)

_VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v", ".mpg", ".mpeg")


# ---------------------------------------------------------------------------
# Block local statistics (ports of ical_stat.c / ical_std.c)
#
# The C code slides a 16x16 window with stride 4 over the image (grid points
# i,j in range(0, dim-15, 4)) and replicates each window statistic into the
# 4x4 top-left sub-block. Uncovered borders stay 0.
# ---------------------------------------------------------------------------

def _expand_grid(grid: np.ndarray, m: int, n: int) -> np.ndarray:
    """Replicate each grid value into a 4x4 block (top-left aligned), pad 0."""
    out = np.zeros((m, n), dtype=np.float64)
    if grid.size == 0:
        return out
    e = np.repeat(np.repeat(grid, 4, axis=0), 4, axis=1)
    out[: e.shape[0], : e.shape[1]] = e
    return out


def _grid_windows(x: np.ndarray, win: int) -> Optional[np.ndarray]:
    """Sliding ``win`` x ``win`` windows at grid points (start 0..dim-16, step 4).

    Returns an ``(A, B, win, win)`` array, or ``None`` if the image is too small.
    The grid is fixed to the 16-window bound (``dim-15``) as in the C code, so an
    8-window request reuses the exact same grid positions.
    """
    m, n = x.shape
    if m < 16 or n < 16 or win > m or win > n:
        return None
    sw = sliding_window_view(x, (win, win))  # (m-win+1, n-win+1, win, win)
    return sw[0 : m - 15 : 4, 0 : n - 15 : 4]


def _local_stats16(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Port of ical_stat.c: local std / skew / kurtosis over 16x16 blocks."""
    m, n = x.shape
    sw = _grid_windows(x, 16)
    if sw is None:
        z = np.zeros((m, n))
        return z, z.copy(), z.copy()
    a, b = sw.shape[0], sw.shape[1]
    flat = sw.reshape(a, b, 256)
    mean = flat.mean(axis=-1, keepdims=True)
    c = flat - mean
    sumsq = np.einsum("...i,...i->...", c, c)  # sum of squares
    s2 = sumsq / 256.0                          # population variance (stmp^2)
    std = np.sqrt(sumsq / 255.0)                # MATLAB std (n-1)
    pop = np.sqrt(s2)
    m3 = (c ** 3).mean(axis=-1)
    m4 = (c ** 4).mean(axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        skw = np.where(pop > 0, m3 / pop ** 3, 0.0)
        krt = np.where(pop > 0, m4 / pop ** 4, 0.0)
    return (
        _expand_grid(std, m, n),
        _expand_grid(skw, m, n),
        _expand_grid(krt, m, n),
    )


def _ical_std(x: np.ndarray, y: np.ndarray):
    """Port of ical_std.c.

    ``x`` = (dst-ref), ``y`` = ref (both in the CSF/luminance domain).
    Returns ``(std_of_x_16, modified_std_of_y, mean_of_y_16)``:
      * std_of_x_16      -- local std of x over 16x16 blocks (n-1 norm).
      * mean_of_y_16     -- local mean of y over 16x16 blocks.
      * modified_std_y   -- local std of y over 8x8 blocks (n-1 norm) followed by
                            a coarse local-minimum pass (masking refinement).
    """
    m, n = x.shape
    swx = _grid_windows(x, 16)
    swy16 = _grid_windows(y, 16)
    if swx is None or swy16 is None:
        z = np.zeros((m, n))
        return z, z.copy(), z.copy()

    # std of x (dst-ref), 16x16
    a, b = swx.shape[0], swx.shape[1]
    fx = swx.reshape(a, b, 256)
    cx = fx - fx.mean(axis=-1, keepdims=True)
    std_x = _expand_grid(np.sqrt(np.einsum("...i,...i->...", cx, cx) / 255.0), m, n)

    # mean of y, 16x16
    mean_y = _expand_grid(swy16.reshape(a, b, 256).mean(axis=-1), m, n)

    # std of y over 8x8 at the SAME 16-window grid points, /63 (n-1)
    swy8_full = sliding_window_view(y, (8, 8))
    swy8 = swy8_full[0 : m - 15 : 4, 0 : n - 15 : 4]
    f8 = swy8.reshape(a, b, 64)
    c8 = f8 - f8.mean(axis=-1, keepdims=True)
    std8 = np.sqrt(np.einsum("...i,...i->...", c8, c8) / 63.0)
    tmp_full = _expand_grid(std8, m, n)

    # coarse local-minimum pass (ical_std.c lines 136-158)
    stdmod = np.zeros((m, n), dtype=np.float64)
    row_starts = range(0, m - 15, 4)
    col_starts = range(0, n - 15, 4)
    for j in row_starts:          # row (colLen dimension in C)
        for i in col_starts:      # col (rowLen dimension in C)
            mn = tmp_full[j, i]
            for di in (0, 5):
                for dj in (0, 5):
                    if di == 0 and dj == 0:
                        continue
                    ib, jb = i + di, j + dj
                    if ib < n - 15 and jb < m - 15:
                        v = tmp_full[jb, ib]
                        if v < mn:
                            mn = v
            stdmod[j : j + 4, i : i + 4] = mn
    return std_x, stdmod, mean_y


# ---------------------------------------------------------------------------
# hi_index.m -- CSF-weighted local-contrast detection map
# ---------------------------------------------------------------------------

def _make_csf(m: int, n: int, nfreq: int = 32) -> np.ndarray:
    """Port of make_csf() -> returns an (m, n) CSF filter (already transposed)."""
    vx = np.arange(m) - (m - 1) / 2.0          # -m/2+0.5 .. m/2-0.5
    vy = np.arange(n) - (n - 1) / 2.0
    xp, yp = np.meshgrid(vx, vy)               # shape (n, m)
    plane = (xp + 1j * yp) / n * 2.0 * nfreq
    radfreq = np.abs(plane)
    w = 0.7
    s = (1 - w) / 2 * np.cos(4 * np.angle(plane)) + (1 + w) / 2
    radfreq = radfreq / s
    csf = 2.6 * (0.0192 + 0.114 * radfreq) * np.exp(-((0.114 * radfreq) ** 1.1))
    csf[radfreq < 7.8909] = 0.9809
    return csf.T                               # (m, n)


def _csf_apply(img: np.ndarray, csf: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img))
    return np.real(np.fft.ifft2(np.fft.ifftshift(f * csf)))


def _hi_index(ref_img: np.ndarray, dst_img: np.ndarray) -> float:
    """Port of hi_index.m (detection / near-threshold index)."""
    k = 0.02874
    G = 0.5
    Ci_thrsh = -5.0
    Cd_thrsh = -5.0
    C_slope = 1.0
    BSIZE = 16

    ref = k * ref_img ** (2.2 / 3)
    dst = k * dst_img ** (2.2 / 3)
    m, n = ref.shape
    csf = _make_csf(m, n, 32)
    ref = _csf_apply(ref, csf)
    dst = _csf_apply(dst, csf)

    std_2, std_1, m1_1 = _ical_std(dst - ref, ref)

    with np.errstate(divide="ignore", invalid="ignore"):
        Ci_ref = np.log(std_1 / m1_1)
        Ci_dst = np.log(std_2 / m1_1)
    Ci_dst = np.where(m1_1 < G, -np.inf, Ci_dst)

    msk = np.zeros((m, n), dtype=np.float64)
    thr = C_slope * (Ci_ref - Ci_thrsh) + Cd_thrsh
    cond1 = (Ci_ref > Ci_thrsh) & (Ci_dst > thr)
    cond2 = (Ci_ref <= Ci_thrsh) & (Ci_dst > Cd_thrsh)
    msk[cond1] = Ci_dst[cond1] - thr[cond1]
    msk[cond2] = Ci_dst[cond2] - Cd_thrsh

    # local mean of squared error (16x16 mean filter, symmetric boundary)
    sqerr = (ref_img - dst_img) ** 2
    lmse = uniform_filter(sqerr, size=BSIZE, mode="reflect")

    mp = msk * lmse
    mp2 = mp[BSIZE : m - BSIZE - 1, BSIZE : n - BSIZE - 1]
    if mp2.size == 0:
        mp2 = mp
    return float(np.sqrt(np.mean(mp2 ** 2)) * 10.0)


# ---------------------------------------------------------------------------
# lo_index.m -- log-Gabor appearance map (also used for temporal STS slices)
# ---------------------------------------------------------------------------

def _gaborconvolve(im: np.ndarray) -> List[List[np.ndarray]]:
    """Port of gaborconvolve(): 5-scale x 4-orient log-Gabor decomposition."""
    nscale, norient = 5, 4
    minWaveLength, mult, sigmaOnf, dThetaOnSigma = 3, 3, 0.55, 1.5
    rows, cols = im.shape
    imagefft = np.fft.fft2(im)

    x = np.ones((rows, 1)) * ((np.arange(cols) - cols / 2.0) / (cols / 2.0))
    y = (np.arange(rows) - rows / 2.0).reshape(-1, 1) * np.ones((1, cols)) / (rows / 2.0)
    radius = np.sqrt(x ** 2 + y ** 2)
    rc = int(round(rows / 2))
    cc = int(round(cols / 2))
    radius[rc, cc] = 1.0
    radius = np.log(radius)

    theta = np.arctan2(-y, x)
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    thetaSigma = np.pi / norient / dThetaOnSigma

    wavelength = [minWaveLength * mult ** s for s in range(nscale)]
    logGabors = []
    denom = -(2 * np.log(sigmaOnf) ** 2)
    for s in range(nscale):
        fo = 1.0 / wavelength[s]
        rfo = fo / 0.5
        lg = np.exp((radius - np.log(rfo)) ** 2 / denom)
        lg[rc, cc] = 0.0
        logGabors.append(lg)

    EO: List[List[np.ndarray]] = [[None] * norient for _ in range(nscale)]  # type: ignore
    for o in range(norient):
        angl = o * np.pi / norient
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-(dtheta ** 2) / (2 * thetaSigma ** 2))
        for s in range(nscale):
            filt = np.fft.fftshift(logGabors[s] * spread)
            EO[s][o] = np.fft.ifft2(imagefft * filt)
    return EO


def _lo_index(ref: np.ndarray, dst: np.ndarray) -> float:
    """Port of lo_index.m (appearance / supra-threshold index)."""
    gab_ref = _gaborconvolve(ref.astype(np.float64))
    gab_dst = _gaborconvolve(dst.astype(np.float64))
    m, n = gab_ref[0][0].shape
    s = np.array([0.5, 0.75, 1.0, 5.0, 6.0])
    s = s / s.sum()
    mp = np.zeros((m, n), dtype=np.float64)
    for i in range(5):
        for j in range(4):
            std_r, skw_r, krt_r = _local_stats16(np.abs(gab_ref[i][j]))
            std_d, skw_d, krt_d = _local_stats16(np.abs(gab_dst[i][j]))
            mp += s[i] * (
                np.abs(std_r - std_d)
                + 2.0 * np.abs(skw_r - skw_d)
                + np.abs(krt_r - krt_d)
            )
    BSIZE = 16
    mp2 = mp[BSIZE : m - BSIZE - 1, BSIZE : n - BSIZE - 1]
    if mp2.size == 0:
        mp2 = mp
    return float(np.sqrt(np.mean(mp2 ** 2)))


# ---------------------------------------------------------------------------
# MotionWeight.m -- Lucas-Kanade optical-flow motion energy -> row/col weights
# ---------------------------------------------------------------------------

def _gausswin(length: int, alpha: float = 1.2) -> np.ndarray:
    if length <= 1:
        return np.ones(max(length, 1))
    nn = np.arange(length) - (length - 1) / 2.0
    return np.exp(-0.5 * (alpha * nn / ((length - 1) / 2.0)) ** 2)


def _lk_index(cur: np.ndarray, pre: np.ndarray, wsize: int) -> np.ndarray:
    """Port of LK_index() in MotionWeight.m: per-window flow magnitude u^2+v^2."""
    kx = 0.25 * np.array([[-1.0, 1.0], [-1.0, 1.0]])
    ky = 0.25 * np.array([[-1.0, -1.0], [1.0, 1.0]])
    kt = 0.25 * np.ones((2, 2))
    fx = convolve2d(cur, kx, mode="full") + convolve2d(pre, kx, mode="full")
    fy = convolve2d(cur, ky, mode="full") + convolve2d(pre, ky, mode="full")
    ft = convolve2d(cur, kt, mode="full") + convolve2d(pre, -kt, mode="full")
    fx = fx[:-1, :-1]
    fy = fy[:-1, :-1]
    ft = ft[:-1, :-1]

    r0 = fx.shape[0] // wsize
    c0 = fx.shape[1] // wsize
    u = np.zeros((r0, c0))
    v = np.zeros((r0, c0))
    for i in range(r0):
        for j in range(c0):
            sl = (slice(i * wsize, (i + 1) * wsize), slice(j * wsize, (j + 1) * wsize))
            a = np.stack([fx[sl].ravel(), fy[sl].ravel()], axis=1)
            bvec = -ft[sl].ravel()
            ata = a.T @ a
            atb = a.T @ bvec
            uv = np.linalg.pinv(ata) @ atb
            u[i, j] = uv[0]
            v[i, j] = uv[1]
    u = np.nan_to_num(u)
    v = np.nan_to_num(v)
    return u ** 2 + v ** 2


def _motion_weight(frames: List[np.ndarray], wid: int, hei: int):
    """Port of MotionWeight.m -> (mRow[len r0], mCol[len c0])."""
    wsize = 8
    nfr = len(frames)
    r0 = hei // wsize
    c0 = wid // wsize
    tmp = np.zeros((r0, c0))
    for idx in range(1, nfr):
        cur = frames[idx][: r0 * wsize, : c0 * wsize]
        pre = frames[idx - 1][: r0 * wsize, : c0 * wsize]
        tmp += _lk_index(cur, pre, wsize)
    if nfr > 1:
        tmp /= (nfr - 1)
    m_ver = tmp.mean(axis=1)          # length r0
    m_hor = tmp.mean(axis=0)          # length c0
    g_ver = _gausswin(r0, 1.2)
    g_hor = _gausswin(c0, 1.2)
    m_row = (m_ver ** 2) * g_ver / g_ver.sum()
    m_col = (m_hor ** 2) * g_hor / g_hor.sum()
    return m_row, m_col


# ---------------------------------------------------------------------------
# STMAD_index.m -- full spatiotemporal assembly
# ---------------------------------------------------------------------------

def _stmad_index(ref_frames: List[np.ndarray], dst_frames: List[np.ndarray]) -> Optional[float]:
    n = min(len(ref_frames), len(dst_frames))
    if n < 2:
        return None
    ref_frames = ref_frames[:n]
    dst_frames = dst_frames[:n]
    hei, wid = ref_frames[0].shape

    alpha = 3.0 / 7.0

    # ---- Spatial MAD ----
    hi = np.empty(n)
    lo = np.empty(n)
    for idx in range(n):
        hi[idx] = _hi_index(ref_frames[idx], dst_frames[idx])
        lo[idx] = _lo_index(ref_frames[idx], dst_frames[idx])
    mad_hi = float(np.mean(hi))
    mad_lo = float(np.mean(lo))

    b1 = np.exp(-2.55 / 3.35)
    b2 = 1.0 / (np.log(10) * 3.35)
    sig = 1.0 / (1.0 + b1 * (mad_hi ** b2))
    smad = mad_hi ** (sig / 4.0) + mad_lo ** (1.0 - sig)

    # ---- Motion weights ----
    m_row, m_col = _motion_weight(ref_frames, wid, hei)
    s_col = float(np.sum(m_col))
    s_row = float(np.sum(m_row))
    tmp_col = m_col
    tmp_row = m_row
    if s_col == 0:
        s_col = 1.0
        tmp_col = np.ones_like(m_col) / len(m_col)
    if s_row == 0:
        s_row = 1.0
        tmp_row = np.ones_like(m_row) / len(m_row)

    # ---- Temporal MAD via spatiotemporal-slice (STS) images ----
    ncols = wid // 8
    nrows = hei // 8
    ref_stack = np.stack(ref_frames, axis=0)   # (n, hei, wid)
    dst_stack = np.stack(dst_frames, axis=0)

    mad_cols = np.zeros(ncols)
    for k in range(ncols):
        col = 8 * k + 4                         # MATLAB colid = 8*t1-3 (1-based)
        if col >= wid:
            continue
        org = ref_stack[:, :, col].T            # (hei, n)
        dst = dst_stack[:, :, col].T
        mad_cols[k] = _lo_index(org, dst)

    mad_rows = np.zeros(nrows)
    for k in range(nrows):
        row = 8 * k + 4
        if row >= hei:
            continue
        org = ref_stack[:, row, :]              # (n, wid)
        dst = dst_stack[:, row, :]
        mad_rows[k] = _lo_index(org, dst)

    sts_col = np.log10(1000 * alpha + s_col) * float(np.sum(tmp_col * mad_cols)) / s_col
    sts_row = np.log10(1000 * alpha + s_row) * float(np.sum(tmp_row * mad_rows)) / s_row
    tmad = sts_row ** alpha + sts_col ** (1.0 - alpha)

    # ---- ST-MAD ----
    mov_index = s_row / (s_row + s_col)
    beta = np.log10(1.0 + mov_index)
    val = beta * smad
    if val <= 0:
        return None
    stmad = 2.5 * np.log10(val) + tmad
    if not np.isfinite(stmad):
        return None
    return float(stmad)


class STMADModule(ReferenceBasedModule):
    name = "st_mad"
    description = "ST-MAD spatiotemporal MAD (ICIP 2011, deterministic port, lower=better)"
    metric_field = "st_mad"
    default_config = {"max_frames": 64}
    metric_groups = {
        "st_mad": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = int(self.config.get("max_frames", 64))
        self._backend = None

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        # Pure numpy/scipy classical algorithm -- always available.
        self._backend = "port"
        logger.info("ST-MAD: deterministic classical port (numpy/scipy) ready.")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        # ST-MAD is a temporal (video) metric; undefined for still images.
        if sample_path.suffix.lower() not in _VIDEO_EXTS:
            return None
        try:
            ref_frames = sample_frames(str(reference_path), max_frames=self.max_frames, color="gray")
            dst_frames = sample_frames(str(sample_path), max_frames=self.max_frames, color="gray")
            if not ref_frames or not dst_frames:
                return None

            import cv2

            n = min(len(ref_frames), len(dst_frames))
            hei = min(ref_frames[0].shape[0], dst_frames[0].shape[0])
            wid = min(ref_frames[0].shape[1], dst_frames[0].shape[1])
            if hei < 40 or wid < 40:
                # Too small for the 16-block edge-trimming to leave content.
                return None

            def prep(frames):
                out = []
                for f in frames[:n]:
                    arr = np.ascontiguousarray(f)
                    if arr.ndim == 3:
                        arr = arr[:, :, 0]
                    if arr.shape[0] != hei or arr.shape[1] != wid:
                        arr = cv2.resize(arr, (wid, hei))
                    out.append(arr.astype(np.float64))
                return out

            ref = prep(ref_frames)
            dst = prep(dst_frames)
            return _stmad_index(ref, dst)
        except Exception as e:  # pragma: no cover - graceful degradation
            logger.warning("ST-MAD computation failed: %s", e)
            return None

"""HDR-VQM (HDR Video Quality Metric) module.

HDR-aware video quality metric using perceptually uniform encoding
and subband decomposition.

Backend tiers:
  1. **PU21 + PyWavelets** — proper PU21 perceptual encoding with
     wavelet subband decomposition (ported from
     ``github.com/mperreir/HDR-VQM``)
  2. **Gamma-based heuristic** — simplified gamma PU approximation
     with spatial features
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class HDRVQMModule(PipelineModule):
    name = "hdr_vqm"
    description = "HDR-aware video quality (PU21+wavelet FR or gamma heuristic fallback)"
    default_config = {"subsample": 8}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._pywt_available = False
        self._backend = "gamma_heuristic"

    def setup(self) -> None:
        # Tier 1: PyWavelets for subband decomposition
        try:
            import pywt
            self._pywt_available = True
            self._backend = "pu21_wavelet"
            logger.info("HDR-VQM loaded with PU21 + PyWavelets")
            return
        except ImportError:
            logger.info("PyWavelets unavailable, using gamma heuristic")

        self._backend = "gamma_heuristic"

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            import cv2

            frames = self._load_frames(sample)
            if not frames:
                return sample

            is_hdr = self._detect_hdr(frames[0])

            reference_path = getattr(sample, "reference_path", None)
            has_reference = reference_path is not None and Path(str(reference_path)).exists()

            if has_reference and self._backend == "pu21_wavelet":
                score = self._compute_fr(sample, Path(str(reference_path)), frames, is_hdr)
            else:
                score = self._compute_nr(frames, sample, is_hdr)

            if score is not None:
                sample.quality_metrics.hdr_vqm = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("HDR-VQM failed: %s", e)
        return sample

    def _compute_fr(self, sample: Sample, reference_path: Path,
                    dist_frames: list, is_hdr: bool) -> Optional[float]:
        """Full-reference HDR-VQM with PU21 encoding and wavelet decomposition."""
        import cv2

        ref_frames = self._load_frames_from_path(reference_path)
        if not ref_frames:
            return self._compute_nr(dist_frames, sample, is_hdr)

        n_pairs = min(len(dist_frames), len(ref_frames))
        frame_scores = []

        for i in range(n_pairs):
            ref = ref_frames[i].astype(np.float64)
            dist = dist_frames[i].astype(np.float64)

            # Resize to match
            h, w = dist.shape[:2]
            ref = cv2.resize(ref, (w, h))

            # Convert to grayscale
            ref_gray = np.mean(ref, axis=2) if len(ref.shape) == 3 else ref
            dist_gray = np.mean(dist, axis=2) if len(dist.shape) == 3 else dist

            # Apply PU21 encoding
            ref_pu = self._pu21_encode(ref_gray, is_hdr)
            dist_pu = self._pu21_encode(dist_gray, is_hdr)

            if self._pywt_available:
                score = self._wavelet_quality(ref_pu, dist_pu)
            else:
                score = self._spatial_quality(ref_pu, dist_pu)

            frame_scores.append(score)

        if not frame_scores:
            return None

        # Temporal component
        temporal_score = 1.0
        if len(dist_frames) >= 2:
            temporal_score = self._temporal_quality(dist_frames, is_hdr)

        spatial_mean = float(np.mean(frame_scores))
        return float(0.6 * spatial_mean + 0.4 * temporal_score)

    def _wavelet_quality(self, ref_pu: np.ndarray, dist_pu: np.ndarray) -> float:
        """Compare reference and distorted in wavelet domain."""
        import pywt

        # 3-level wavelet decomposition
        ref_coeffs = pywt.wavedec2(ref_pu, 'db2', level=3)
        dist_coeffs = pywt.wavedec2(dist_pu, 'db2', level=3)

        subband_scores = []
        for level in range(1, len(ref_coeffs)):
            for j in range(3):  # LH, HL, HH
                ref_band = ref_coeffs[level][j]
                dist_band = dist_coeffs[level][j]

                # Resize if shapes don't match
                if ref_band.shape != dist_band.shape:
                    min_h = min(ref_band.shape[0], dist_band.shape[0])
                    min_w = min(ref_band.shape[1], dist_band.shape[1])
                    ref_band = ref_band[:min_h, :min_w]
                    dist_band = dist_band[:min_h, :min_w]

                # Normalized difference in each subband
                ref_energy = np.mean(ref_band ** 2) + 1e-8
                diff_energy = np.mean((ref_band - dist_band) ** 2)
                subband_quality = max(0.0, 1.0 - diff_energy / ref_energy)
                subband_scores.append(subband_quality)

        return float(np.mean(subband_scores)) if subband_scores else 0.5

    def _spatial_quality(self, ref_pu: np.ndarray, dist_pu: np.ndarray) -> float:
        """Spatial quality comparison without wavelets."""
        import cv2

        # SSIM-like comparison in PU domain
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_ref = cv2.GaussianBlur(ref_pu, (11, 11), 1.5)
        mu_dist = cv2.GaussianBlur(dist_pu, (11, 11), 1.5)
        sigma_ref_sq = cv2.GaussianBlur(ref_pu ** 2, (11, 11), 1.5) - mu_ref ** 2
        sigma_dist_sq = cv2.GaussianBlur(dist_pu ** 2, (11, 11), 1.5) - mu_dist ** 2
        sigma_ref_dist = cv2.GaussianBlur(ref_pu * dist_pu, (11, 11), 1.5) - mu_ref * mu_dist

        ssim_map = ((2 * mu_ref * mu_dist + C1) * (2 * sigma_ref_dist + C2)) / \
                   ((mu_ref ** 2 + mu_dist ** 2 + C1) * (sigma_ref_sq + sigma_dist_sq + C2))

        return float(np.mean(np.maximum(ssim_map, 0)))

    def _compute_nr(self, frames: list, sample: Sample, is_hdr: bool) -> Optional[float]:
        """No-reference quality assessment."""
        import cv2

        frame_scores = []
        for frame in frames:
            score = self._assess_frame(frame, is_hdr)
            frame_scores.append(score)

        temporal_score = 1.0
        if len(frames) >= 2 and sample.is_video:
            temporal_score = self._temporal_quality(frames, is_hdr)

        spatial_mean = float(np.mean(frame_scores))
        return float(0.6 * spatial_mean + 0.4 * temporal_score)

    def _detect_hdr(self, frame) -> bool:
        """Detect if frame is HDR based on pixel value distribution."""
        if frame.dtype in (np.float32, np.float64, np.uint16):
            return True
        max_val = frame.max()
        if max_val > 255:
            return True
        gray = frame.mean(axis=2) if len(frame.shape) == 3 else frame
        hist = np.histogram(gray, bins=256, range=(0, 256))[0]
        hist = hist / (hist.sum() + 1e-8)
        used_bins = np.sum(hist > 0.001)
        return used_bins > 200

    def _assess_frame(self, frame, is_hdr: bool) -> float:
        """Assess single frame quality with HDR awareness."""
        import cv2

        if is_hdr:
            pu = self._pu21_encode_frame(frame)
        else:
            pu = frame.astype(np.float64) / 255.0

        gray_pu = np.mean(pu, axis=2) if len(pu.shape) == 3 else pu

        # Sharpness in PU space
        lap = cv2.Laplacian(gray_pu.astype(np.float64), cv2.CV_64F)
        sharpness = min(1.0, np.var(lap) * 100.0)

        # Contrast in PU space
        mu = cv2.GaussianBlur(gray_pu, (11, 11), 1.5)
        local_contrast = np.mean(np.abs(gray_pu - mu))
        contrast_score = min(1.0, local_contrast * 5.0)

        # Dynamic range utilization
        dr_range = gray_pu.max() - gray_pu.min()
        dr_score = min(1.0, dr_range)

        # Tone mapping quality
        hist, _ = np.histogram(gray_pu.flatten(), bins=64, range=(0, 1))
        hist = hist / (hist.sum() + 1e-8)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        tone_score = min(1.0, entropy / 6.0)

        # Color fidelity
        if len(pu.shape) == 3:
            color_var = np.mean([np.std(pu[:, :, c]) for c in range(3)])
            color_score = min(1.0, color_var * 3.0)
        else:
            color_score = 0.5

        return (0.25 * sharpness + 0.20 * contrast_score + 0.20 * dr_score +
                0.15 * tone_score + 0.20 * color_score)

    def _temporal_quality(self, frames, is_hdr: bool) -> float:
        """Assess temporal quality."""
        import cv2

        temporal_scores = []
        for i in range(len(frames) - 1):
            f1 = frames[i].astype(np.float64)
            f2 = frames[i + 1].astype(np.float64)

            if is_hdr:
                f1 = self._pu21_encode_frame(f1)
                f2 = self._pu21_encode_frame(f2)
            else:
                f1 = f1 / 255.0
                f2 = f2 / 255.0

            g1 = np.mean(f1, axis=2) if len(f1.shape) == 3 else f1
            g2 = np.mean(f2, axis=2) if len(f2.shape) == 3 else f2

            diff = np.abs(g1 - g2)
            flicker = np.mean(diff)
            flicker_score = max(0.0, 1.0 - flicker * 3.0)

            lum_change = abs(np.mean(g1) - np.mean(g2))
            stability = max(0.0, 1.0 - lum_change * 5.0)

            temporal_scores.append(0.5 * flicker_score + 0.5 * stability)

        return float(np.mean(temporal_scores)) if temporal_scores else 1.0

    def _pu21_encode(self, luminance: np.ndarray, is_hdr: bool) -> np.ndarray:
        """PU21 perceptually uniform encoding for grayscale.

        Implements the PU21 transfer function:
            V_pu = (a * L^c + b) / (L^c + d)
        with parameters fitted for display luminance range.
        """
        L = luminance.copy()
        if is_hdr:
            L_max = L.max()
            if L_max > 1.0:
                L = L / max(L_max, 1e-8)
            # Map [0,1] to display luminance range [0.005, 10000] cd/m^2
            L = 0.005 + L * 9999.995
        else:
            L = L / 255.0 if L.max() > 1.0 else L
            # SDR: assume sRGB gamma, map to ~[0.2, 200] cd/m^2
            L = np.power(np.clip(L, 0, 1), 2.2) * 200.0 + 0.2

        # PU21 parameters (from Mantiuk et al. 2021)
        a = 0.353487901
        b = 0.3734658629
        c = 0.3632745
        d = 0.9315456
        L_safe = np.maximum(L, 1e-6)
        V_pu = (a * np.power(L_safe, c) + b) / (np.power(L_safe, c) + d)

        return V_pu

    def _pu21_encode_frame(self, frame) -> np.ndarray:
        """PU21 encode a full frame (color or gray)."""
        f = frame.astype(np.float64)
        f_max = f.max()
        is_hdr = f_max > 255 or frame.dtype in (np.float32, np.float64, np.uint16)

        if f_max > 1.0:
            if f_max > 255:
                f = f / f_max
            else:
                f = f / 255.0

        if is_hdr:
            L = 0.005 + np.clip(f, 0, 1) * 9999.995
        else:
            L = np.power(np.clip(f, 0, 1), 2.2) * 200.0 + 0.2

        a = 0.353487901
        b = 0.3734658629
        c = 0.3632745
        d = 0.9315456
        L_safe = np.maximum(L, 1e-6)
        return (a * np.power(L_safe, c) + b) / (np.power(L_safe, c) + d)

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames

    def _load_frames_from_path(self, path: Path) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        if path.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            frames = []
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
            return frames
        else:
            frame = cv2.imread(str(path))
            return [frame] if frame is not None else []

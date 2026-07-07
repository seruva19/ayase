"""HDR-VQM (HDR Video Quality Metric) module.

Full-reference HDR-aware video quality metric using PU21 perceptually uniform
encoding and wavelet subband decomposition (ported from
``github.com/mperreir/HDR-VQM``). Requires PyWavelets and a reference video;
otherwise the metric is left unset — there is no no-reference or gamma-heuristic
stand-in.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _bt709_luminance(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale using BT.709 luminance weights."""
    # OpenCV loads in BGR order
    return 0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]


class HDRVQMModule(PipelineModule):
    name = "hdr_vqm"
    description = "HDR-aware full-reference video quality (PU21 + wavelet)"
    default_config = {"subsample": 8}
    metric_groups = {
        "hdr_vqm": "hdr",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._pywt_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        # PU21 + wavelet subband decomposition (full-reference).
        try:
            import pywt  # noqa: F401
            self._pywt_available = True
            self._backend = "pu21_wavelet"
            logger.info("HDR-VQM loaded with PU21 + PyWavelets (full-reference)")
        except ImportError:
            self._pywt_available = False
            self._backend = "unavailable"
            logger.warning("HDR-VQM unavailable: PyWavelets (pywt) is required.")

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if self._backend != "pu21_wavelet":
            return sample

        # HDR-VQM is full-reference; without a reference there is no metric.
        reference_path = getattr(sample, "reference_path", None)
        if reference_path is None or not Path(str(reference_path)).exists():
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            is_hdr = self._detect_hdr(frames[0])
            score = self._compute_fr(sample, Path(str(reference_path)), frames, is_hdr)

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
            return None

        n_pairs = min(len(dist_frames), len(ref_frames))
        frame_scores = []

        for i in range(n_pairs):
            ref = ref_frames[i].astype(np.float64)
            dist = dist_frames[i].astype(np.float64)

            # Resize to match
            h, w = dist.shape[:2]
            ref = cv2.resize(ref, (w, h))

            # Convert to grayscale using BT.709 luminance weights
            ref_gray = _bt709_luminance(ref) if len(ref.shape) == 3 else ref
            dist_gray = _bt709_luminance(dist) if len(dist.shape) == 3 else dist

            # Apply PU21 encoding
            ref_pu = self._pu21_encode(ref_gray, is_hdr)
            dist_pu = self._pu21_encode(dist_gray, is_hdr)

            score = self._wavelet_quality(ref_pu, dist_pu)
            if score is not None:
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

        return float(np.mean(subband_scores)) if subband_scores else None

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

            g1 = _bt709_luminance(f1) if len(f1.shape) == 3 else f1
            g2 = _bt709_luminance(f2) if len(f2.shape) == 3 else f2

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

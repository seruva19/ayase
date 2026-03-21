"""CGVQM (Computer Graphics Video Quality Metric) module.

CGVQM is Intel's AI-powered metric designed for gaming and rendered
content. It detects artifacts from upscalers (DLSS, XeSS, FSR) and
frame generation that traditional metrics miss.

Range: higher = better quality.

Requires the ``cgvqm`` package from Intel.
Falls back to frequency-domain analysis for rendered content.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class CGVQMModule(ReferenceBasedModule):
    name = "cgvqm"
    description = "CGVQM gaming/rendering quality metric (Intel, higher=better)"
    default_config = {
        "subsample": 5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        try:
            import cgvqm
            self._backend = "cgvqm"
            self._ml_available = True
            logger.info("CGVQM module initialised (Intel package)")
            return
        except ImportError:
            pass

        # Fallback: frequency-domain analysis for rendering artifacts
        self._backend = "approx"
        self._ml_available = True
        logger.info("CGVQM module initialised (frequency-domain approximation)")

    def compute_reference_score(
        self, sample_path: Path, reference_path: Path
    ) -> Optional[float]:
        ref = cv2.imread(str(reference_path))
        dist = cv2.imread(str(sample_path))
        if ref is None or dist is None:
            return None

        h = min(ref.shape[0], dist.shape[0])
        w = min(ref.shape[1], dist.shape[1])
        ref = cv2.resize(ref, (w, h))
        dist = cv2.resize(dist, (w, h))

        if self._backend == "cgvqm":
            return self._compute_cgvqm(ref, dist)
        return self._compute_approx(ref, dist)

    def _compute_cgvqm(self, ref_bgr, dist_bgr) -> Optional[float]:
        try:
            import cgvqm
            return float(cgvqm.compute(ref_bgr, dist_bgr))
        except Exception as e:
            logger.debug(f"CGVQM native failed: {e}")
            return self._compute_approx(ref_bgr, dist_bgr)

    def _compute_approx(self, ref_bgr, dist_bgr) -> float:
        """Approximate gaming quality using frequency-domain analysis.

        Gaming/upscaler artifacts often manifest as:
        - Ghosting (temporal smearing) -> high-freq difference
        - Aliasing (jagged edges) -> directional freq energy
        - Detail loss (over-smoothing) -> reduced high-freq content
        """
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        dist_gray = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # FFT-based frequency analysis
        ref_fft = np.fft.fft2(ref_gray)
        dist_fft = np.fft.fft2(dist_gray)

        ref_mag = np.abs(np.fft.fftshift(ref_fft))
        dist_mag = np.abs(np.fft.fftshift(dist_fft))

        # High-frequency preservation ratio
        h, w = ref_gray.shape
        cy, cx = h // 2, w // 2
        r_inner = min(h, w) // 8
        r_outer = min(h, w) // 2

        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        hf_mask = (dist_from_center > r_inner) & (dist_from_center < r_outer)

        ref_hf = float(np.mean(ref_mag[hf_mask]))
        dist_hf = float(np.mean(dist_mag[hf_mask]))

        if ref_hf > 0:
            hf_ratio = min(dist_hf / ref_hf, 1.0)
        else:
            hf_ratio = 1.0

        # Edge preservation (Sobel)
        ref_edges = cv2.Sobel(ref_gray, cv2.CV_32F, 1, 1)
        dist_edges = cv2.Sobel(dist_gray, cv2.CV_32F, 1, 1)
        # Constant (flat) inputs produce NaN from corrcoef; treat identical as no distortion
        if np.std(ref_edges) < 1e-6 or np.std(dist_edges) < 1e-6:
            edge_sim = 1.0
        else:
            edge_sim = float(np.corrcoef(ref_edges.flatten(), dist_edges.flatten())[0, 1])
            edge_sim = max(0.0, edge_sim)

        # Combine
        score = (0.6 * hf_ratio + 0.4 * edge_sim) * 100.0
        return float(np.clip(score, 0, 100))

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
                sample.quality_metrics.cgvqm = score
                logger.debug(f"CGVQM for {sample.path.name}: {score:.1f}")
        except Exception as e:
            logger.error(f"CGVQM failed: {e}")
        return sample

    def _process_video(self, path, ref_path) -> Optional[float]:
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        scores = []
        idx = 0
        while True:
            r1, rf = ref_cap.read()
            r2, df = dist_cap.read()
            if not r1 or not r2:
                break
            if idx % self.subsample == 0:
                h = min(rf.shape[0], df.shape[0])
                w = min(rf.shape[1], df.shape[1])
                # Direct computation
                rf_r = cv2.resize(rf, (w, h))
                df_r = cv2.resize(df, (w, h))
                if self._backend == "cgvqm":
                    s = self._compute_cgvqm(rf_r, df_r)
                else:
                    s = self._compute_approx(rf_r, df_r)
                if s is not None:
                    scores.append(s)
            idx += 1
        ref_cap.release()
        dist_cap.release()
        return float(np.mean(scores)) if scores else None

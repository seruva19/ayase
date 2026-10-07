"""PSNR-HVS (Egiazarian et al., 2006) + own AC masking — author CSF table.

``sample.path`` is the distorted/candidate image or video and
``sample.reference_path`` is its spatially and temporally corresponding reference;
prompts and masks are not used. Images are resized to their minimum common width
and height. Videos compare same-index decoded frames every ``subsample`` frames
until either stream ends, then average the frame scores.

The in-tree backend converts frames to grayscale and applies the authors'
published CSF coefficients to 8x8 DCT errors, with the author-defined
normalisation 10·log10(255²·ΣCSF²·N / Σ(err·CSF)²). ``psnr_acmask`` additionally
attenuates errors using reference-block AC energy — an own extension, not the
authors' PSNR-HVS-M masking model. Values are in dB and higher is better;
numerically negligible error is reported as 100 dB, while other values have no
enforced range. Results depend on registration, resizing, frame
synchronization, and luma while ignoring chroma.

Algorithm basis and reference implementation: https://www.ponomarenko.info/psnrhvsm.htm
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)

# Authors' CSF coefficient table for 8x8 DCT (psnrhvsm.m, ponomarenko.info).
_CSF = np.array(
    [
        [1.608, 1.563, 1.396, 1.112, 0.749, 0.457, 0.262, 0.143],
        [1.563, 1.523, 1.363, 1.091, 0.740, 0.456, 0.264, 0.145],
        [1.396, 1.363, 1.224, 0.990, 0.682, 0.427, 0.250, 0.139],
        [1.112, 1.091, 0.990, 0.818, 0.581, 0.373, 0.224, 0.127],
        [0.749, 0.740, 0.682, 0.581, 0.434, 0.294, 0.185, 0.109],
        [0.457, 0.456, 0.427, 0.373, 0.294, 0.213, 0.143, 0.089],
        [0.262, 0.264, 0.250, 0.224, 0.185, 0.143, 0.102, 0.068],
        [0.143, 0.145, 0.139, 0.127, 0.109, 0.089, 0.068, 0.049],
    ]
)
_CSF_SQ_SUM = float((_CSF ** 2).sum())


class PSNRHVSModule(ReferenceBasedModule):
    name = "psnr_hvs_approx"
    provenance = {
        "psnr_hvs_approx": "adapted",
        "psnr_acmask": "own",
    }
    sources = {
        "psnr_hvs_approx": "PSNR-HVS (Egiazarian et al., 2006) — https://www.ponomarenko.info/psnrhvsm.htm",
        "psnr_acmask": "PSNR-HVS-M (Ponomarenko et al., 2007) claimed — https://www.ponomarenko.info/psnrhvsm.htm",
    }
    deviations = {
        "psnr_hvs_approx": "Ayase reimplements the published CSF-weighted DCT formula; "
        "numerical equivalence with the authors' reference implementation has not been "
        "established, hence the explicit _approx field name",
        "psnr_acmask": "masking by reference AC energy is an own model, not the authors' PSNR-HVS-M",
    }
    description = "PSNR-HVS approximation with published CSF + own AC masking (dB, higher=better)"
    default_config = {
        "subsample": 5,
    }
    metric_groups = {
        "psnr_hvs_approx": "fr_quality",
        "psnr_acmask": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 5)
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        # DCT-based CSF-weighted implementation is the correct PSNR-HVS algorithm.
        # piq.psnr() computes plain PSNR (not HVS-weighted) — do not use it here.
        self._backend = "dct"
        self._ml_available = True
        logger.info("PSNR-HVS module initialised (CSF-weighted DCT)")

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

        return self._compute_psnr_hvs(ref, dist)

    def _compute_psnr_hvs(self, ref_bgr, dist_bgr) -> Optional[float]:
        """Compute PSNR-HVS using CSF-weighted DCT blocks."""
        return self._compute_psnr_hvs_dct(ref_bgr, dist_bgr)

    def _compute_psnr_hvs_dct(self, ref_bgr, dist_bgr) -> Optional[float]:
        """Compute PSNR-HVS using CSF-weighted DCT blocks."""
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        dist_gray = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

        h, w = ref_gray.shape
        err_sq = 0.0
        n_coef = 0

        for y in range(0, h - 7, 8):
            for x in range(0, w - 7, 8):
                ref_dct = cv2.dct(ref_gray[y:y+8, x:x+8])
                dist_dct = cv2.dct(dist_gray[y:y+8, x:x+8])
                diff = (ref_dct - dist_dct) * _CSF
                err_sq += float((diff ** 2).sum())
                n_coef += 64

        if n_coef == 0:
            return None
        if err_sq < 1e-10:
            return 100.0

        # Authors' normalisation (psnrhvs.m): 10*log10(255²·ΣCSF²·N / Σ(err·CSF)²)
        return float(10.0 * np.log10(255.0 ** 2 * _CSF_SQ_SUM * n_coef / (64.0 * err_sq)))

    def _compute_psnr_hvs_m(self, ref_bgr, dist_bgr) -> Optional[float]:
        """Compute PSNR-HVS-M using CSF-weighted DCT blocks with masking.

        PSNR-HVS-M extends PSNR-HVS by adding a masking correction that
        reduces the weight of errors in regions with high local contrast
        (where distortions are less visible).
        """
        ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
        dist_gray = cv2.cvtColor(dist_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)

        h, w = ref_gray.shape
        weighted_mse = 0.0
        count = 0

        for y in range(0, h - 7, 8):
            for x in range(0, w - 7, 8):
                ref_dct = cv2.dct(ref_gray[y:y+8, x:x+8])
                dist_dct = cv2.dct(dist_gray[y:y+8, x:x+8])

                # Masking: compute local contrast energy from reference DCT
                # AC energy (excluding DC component at [0,0])
                ac_energy = np.sum(ref_dct[1:, :] ** 2) + np.sum(ref_dct[0, 1:] ** 2)
                mask_factor = max(1.0, ac_energy / 1000.0)

                diff = (ref_dct - dist_dct) * _CSF
                block_mse = float(np.mean(diff ** 2))
                # Masking reduces perceived error in high-contrast regions
                weighted_mse += block_mse / mask_factor
                count += 1

        if count == 0:
            return None

        avg_wmse = weighted_mse / count
        if avg_wmse < 1e-10:
            return 100.0

        return float(10.0 * np.log10(255.0 ** 2 / avg_wmse))

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
                hvs, hvs_m = self._process_video(sample.path, reference)
            else:
                hvs = self.compute_reference_score(sample.path, reference)
                ref = cv2.imread(str(reference))
                dist = cv2.imread(str(sample.path))
                if ref is not None and dist is not None:
                    h = min(ref.shape[0], dist.shape[0])
                    w = min(ref.shape[1], dist.shape[1])
                    hvs_m = self._compute_psnr_hvs_m(
                        cv2.resize(ref, (w, h)), cv2.resize(dist, (w, h))
                    )
                else:
                    hvs_m = None

            if hvs is not None or hvs_m is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                if hvs is not None:
                    sample.quality_metrics.psnr_hvs_approx = hvs
                if hvs_m is not None:
                    sample.quality_metrics.psnr_acmask = hvs_m
                logger.debug(
                    f"PSNR-HVS for {sample.path.name}: "
                    f"HVS={hvs:.2f} dB, HVS-M={hvs_m:.2f} dB"
                    if hvs is not None and hvs_m is not None
                    else f"PSNR-HVS for {sample.path.name}: computed"
                )
        except Exception as e:
            logger.error(f"PSNR-HVS failed: {e}")
        return sample

    def _process_video(self, path, ref_path):
        ref_cap = cv2.VideoCapture(str(ref_path))
        dist_cap = cv2.VideoCapture(str(path))
        hvs_scores = []
        hvs_m_scores = []
        idx = 0
        try:
            while True:
                r1, rf = ref_cap.read()
                r2, df = dist_cap.read()
                if not r1 or not r2:
                    break
                if idx % self.subsample == 0:
                    h = min(rf.shape[0], df.shape[0])
                    w = min(rf.shape[1], df.shape[1])
                    rf_r = cv2.resize(rf, (w, h))
                    df_r = cv2.resize(df, (w, h))
                    s = self._compute_psnr_hvs(rf_r, df_r)
                    if s is not None:
                        hvs_scores.append(s)
                    s_m = self._compute_psnr_hvs_m(rf_r, df_r)
                    if s_m is not None:
                        hvs_m_scores.append(s_m)
                idx += 1
        finally:
            ref_cap.release()
            dist_cap.release()
        hvs = float(np.mean(hvs_scores)) if hvs_scores else None
        hvs_m = float(np.mean(hvs_m_scores)) if hvs_m_scores else None
        return hvs, hvs_m

"""BD-Rate (Bjøntegaard Delta Rate) module.

BD-Rate measures the average bitrate difference between two codecs
at equivalent quality for codec comparison.

BD-Rate: % (negative = better compression, e.g., -20% means 20% bitrate savings)
BD-PSNR: dB (positive = better quality at same bitrate)

This is a dataset-level metric. Candidate curve points come from processed
samples; the baseline codec curve is supplied through ``reference_curve`` in
module configuration. Both curves require at least four encoding points.

Stores results in DatasetStats.bd_rate and DatasetStats.bd_psnr.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.models import Sample
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


def _bd_rate(rate1, quality1, rate2, quality2) -> Optional[float]:
    """Compute BD-Rate using the original global cubic-polynomial method.

    Based on JCTVC-L1100 (Bjøntegaard 2001, updated by Pateux/Jung 2007).

    Args:
        rate1, quality1: Arrays of (bitrate, quality) for codec 1
        rate2, quality2: Arrays of (bitrate, quality) for codec 2

    Returns:
        BD-Rate in percentage (negative = codec2 is better)
    """
    R1 = np.asarray(rate1, dtype=float)
    R2 = np.asarray(rate2, dtype=float)
    Q1 = np.asarray(quality1, dtype=float)
    Q2 = np.asarray(quality2, dtype=float)
    if any(a.ndim != 1 for a in (R1, R2, Q1, Q2)):
        return None
    if len(R1) != len(Q1) or len(R2) != len(Q2):
        return None
    if any(not np.all(np.isfinite(a)) for a in (R1, R2, Q1, Q2)):
        return None
    if np.any(R1 <= 0) or np.any(R2 <= 0):
        return None
    if len(np.unique(Q1)) < 4 or len(np.unique(Q2)) < 4:
        return None
    lR1 = np.log(R1)
    lR2 = np.log(R2)

    # Need at least 4 points per curve for the cubic interpolation —
    # no linear substitute: a mean-rate difference is not BD-Rate.
    if len(Q1) < 4 or len(Q2) < 4:
        return None

    # Sort by quality
    idx1 = np.argsort(Q1)
    idx2 = np.argsort(Q2)
    Q1, lR1 = Q1[idx1], lR1[idx1]
    Q2, lR2 = Q2[idx2], lR2[idx2]

    # Overlap range
    q_min = max(Q1[0], Q2[0])
    q_max = min(Q1[-1], Q2[-1])

    if q_min >= q_max:
        # No quality-range overlap — BD-Rate is undefined, not 0.
        return None

    # VCEG-M33's global cubic polynomial in log-rate as a function of quality.
    p1 = np.polyfit(Q1, lR1, min(3, len(Q1) - 1))
    p2 = np.polyfit(Q2, lR2, min(3, len(Q2) - 1))

    # Integrate
    int1 = np.polyint(p1)
    int2 = np.polyint(p2)

    area1 = np.polyval(int1, q_max) - np.polyval(int1, q_min)
    area2 = np.polyval(int2, q_max) - np.polyval(int2, q_min)

    avg_diff = (area2 - area1) / (q_max - q_min)
    return float((np.exp(avg_diff) - 1) * 100)


class BDRateModule(BatchMetricModule):
    name = "bd_rate"
    provenance = {
        "bd_rate": "published",
    }
    sources = {
        "bd_rate": "Bjøntegaard, VCEG-M33 (2001) — https://www.itu.int/wftp3/av-arch/video-site/0104_Aus/VCEG-M33.doc",
    }
    description = "BD-Rate codec comparison (dataset-level, negative%=better)"
    default_config = {
        "quality_metric": "vmaf",  # Higher-is-better quality axis for both codecs
        # Baseline codec curve. Items are [bitrate, quality] pairs or mappings
        # with ``bitrate`` and ``quality`` keys. Candidate points come from
        # processed samples. VCEG-M33 needs four points per curve.
        "reference_curve": [],
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.quality_metric = self.config.get("quality_metric", "vmaf")
        self._rate_quality_pairs: List[tuple] = []
        self._backend = "algorithmic"

    def extract_features(self, sample: Sample) -> Optional[object]:
        """Extract (bitrate, quality) pair from sample."""
        if sample.video_metadata is None:
            return None

        bitrate = sample.video_metadata.bitrate
        if bitrate is None or bitrate <= 0:
            return None

        quality = None
        if sample.quality_metrics:
            quality = getattr(sample.quality_metrics, self.quality_metric, None)

        if quality is None:
            return None

        return (float(bitrate), float(quality))

    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> Optional[float]:
        """Compute BD-Rate between feature sets.

        features: [(bitrate, quality), ...] from test codec
        reference_features: [(bitrate, quality), ...] from reference codec

        Returns ``None`` when there is no reference codec curve (or too few
        rate-quality points), because BD-Rate is undefined in that case.
        """
        if reference_features is None:
            reference_features = self._configured_reference_curve()
        if len(features) < 4 or len(reference_features) < 4:
            return None

        rates = [f[0] for f in features]
        qualities = [f[1] for f in features]
        ref_rates = [f[0] for f in reference_features]
        ref_qualities = [f[1] for f in reference_features]

        return _bd_rate(ref_rates, ref_qualities, rates, qualities)

    def _configured_reference_curve(self) -> List[tuple]:
        """Parse the explicit baseline curve supplied in module configuration."""
        parsed = []
        for point in self.config.get("reference_curve", []):
            try:
                if isinstance(point, dict):
                    bitrate, quality = point["bitrate"], point["quality"]
                else:
                    bitrate, quality = point
                bitrate, quality = float(bitrate), float(quality)
            except (KeyError, TypeError, ValueError):
                logger.warning("bd_rate: invalid reference-curve point %r", point)
                return []
            if bitrate <= 0 or not np.isfinite(bitrate) or not np.isfinite(quality):
                logger.warning("bd_rate: invalid reference-curve point %r", point)
                return []
            parsed.append((bitrate, quality))
        return parsed

    def process(self, sample: Sample) -> Sample:
        """Accumulate rate-quality pairs."""
        feat = self.extract_features(sample)
        if feat is not None:
            self._feature_cache.append(feat)
        return sample

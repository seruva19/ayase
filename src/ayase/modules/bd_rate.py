"""BD-Rate (Bjøntegaard Delta Rate) module.

BD-Rate measures the average bitrate difference between two codecs
at equivalent quality. It is the industry standard for codec comparison.

BD-Rate: % (negative = better compression, e.g., -20% means 20% bitrate savings)
BD-PSNR: dB (positive = better quality at same bitrate)

This is a dataset-level metric that requires rate-quality curves from
multiple encoding points.

Stores results in DatasetStats.bd_rate and DatasetStats.bd_psnr.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


def _bd_rate(rate1, quality1, rate2, quality2) -> float:
    """Compute BD-Rate using piecewise cubic interpolation.

    Based on JCTVC-L1100 (Bjøntegaard 2001, updated by Pateux/Jung 2007).

    Args:
        rate1, quality1: Arrays of (bitrate, quality) for codec 1
        rate2, quality2: Arrays of (bitrate, quality) for codec 2

    Returns:
        BD-Rate in percentage (negative = codec2 is better)
    """
    lR1 = np.log(np.array(rate1))
    lR2 = np.log(np.array(rate2))
    Q1 = np.array(quality1)
    Q2 = np.array(quality2)

    # Need at least 4 points for cubic interpolation
    if len(Q1) < 4 or len(Q2) < 4:
        # Linear fallback
        avg_r1 = float(np.mean(lR1))
        avg_r2 = float(np.mean(lR2))
        return float((np.exp(avg_r2 - avg_r1) - 1) * 100)

    # Sort by quality
    idx1 = np.argsort(Q1)
    idx2 = np.argsort(Q2)
    Q1, lR1 = Q1[idx1], lR1[idx1]
    Q2, lR2 = Q2[idx2], lR2[idx2]

    # Overlap range
    q_min = max(Q1[0], Q2[0])
    q_max = min(Q1[-1], Q2[-1])

    if q_min >= q_max:
        return 0.0

    # Piecewise cubic interpolation
    from numpy.polynomial import polynomial as P

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
    description = "BD-Rate codec comparison (dataset-level, negative%=better)"
    default_config = {
        "quality_metric": "psnr",  # Which quality metric to use
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.quality_metric = self.config.get("quality_metric", "psnr")
        self._rate_quality_pairs: List[tuple] = []

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
    ) -> float:
        """Compute BD-Rate between feature sets.

        features: [(bitrate, quality), ...] from test codec
        reference_features: [(bitrate, quality), ...] from reference codec
        """
        if reference_features is None or len(features) < 2 or len(reference_features) < 2:
            return 0.0

        rates = [f[0] for f in features]
        qualities = [f[1] for f in features]
        ref_rates = [f[0] for f in reference_features]
        ref_qualities = [f[1] for f in reference_features]

        return _bd_rate(ref_rates, ref_qualities, rates, qualities)

    def process(self, sample: Sample) -> Sample:
        """Accumulate rate-quality pairs."""
        feat = self.extract_features(sample)
        if feat is not None:
            self._feature_cache.append(feat)
        return sample

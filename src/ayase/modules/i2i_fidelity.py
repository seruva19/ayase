"""Actionable deterministic image-to-image fidelity diagnostics.

Compares an image with ``sample.reference_path`` and reports a compact set of
non-interchangeable pixel, color, structure, frequency, and information
signals. Derived transformations and arbitrary threshold variants are omitted.
"""

import logging
from pathlib import Path

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

ALL_FIELDS = (
    "i2i_mse",
    "i2i_mae",
    "i2i_gradient_similarity_mean",
)


class I2IFidelityModule(PipelineModule):
    """Compute 19 complementary full-reference diagnostics for an image pair."""

    name = "i2i_fidelity"
    provenance = {
        "i2i_gradient_similarity_mean": "published",
        "i2i_mae": "published",
        "i2i_mse": "published",
    }
    sources = {
        "i2i_gradient_similarity_mean": "GMSM (Xue et al., IEEE TIP 2014, DOI:10.1109/TIP.2013.2293423)",
        "i2i_mae": "standard MAE definition (https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm)",
        "i2i_mse": "standard MSE definition (https://www.itl.nist.gov/div898/handbook/eda/section3/eda3661.htm)",
    }
    deviations = {}
    description = "Pixel-level MSE/MAE plus published GMSM gradient similarity (FR)"
    default_config = {}
    metric_info = {
        "i2i_mse": "Mean squared RGB error; emphasizes large pixel deviations",
        "i2i_mae": "Mean absolute RGB error; robust aggregate pixel deviation",
        "i2i_gradient_similarity_mean": "Mean gradient-magnitude similarity",
    }
    metric_groups = {field: "fr_quality" for field in ALL_FIELDS}

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None

    def _compute(self, ref: np.ndarray, gen: np.ndarray) -> dict:
        ref_float = ref.astype(np.float64) / 255.0
        gen_float = gen.astype(np.float64) / 255.0
        error = gen_float - ref_float
        absolute = np.abs(error)
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)

        # GMSM (Xue et al., TIP 2014): Prewitt gradient magnitudes on the
        # 2x-downsampled luminance planes, similarity constant c=170 on [0,255].
        ref_g = ref_gray.astype(np.float32)
        gen_g = gen_gray.astype(np.float32)
        ref_g = cv2.resize(ref_g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        gen_g = cv2.resize(gen_g, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32) / 3.0
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32) / 3.0
        ref_magnitude = cv2.magnitude(
            cv2.filter2D(ref_g, cv2.CV_32F, prewitt_x),
            cv2.filter2D(ref_g, cv2.CV_32F, prewitt_y),
        )
        gen_magnitude = cv2.magnitude(
            cv2.filter2D(gen_g, cv2.CV_32F, prewitt_x),
            cv2.filter2D(gen_g, cv2.CV_32F, prewitt_y),
        )
        gradient_similarity = (
            2.0 * ref_magnitude * gen_magnitude + 170.0
        ) / (ref_magnitude**2 + gen_magnitude**2 + 170.0)

        return {
            "i2i_mse": float(np.mean(error**2)),
            "i2i_mae": float(absolute.mean()),
            "i2i_gradient_similarity_mean": float(gradient_similarity.mean()),
        }

    @staticmethod
    def _store(sample: Sample, metrics: dict) -> None:
        qm = sample.quality_metrics
        qm.i2i_mse = metrics["i2i_mse"]
        qm.i2i_mae = metrics["i2i_mae"]
        qm.i2i_gradient_similarity_mean = metrics["i2i_gradient_similarity_mean"]

    def process(self, sample: Sample) -> Sample:
        reference = getattr(sample, "reference_path", None)
        if reference is None or sample.is_video:
            return sample
        reference = Path(reference)
        if not reference.is_file() or not sample.path.is_file():
            return sample
        try:
            ref = cv2.imread(str(reference), cv2.IMREAD_COLOR)
            gen = cv2.imread(str(sample.path), cv2.IMREAD_COLOR)
            if ref is None or gen is None:
                return sample
            if gen.shape[:2] != ref.shape[:2]:
                gen = cv2.resize(gen, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
            metrics = self._compute(ref, gen)
            if set(metrics) != set(ALL_FIELDS) or not all(
                np.isfinite(value) for value in metrics.values()
            ):
                logger.warning("i2i_fidelity produced invalid output")
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            self._store(sample, metrics)
            self._backend = "opencv_numpy"
        except Exception as exc:
            logger.warning("i2i_fidelity failed for %s: %s", sample.path, exc)
        return sample

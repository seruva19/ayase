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
    "i2i_mean_bias",
    "i2i_exact_match_ratio",
    "i2i_red_bias",
    "i2i_green_bias",
    "i2i_blue_bias",
    "i2i_luminance_mae",
    "i2i_chroma_cr_mae",
    "i2i_chroma_cb_mae",
    "i2i_hue_mae_degrees",
    "i2i_colorfulness_delta",
    "i2i_hist_bhattacharyya_red",
    "i2i_hist_bhattacharyya_green",
    "i2i_hist_bhattacharyya_blue",
    "i2i_gradient_similarity_mean",
    "i2i_edge_f1",
    "i2i_spectral_cosine",
    "i2i_mutual_information",
)


class I2IFidelityModule(PipelineModule):
    """Compute 19 complementary full-reference diagnostics for an image pair."""

    name = "i2i_fidelity"
    description = "19 actionable pixel, color, structure, frequency, and information I2I metrics"
    default_config = {
        "histogram_bins": 64,
        "edge_threshold_low": 100,
        "edge_threshold_high": 200,
    }
    metric_info = {
        "i2i_mse": "Mean squared RGB error; emphasizes large pixel deviations",
        "i2i_mae": "Mean absolute RGB error; robust aggregate pixel deviation",
        "i2i_mean_bias": "Signed aggregate RGB bias",
        "i2i_exact_match_ratio": "Fraction of pixels preserved exactly",
        "i2i_red_bias": "Signed red-channel bias",
        "i2i_green_bias": "Signed green-channel bias",
        "i2i_blue_bias": "Signed blue-channel bias",
        "i2i_luminance_mae": "Mean absolute luminance error",
        "i2i_chroma_cr_mae": "Mean absolute red-chroma error",
        "i2i_chroma_cb_mae": "Mean absolute blue-chroma error",
        "i2i_hue_mae_degrees": "Circular mean absolute hue error in degrees",
        "i2i_colorfulness_delta": "Absolute Hasler-Suesstrunk colorfulness change",
        "i2i_hist_bhattacharyya_red": "Bhattacharyya distance between red histograms",
        "i2i_hist_bhattacharyya_green": "Bhattacharyya distance between green histograms",
        "i2i_hist_bhattacharyya_blue": "Bhattacharyya distance between blue histograms",
        "i2i_gradient_similarity_mean": "Mean gradient-magnitude similarity",
        "i2i_edge_f1": "F1 overlap of Canny edge maps",
        "i2i_spectral_cosine": "Cosine similarity of log Fourier magnitudes",
        "i2i_mutual_information": "Mutual information between luminance images",
    }
    metric_groups = {field: "fr_quality" for field in ALL_FIELDS}

    def __init__(self, config=None):
        super().__init__(config)
        self.histogram_bins = max(8, int(self.config.get("histogram_bins", 64)))
        self.edge_low = int(self.config.get("edge_threshold_low", 100))
        self.edge_high = int(self.config.get("edge_threshold_high", 200))
        self._backend = None

    @staticmethod
    def _colorfulness(image: np.ndarray) -> float:
        b, g, r = cv2.split(image.astype(np.float32))
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))

    def _hist_distance(self, ref: np.ndarray, gen: np.ndarray, channel: int) -> float:
        ref_hist = cv2.calcHist([ref], [channel], None, [self.histogram_bins], [0, 256])
        gen_hist = cv2.calcHist([gen], [channel], None, [self.histogram_bins], [0, 256])
        cv2.normalize(ref_hist, ref_hist, alpha=1.0, norm_type=cv2.NORM_L1)
        cv2.normalize(gen_hist, gen_hist, alpha=1.0, norm_type=cv2.NORM_L1)
        return float(cv2.compareHist(ref_hist, gen_hist, cv2.HISTCMP_BHATTACHARYYA))

    @staticmethod
    def _spectral_cosine(ref_gray: np.ndarray, gen_gray: np.ndarray) -> float:
        ref = np.log1p(np.abs(np.fft.fft2(ref_gray.astype(np.float32))))
        gen = np.log1p(np.abs(np.fft.fft2(gen_gray.astype(np.float32))))
        denominator = float(np.linalg.norm(ref) * np.linalg.norm(gen))
        if denominator == 0:
            return 1.0 if np.allclose(ref, gen) else 0.0
        return float(np.clip(np.sum(ref * gen) / denominator, -1.0, 1.0))

    @staticmethod
    def _mutual_information(ref_gray: np.ndarray, gen_gray: np.ndarray) -> float:
        joint = np.histogram2d(
            ref_gray.ravel(), gen_gray.ravel(), bins=64, range=((0, 256), (0, 256))
        )[0]
        joint /= max(float(joint.sum()), 1.0)
        independent = joint.sum(axis=1)[:, None] * joint.sum(axis=0)[None, :]
        nonzero = joint > 0
        return float(np.sum(joint[nonzero] * np.log2(joint[nonzero] / independent[nonzero])))

    def _compute(self, ref: np.ndarray, gen: np.ndarray) -> dict:
        ref_float = ref.astype(np.float64) / 255.0
        gen_float = gen.astype(np.float64) / 255.0
        error = gen_float - ref_float
        absolute = np.abs(error)
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        gen_gray = cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)
        ref_ycc = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)
        gen_ycc = cv2.cvtColor(gen, cv2.COLOR_BGR2YCrCb)
        ref_hsv = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
        gen_hsv = cv2.cvtColor(gen, cv2.COLOR_BGR2HSV)
        hue_delta = np.abs(ref_hsv[..., 0].astype(float) - gen_hsv[..., 0].astype(float))
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta) * 2.0

        ref_f = ref_gray.astype(np.float32)
        gen_f = gen_gray.astype(np.float32)
        ref_x = cv2.Sobel(ref_f, cv2.CV_32F, 1, 0)
        ref_y = cv2.Sobel(ref_f, cv2.CV_32F, 0, 1)
        gen_x = cv2.Sobel(gen_f, cv2.CV_32F, 1, 0)
        gen_y = cv2.Sobel(gen_f, cv2.CV_32F, 0, 1)
        ref_magnitude = cv2.magnitude(ref_x, ref_y)
        gen_magnitude = cv2.magnitude(gen_x, gen_y)
        gradient_similarity = (
            2.0 * ref_magnitude * gen_magnitude + 170.0
        ) / (ref_magnitude**2 + gen_magnitude**2 + 170.0)

        ref_edges = cv2.Canny(ref_gray, self.edge_low, self.edge_high) > 0
        gen_edges = cv2.Canny(gen_gray, self.edge_low, self.edge_high) > 0
        intersection = int(np.logical_and(ref_edges, gen_edges).sum())
        precision = intersection / max(int(gen_edges.sum()), 1)
        recall = intersection / max(int(ref_edges.sum()), 1)

        channel_error = gen.astype(np.float32) - ref.astype(np.float32)
        return {
            "i2i_mse": float(np.mean(error**2)),
            "i2i_mae": float(absolute.mean()),
            "i2i_mean_bias": float(error.mean()),
            "i2i_exact_match_ratio": float(np.mean(np.all(ref == gen, axis=2))),
            "i2i_red_bias": float(np.mean(channel_error[..., 2]) / 255.0),
            "i2i_green_bias": float(np.mean(channel_error[..., 1]) / 255.0),
            "i2i_blue_bias": float(np.mean(channel_error[..., 0]) / 255.0),
            "i2i_luminance_mae": float(
                np.mean(np.abs(gen_ycc[..., 0].astype(float) - ref_ycc[..., 0])) / 255.0
            ),
            "i2i_chroma_cr_mae": float(
                np.mean(np.abs(gen_ycc[..., 1].astype(float) - ref_ycc[..., 1])) / 255.0
            ),
            "i2i_chroma_cb_mae": float(
                np.mean(np.abs(gen_ycc[..., 2].astype(float) - ref_ycc[..., 2])) / 255.0
            ),
            "i2i_hue_mae_degrees": float(hue_delta.mean()),
            "i2i_colorfulness_delta": abs(self._colorfulness(gen) - self._colorfulness(ref)),
            "i2i_hist_bhattacharyya_red": self._hist_distance(ref, gen, 2),
            "i2i_hist_bhattacharyya_green": self._hist_distance(ref, gen, 1),
            "i2i_hist_bhattacharyya_blue": self._hist_distance(ref, gen, 0),
            "i2i_gradient_similarity_mean": float(gradient_similarity.mean()),
            "i2i_edge_f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
            "i2i_spectral_cosine": self._spectral_cosine(ref_gray, gen_gray),
            "i2i_mutual_information": self._mutual_information(ref_gray, gen_gray),
        }

    @staticmethod
    def _store(sample: Sample, metrics: dict) -> None:
        qm = sample.quality_metrics
        qm.i2i_mse = metrics["i2i_mse"]
        qm.i2i_mae = metrics["i2i_mae"]
        qm.i2i_mean_bias = metrics["i2i_mean_bias"]
        qm.i2i_exact_match_ratio = metrics["i2i_exact_match_ratio"]
        qm.i2i_red_bias = metrics["i2i_red_bias"]
        qm.i2i_green_bias = metrics["i2i_green_bias"]
        qm.i2i_blue_bias = metrics["i2i_blue_bias"]
        qm.i2i_luminance_mae = metrics["i2i_luminance_mae"]
        qm.i2i_chroma_cr_mae = metrics["i2i_chroma_cr_mae"]
        qm.i2i_chroma_cb_mae = metrics["i2i_chroma_cb_mae"]
        qm.i2i_hue_mae_degrees = metrics["i2i_hue_mae_degrees"]
        qm.i2i_colorfulness_delta = metrics["i2i_colorfulness_delta"]
        qm.i2i_hist_bhattacharyya_red = metrics["i2i_hist_bhattacharyya_red"]
        qm.i2i_hist_bhattacharyya_green = metrics["i2i_hist_bhattacharyya_green"]
        qm.i2i_hist_bhattacharyya_blue = metrics["i2i_hist_bhattacharyya_blue"]
        qm.i2i_gradient_similarity_mean = metrics["i2i_gradient_similarity_mean"]
        qm.i2i_edge_f1 = metrics["i2i_edge_f1"]
        qm.i2i_spectral_cosine = metrics["i2i_spectral_cosine"]
        qm.i2i_mutual_information = metrics["i2i_mutual_information"]

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

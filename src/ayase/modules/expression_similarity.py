"""Facial-expression similarity between two videos of a person, without time alignment.

Answers a different question from ``expression_following``. That metric asks
whether a generation reproduced the driver's expression *frame by frame*, so it
needs both videos to be the same performance on a common clock. This one asks
whether the person in one video emotes *like* the person in the other: same
repertoire, same co-activations, same tempo. The two clips need not be the same
take, the same length, or even the same speech -- there is no temporal alignment
anywhere in this metric, and a time-shifted copy of a clip scores as its twin.

Features are the same 52 ARKit-style MediaPipe blendshape coefficients used by
``expression_following``, read through the shared extractor so the two metrics
cannot drift apart. Each clip is reduced to three time-free descriptions:

* **distribution** -- per-coefficient quantiles: which expressions occur and how
  often. Compared as a 1-D Wasserstein distance, averaged over coefficients.
* **co-activation** -- the coefficient-by-coefficient correlation matrix: what
  moves together. This is the part that carries personal manner rather than
  momentary mood; the idea is the person-specific behavioural signature used to
  spot forged videos of public figures (Agarwal and Farid, CVPRW 2019).
* **dynamics** -- mean absolute change per second per coefficient: whether the
  face moves at the same rate at all.

Reported alongside them is ``expression_similarity_range_ratio``, the ratio of
overall expressive spread (sample over reference). It is a diagnosis, not a
score: a value well below 1 is the signature of a face that stayed still, which
is the usual failure of an over-trained identity adapter and is invisible to
identity metrics -- face-recognition embeddings are trained to ignore expression.

Higher is better for every score, all bounded to 0-1. What the metric cannot do:
it will not separate manner from content when the two clips differ in kind (a
speech against a silent take moves the mouth differently no matter who is in
frame), and, like any blendshape comparison, it carries residual framing signal,
so compare clips of similar crop.

Measured behaviour, on speech video of two speakers, 12 clips each, where the
clips of one speaker span many different shots so that framing varies within a
speaker as well as between speakers.

Asked to pick the closest of several candidates, which is what the metric is for:

* nearest of 23 candidates is the same speaker -- 95.8% of queries;
* pick the matching clip out of three (one match, two impostors) -- 80.9%,
  against 33.3% for guessing;
* out of five -- 74.1%, against 20%.

Asked instead to order pairs, same-speaker pairs outrank different-speaker pairs
with AUC 0.84. The components contribute unequally: distribution 0.93, dynamics
0.79, co-activation 0.67.

What follows for use: the absolute scores sit close together (median 0.79 for
same-speaker pairs against 0.75 for different-speaker ones), so the metric is
strong at *ranking* candidates and weak as an absolute verdict. Compare
candidates against a common reference and take the ordering; do not attach a
fixed pass/fail threshold to a single score. Scores are also comparable only
within one comparison: material recorded under different conditions shifts them,
so candidates ranked against a shared reference must be alike in framing and
capture.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

from ._blendshape_utils import (
    BLENDSHAPE_DIM,
    CANONICAL_BLENDSHAPES,
    GAZE_BLENDSHAPES,
    MODEL_FILENAME,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_URL,
    BlendshapeExtractor,
    BlendshapeTrajectory,
)

logger = logging.getLogger(__name__)

# A coefficient whose spread over the clip is below this never activated in any
# meaningful way; correlating it with anything yields noise amplified to unit
# scale, so it is excluded from the co-activation comparison.
ACTIVITY_EPS = 1e-3

__all__ = ["ExpressionSimilarity", "ExpressionSimilarityModule"]


class ExpressionSimilarityModule(PipelineModule):
    """Compare how two videos of a person emote, without aligning them in time.

    Produces one composite score and three components -- distribution,
    co-activation and dynamics -- plus a range ratio that exposes expression
    collapse. Requires a reference video in ``sample.reference_path``.
    """

    name = "expression_similarity"
    description = "Time-free facial-expression manner similarity via MediaPipe blendshapes"
    default_config = {
        "models_dir": "models",
        "min_face_detection_confidence": 0.5,
        "min_face_presence_confidence": 0.5,
        "min_tracking_confidence": 0.5,
        "low_coverage_threshold": 0.5,
        "min_valid_frames": 15,
        "quantile_count": 21,
        "exclude_gaze": False,
        "num_faces": 5,
        "face_index": None,
    }
    metric_groups = {
        "expression_similarity": "face",
        "expression_similarity_distribution": "face",
        "expression_similarity_coactivation": "face",
        "expression_similarity_dynamics": "face",
        "expression_similarity_range_ratio": "face",
        "expression_similarity_coverage": "face",
    }
    models = [
        {
            "id": MODEL_REPO_ID,
            "type": "huggingface",
            "task": "MediaPipe Face Landmarker ARKit-style blendshapes",
            "url": MODEL_URL,
            "auto_download": "yes",
            "notes": f"Pinned revision {MODEL_REVISION}; file {MODEL_FILENAME}",
        }
    ]
    metric_info = {
        "expression_similarity": "Composite expression-manner similarity (0-1, higher=better)",
        "expression_similarity_distribution": "Expression-repertoire agreement, Wasserstein-based (0-1)",
        "expression_similarity_coactivation": "Agreement of coefficient correlation structure (0-1)",
        "expression_similarity_dynamics": "Agreement of expression change rate (0-1)",
        "expression_similarity_range_ratio": "Expressive spread, sample over reference (1.0 = equal)",
        "expression_similarity_coverage": "Lower of the two per-video valid-face coverages (0-1)",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self.models_dir = str(self.config.get("models_dir", "models"))
        self.low_coverage_threshold = float(self.config.get("low_coverage_threshold", 0.5))
        self.min_valid_frames = max(2, int(self.config.get("min_valid_frames", 15)))
        self.quantile_count = max(3, int(self.config.get("quantile_count", 21)))
        self.exclude_gaze = bool(self.config.get("exclude_gaze", False))
        self.num_faces = max(1, int(self.config.get("num_faces", 5)))
        self.face_index = self.config.get("face_index")
        self._extractor = BlendshapeExtractor(
            self.models_dir,
            num_faces=self.num_faces,
            face_index=self.face_index,
            min_face_detection_confidence=float(
                self.config.get("min_face_detection_confidence", 0.5)
            ),
            min_face_presence_confidence=float(
                self.config.get("min_face_presence_confidence", 0.5)
            ),
            min_tracking_confidence=float(self.config.get("min_tracking_confidence", 0.5)),
        )
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        self._ml_available = self._extractor.setup("ExpressionSimilarity")
        self._backend = self._extractor.backend

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        if sample.reference_path is None:
            self._input_issue(sample, "A reference video is required")
            return sample
        reference = Path(sample.reference_path)
        if not reference.is_file():
            self._input_issue(sample, "Reference video is missing or unreadable")
            return sample

        try:
            generation = self._extractor.extract(Path(sample.path))
            target = self._extractor.extract(reference)
            result = self.compare_trajectories(generation, target)
            self._store_result(sample, result)
        except Exception as e:  # noqa: BLE001 - a module must degrade, not raise
            logger.warning("ExpressionSimilarity failed for %s: %s", sample.path, e)
            sample.metadata["expression_similarity_error"] = f"{type(e).__name__}: {e}"
        return sample

    @staticmethod
    def _input_issue(sample: Sample, message: str) -> None:
        sample.validation_issues.append(
            ValidationIssue(severity=ValidationSeverity.WARNING, message=message)
        )

    def _channel_mask(self) -> np.ndarray:
        """Which of the 52 coefficients take part. Gaze may be dropped on request."""
        mask = np.ones(BLENDSHAPE_DIM, dtype=bool)
        if self.exclude_gaze:
            for index, name in enumerate(CANONICAL_BLENDSHAPES):
                if name in GAZE_BLENDSHAPES:
                    mask[index] = False
        return mask

    def _distribution_score(self, left: np.ndarray, right: np.ndarray) -> float:
        """1-D Wasserstein distance per coefficient, averaged, turned into a score.

        Quantiles at equally spaced levels are the empirical inverse CDF, so the
        mean absolute difference between two quantile vectors is the 1-Wasserstein
        distance between the two samples. It needs no shared clock and no equal
        clip length, which is the whole point here.
        """
        levels = np.linspace(0.0, 1.0, self.quantile_count)
        left_q = np.quantile(left, levels, axis=0)
        right_q = np.quantile(right, levels, axis=0)
        distance = float(np.mean(np.abs(left_q - right_q)))
        return float(np.clip(1.0 - distance, 0.0, 1.0))

    @staticmethod
    def _correlation(values: np.ndarray, active: np.ndarray) -> np.ndarray:
        centred = values[:, active] - values[:, active].mean(axis=0, keepdims=True)
        scale = np.sqrt(np.sum(centred**2, axis=0))
        scale[scale == 0.0] = 1.0
        normalised = centred / scale
        return np.asarray(normalised.T @ normalised, dtype=np.float64)

    def _coactivation_score(
        self, left: np.ndarray, right: np.ndarray, active: np.ndarray
    ) -> Optional[float]:
        """Compare correlation structure over coefficients active in BOTH clips.

        Restricting to the shared active set is what keeps the number honest: a
        coefficient that never moved in one clip has no correlation to compare,
        and letting it in would score detector noise.
        """
        if int(active.sum()) < 2:
            return None
        left_corr = self._correlation(left, active)
        right_corr = self._correlation(right, active)
        upper = np.triu_indices(int(active.sum()), k=1)
        # Correlations live in [-1, 1], so a difference spans [0, 2].
        distance = float(np.mean(np.abs(left_corr[upper] - right_corr[upper])) / 2.0)
        return float(np.clip(1.0 - distance, 0.0, 1.0))

    @staticmethod
    def _speed(values: np.ndarray, fps: float) -> np.ndarray:
        """Mean absolute change per second, per coefficient."""
        if values.shape[0] < 2:
            return np.zeros(values.shape[1], dtype=np.float64)
        return np.asarray(np.mean(np.abs(np.diff(values, axis=0)), axis=0) * float(fps))

    @staticmethod
    def _ratio_agreement(left: np.ndarray, right: np.ndarray) -> float:
        """Per-coefficient min/max agreement; two quiet coefficients agree."""
        agreements = []
        for a, b in zip(left, right):
            top = max(float(a), float(b))
            if top <= ACTIVITY_EPS:
                agreements.append(1.0)
            else:
                agreements.append(min(float(a), float(b)) / top)
        return float(np.clip(np.mean(agreements), 0.0, 1.0)) if agreements else 0.0

    def compare_trajectories(
        self, generation: BlendshapeTrajectory, reference: BlendshapeTrajectory
    ) -> Dict[str, Any]:
        """Score two trajectories against each other. Public so it can be tested directly."""
        flags: List[str] = []
        gen_values = generation.valid_coefficients().astype(np.float64)
        ref_values = reference.valid_coefficients().astype(np.float64)
        gen_coverage = self._coverage(generation)
        ref_coverage = self._coverage(reference)
        coverage = min(gen_coverage, ref_coverage)

        if gen_values.shape[0] == 0:
            flags.append("no_face_generation")
        if ref_values.shape[0] == 0:
            flags.append("no_face_reference")
        if min(gen_values.shape[0], ref_values.shape[0]) < self.min_valid_frames:
            flags.append("too_few_valid_frames")
            return self._result(None, None, None, None, None, coverage, gen_coverage,
                                ref_coverage, generation, reference, flags, 0)
        if coverage < self.low_coverage_threshold:
            flags.append("low_face_visibility")
        if generation.multiple_faces:
            flags.append("multiple_faces_generation")
        if reference.multiple_faces:
            flags.append("multiple_faces_reference")

        mask = self._channel_mask()
        gen_masked = gen_values[:, mask]
        ref_masked = ref_values[:, mask]

        distribution = self._distribution_score(gen_masked, ref_masked)
        active = (gen_masked.std(axis=0) > ACTIVITY_EPS) & (ref_masked.std(axis=0) > ACTIVITY_EPS)
        coactivation = self._coactivation_score(gen_masked, ref_masked, active)
        if coactivation is None:
            flags.append("too_few_active_coefficients")
        dynamics = self._ratio_agreement(
            self._speed(gen_masked, generation.fps), self._speed(ref_masked, reference.fps)
        )

        gen_spread = float(np.mean(gen_masked.std(axis=0)))
        ref_spread = float(np.mean(ref_masked.std(axis=0)))
        range_ratio = float(gen_spread / ref_spread) if ref_spread > ACTIVITY_EPS else None
        if range_ratio is None:
            flags.append("reference_expression_flat")

        components = [distribution, dynamics]
        if coactivation is not None:
            components.append(coactivation)
        composite = float(np.clip(np.mean(components), 0.0, 1.0))

        return self._result(composite, distribution, coactivation, dynamics, range_ratio,
                            coverage, gen_coverage, ref_coverage, generation, reference,
                            flags, int(active.sum()))

    @staticmethod
    def _coverage(trajectory: BlendshapeTrajectory) -> float:
        if trajectory.decoded_frames <= 0:
            return 0.0
        return float(trajectory.face_frames / trajectory.decoded_frames)

    def _result(
        self,
        composite: Optional[float],
        distribution: Optional[float],
        coactivation: Optional[float],
        dynamics: Optional[float],
        range_ratio: Optional[float],
        coverage: float,
        gen_coverage: float,
        ref_coverage: float,
        generation: BlendshapeTrajectory,
        reference: BlendshapeTrajectory,
        flags: List[str],
        active_coefficients: int,
    ) -> Dict[str, Any]:
        return {
            "expression_similarity": composite,
            "expression_similarity_distribution": distribution,
            "expression_similarity_coactivation": coactivation,
            "expression_similarity_dynamics": dynamics,
            "expression_similarity_range_ratio": range_ratio,
            "expression_similarity_coverage": coverage,
            "expression_similarity_generation_coverage": gen_coverage,
            "expression_similarity_reference_coverage": ref_coverage,
            "expression_similarity_generation_face_frames": generation.face_frames,
            "expression_similarity_reference_face_frames": reference.face_frames,
            "expression_similarity_generation_fps": float(generation.fps),
            "expression_similarity_reference_fps": float(reference.fps),
            "expression_similarity_active_coefficients": active_coefficients,
            "expression_similarity_flags": flags,
            "expression_similarity_feature_type": "mediapipe_arkit_blendshapes_52",
            "expression_similarity_alignment": "none_time_free_statistics",
            "expression_similarity_components": "distribution_coactivation_dynamics_equal_weight",
        }

    @staticmethod
    def _store_result(sample: Sample, result: Dict[str, Any]) -> None:
        sample.metadata.update(result)
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        for field in (
            "expression_similarity",
            "expression_similarity_distribution",
            "expression_similarity_coactivation",
            "expression_similarity_dynamics",
            "expression_similarity_range_ratio",
            "expression_similarity_coverage",
        ):
            setattr(sample.quality_metrics, field, result[field])


# Short alias; the registry-facing convention uses *Module.
ExpressionSimilarity = ExpressionSimilarityModule

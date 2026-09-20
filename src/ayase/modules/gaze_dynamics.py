"""Reference-relative MediaPipe ``eyeLook*`` activation diagnostics.

Intended use: compare the distribution and temporal dynamics of ocular-control
activations in ``sample.path`` with those in ``sample.reference_path`` when both
videos predominantly show one visible face at sufficient eye resolution.  Each
clip is summarized independently; frames, duration, and performed actions need
not correspond.

The inputs are the eight ``eyeLookDown/In/Out/Up`` coefficients emitted by the
pinned MediaPipe Face Landmarker blendshape model.  They are model activation
scores in ``[0, 1]``, not calibrated gaze angles, eye-contact estimates, or measurements in
physical units.  The diagnostics are not an identity, person-similarity,
perceptual-quality, or behavioral-trait score.  Occlusion, glasses, eye-region
resolution, head pose, lighting, camera viewpoint, and domain mismatch can
change the activations.  Multi-person footage is outside the intended use;
selecting one face is not identity tracking.

Horizontal and vertical controls use paired in/out and up/down activation
differences for the two eyes.  Location is the median binocular-average
control, amplitude is its 10th-to-90th-percentile span, speed is the median
Euclidean control change per second over adjacent valid decoded frames, and
binocular disagreement is the median Euclidean difference between the two eye
controls.  Outputs are absolute generated/reference differences (0 means the
corresponding summaries agree) plus the two per-clip valid-activation coverages.
Missing detections break speed runs.  No frame-aligned output or unvalidated
aggregate is produced.

Backend and coefficient definitions: Google MediaPipe Face Landmarker,
https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker/python
and its documented 52 face blendshape categories.  Intended uses and limits are
documented in Google's Blendshape V2 model card:
https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Blendshape%20V2.pdf
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

from ._blendshape_utils import (
    CANONICAL_BLENDSHAPES,
    MODEL_FILENAME,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_URL,
    BlendshapeExtractor,
    BlendshapeTrajectory,
)

logger = logging.getLogger(__name__)

_INDEX = {name: index for index, name in enumerate(CANONICAL_BLENDSHAPES)}
_REQUIRED = tuple(
    _INDEX[name]
    for name in (
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
    )
)


@dataclass(frozen=True)
class _OcularSummary:
    """Interpretable summaries of one clip's MediaPipe eye-look controls."""

    horizontal_location: float
    vertical_location: float
    horizontal_amplitude: float
    vertical_amplitude: float
    speed: Optional[float]
    binocular_disagreement: float


def _ocular_controls(coefficients: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return left/right ``(horizontal, vertical)`` signed controls.

    Positive horizontal is the module's explicit common-direction convention:
    ``out-in`` for the left eye and ``in-out`` for the right eye.  Positive
    vertical is ``up-down`` for both eyes.  These are activation differences,
    not angles.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    left = np.column_stack(
        (
            values[:, _INDEX["eyeLookOutLeft"]]
            - values[:, _INDEX["eyeLookInLeft"]],
            values[:, _INDEX["eyeLookUpLeft"]]
            - values[:, _INDEX["eyeLookDownLeft"]],
        )
    )
    right = np.column_stack(
        (
            values[:, _INDEX["eyeLookInRight"]]
            - values[:, _INDEX["eyeLookOutRight"]],
            values[:, _INDEX["eyeLookUpRight"]]
            - values[:, _INDEX["eyeLookDownRight"]],
        )
    )
    return left, right


def summarize_ocular_trajectory(
    trajectory: BlendshapeTrajectory, min_samples: int = 8
) -> Tuple[Optional[_OcularSummary], float]:
    """Summarize one decoded trajectory and return its effective coverage.

    Rows with malformed or out-of-range required coefficients are treated as
    unavailable.  Speed uses only adjacent decoded frames that are valid on
    both sides, so a missed detection never creates an artificial transition.
    """
    coefficients = np.asarray(trajectory.coefficients, dtype=np.float64)
    timestamps = np.asarray(trajectory.timestamps_sec, dtype=np.float64)
    frame_indices = np.asarray(trajectory.frame_indices)
    declared_valid = np.asarray(trajectory.valid, dtype=bool)
    count = int(timestamps.size)
    if (
        count == 0
        or coefficients.ndim != 2
        or coefficients.shape[0] != count
        or coefficients.shape[1] != len(CANONICAL_BLENDSHAPES)
        or frame_indices.shape != (count,)
        or declared_valid.shape != (count,)
    ):
        return None, 0.0

    required = coefficients[:, _REQUIRED]
    finite = np.isfinite(required).all(axis=1)
    in_range = ((required >= -1e-5) & (required <= 1.0 + 1e-5)).all(axis=1)
    effective_valid = declared_valid & finite & in_range & np.isfinite(timestamps)
    coverage = float(np.count_nonzero(effective_valid) / count)
    minimum = max(2, int(min_samples))
    if np.count_nonzero(effective_valid) < minimum:
        return None, coverage

    left, right = _ocular_controls(coefficients)
    binocular = (left + right) * 0.5
    usable = binocular[effective_valid]
    low, high = np.percentile(usable, (10.0, 90.0), axis=0)
    disagreement = np.linalg.norm(left[effective_valid] - right[effective_valid], axis=1)

    transitions = (
        effective_valid[:-1]
        & effective_valid[1:]
        & (np.diff(frame_indices) == 1)
    )
    dt = np.diff(timestamps)
    transitions &= np.isfinite(dt) & (dt > 0.0)
    speed: Optional[float] = None
    if np.any(transitions):
        delta = np.diff(binocular, axis=0)[transitions]
        speeds = np.linalg.norm(delta, axis=1) / dt[transitions]
        speeds = speeds[np.isfinite(speeds)]
        if speeds.size:
            speed = float(np.median(speeds))

    summary = _OcularSummary(
        horizontal_location=float(np.median(usable[:, 0])),
        vertical_location=float(np.median(usable[:, 1])),
        horizontal_amplitude=float(high[0] - low[0]),
        vertical_amplitude=float(high[1] - low[1]),
        speed=speed,
        binocular_disagreement=float(np.median(disagreement)),
    )
    if not all(
        math.isfinite(value)
        for value in (
            summary.horizontal_location,
            summary.vertical_location,
            summary.horizontal_amplitude,
            summary.vertical_amplitude,
            summary.binocular_disagreement,
        )
    ):
        return None, coverage
    return summary, coverage


def compare_ocular_trajectories(
    sample_track: BlendshapeTrajectory,
    reference_track: BlendshapeTrajectory,
    min_samples: int = 8,
) -> Dict[str, Optional[float]]:
    """Compare unaligned per-clip summaries without inventing an aggregate."""
    sample_summary, sample_coverage = summarize_ocular_trajectory(
        sample_track, min_samples
    )
    reference_summary, reference_coverage = summarize_ocular_trajectory(
        reference_track, min_samples
    )
    result: Dict[str, Optional[float]] = {
        "gaze_blendshape_horizontal_location_difference": None,
        "gaze_blendshape_vertical_location_difference": None,
        "gaze_blendshape_horizontal_amplitude_difference": None,
        "gaze_blendshape_vertical_amplitude_difference": None,
        "gaze_blendshape_speed_difference": None,
        "gaze_blendshape_binocular_disagreement_difference": None,
        "gaze_blendshape_sample_coverage": sample_coverage,
        "gaze_blendshape_reference_coverage": reference_coverage,
    }
    if sample_summary is None or reference_summary is None:
        return result

    result.update(
        {
            "gaze_blendshape_horizontal_location_difference": abs(
                sample_summary.horizontal_location
                - reference_summary.horizontal_location
            ),
            "gaze_blendshape_vertical_location_difference": abs(
                sample_summary.vertical_location - reference_summary.vertical_location
            ),
            "gaze_blendshape_horizontal_amplitude_difference": abs(
                sample_summary.horizontal_amplitude
                - reference_summary.horizontal_amplitude
            ),
            "gaze_blendshape_vertical_amplitude_difference": abs(
                sample_summary.vertical_amplitude
                - reference_summary.vertical_amplitude
            ),
            "gaze_blendshape_speed_difference": (
                abs(sample_summary.speed - reference_summary.speed)
                if sample_summary.speed is not None
                and reference_summary.speed is not None
                else None
            ),
            "gaze_blendshape_binocular_disagreement_difference": abs(
                sample_summary.binocular_disagreement
                - reference_summary.binocular_disagreement
            ),
        }
    )
    return result


class GazeDynamicsModule(PipelineModule):
    """Compare unaligned MediaPipe ocular-control activation summaries."""

    name = "gaze_dynamics"
    description = (
        "Reference-relative MediaPipe eye-look activation distributions and dynamics"
    )
    default_config = {
        "models_dir": "models",
        "min_samples": 8,
        "num_faces": 1,
        "face_index": None,
    }
    models = [
        {
            "id": MODEL_REPO_ID,
            "type": "huggingface",
            "url": MODEL_URL,
            "revision": MODEL_REVISION,
            "task": f"MediaPipe eye-look blendshape activations ({MODEL_FILENAME})",
            "notes": (
                "Shared pinned Face Landmarker bundle; upstream Blendshape V2 model "
                "card is Apache-2.0 and outputs are not gaze angles"
            ),
        }
    ]
    metric_info = {
        "gaze_blendshape_horizontal_location_difference": (
            "Absolute median horizontal eye-look activation difference (0=equal)"
        ),
        "gaze_blendshape_vertical_location_difference": (
            "Absolute median vertical eye-look activation difference (0=equal)"
        ),
        "gaze_blendshape_horizontal_amplitude_difference": (
            "Absolute horizontal 10th-to-90th-percentile span difference (0=equal)"
        ),
        "gaze_blendshape_vertical_amplitude_difference": (
            "Absolute vertical 10th-to-90th-percentile span difference (0=equal)"
        ),
        "gaze_blendshape_speed_difference": (
            "Absolute median 2-D eye-look activation-speed difference per second (0=equal)"
        ),
        "gaze_blendshape_binocular_disagreement_difference": (
            "Absolute median left/right ocular-control disagreement difference (0=equal)"
        ),
        "gaze_blendshape_sample_coverage": (
            "Share of sample frames with valid MediaPipe eye-look activations (0-1)"
        ),
        "gaze_blendshape_reference_coverage": (
            "Share of reference frames with valid MediaPipe eye-look activations (0-1)"
        ),
    }
    metric_groups = {field: "face" for field in metric_info}

    def __init__(self, config=None):
        super().__init__(config)
        self.min_samples = max(2, int(self.config.get("min_samples", 8)))
        self._extractor = BlendshapeExtractor(
            str(self.config.get("models_dir", "models")),
            num_faces=max(1, int(self.config.get("num_faces", 1))),
            face_index=self.config.get("face_index"),
        )
        self._available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        self._available = self._extractor.setup("GazeDynamics")
        self._backend = self._extractor.backend

    def process(self, sample: Sample) -> Sample:
        if not self._available or not sample.is_video or sample.reference_path is None:
            return sample
        try:
            result = self._compare(Path(sample.path), Path(sample.reference_path))
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            metrics = sample.quality_metrics
            metrics.gaze_blendshape_horizontal_location_difference = result[
                "gaze_blendshape_horizontal_location_difference"
            ]
            metrics.gaze_blendshape_vertical_location_difference = result[
                "gaze_blendshape_vertical_location_difference"
            ]
            metrics.gaze_blendshape_horizontal_amplitude_difference = result[
                "gaze_blendshape_horizontal_amplitude_difference"
            ]
            metrics.gaze_blendshape_vertical_amplitude_difference = result[
                "gaze_blendshape_vertical_amplitude_difference"
            ]
            metrics.gaze_blendshape_speed_difference = result[
                "gaze_blendshape_speed_difference"
            ]
            metrics.gaze_blendshape_binocular_disagreement_difference = result[
                "gaze_blendshape_binocular_disagreement_difference"
            ]
            metrics.gaze_blendshape_sample_coverage = result[
                "gaze_blendshape_sample_coverage"
            ]
            metrics.gaze_blendshape_reference_coverage = result[
                "gaze_blendshape_reference_coverage"
            ]
            for field, value in result.items():
                if value is not None:
                    metrics.metric_backends[field] = self._backend
        except Exception as exc:  # noqa: BLE001 - metric modules degrade gracefully
            logger.warning("gaze_dynamics failed for %s: %s", sample.path, exc)
        return sample

    def _compare(
        self, sample_path: Path, reference_path: Path
    ) -> Dict[str, Optional[float]]:
        sample_track = self._extractor.extract(sample_path)
        reference_track = self._extractor.extract(reference_path)
        return compare_ocular_trajectories(
            sample_track, reference_track, self.min_samples
        )

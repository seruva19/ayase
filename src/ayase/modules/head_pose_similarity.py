"""Similarity of head-motion manner to a reference clip, without time alignment.

``head_motion_dynamics`` answers how much a head moves; this answers whether it
moves *like* the head in the reference. The two clips need not be the same
performance: nothing is matched frame to frame, so a generated clip driven by a
text prompt can still be compared with a recording of the person it is meant to
portray.

Both clips are reduced to two distributions -- the three head angles themselves,
and the rate at which they change -- and the distributions are compared with the
Wasserstein distance, averaged over the axes. Rates are expressed per second, so a
clip rendered at a lower frame rate is not reported as a calmer person.

The two halves answer different questions and are reported separately. Angles
describe the postures a head visits, and they carry camera placement as much as
habit: a speaker filmed from below yields different pitch than the same speaker
filmed level. Rates describe how the head travels between postures and are the
part that survives a change of camera. When the two disagree, the composite is
worth less than the rate half alone.

Measured behaviour on real recordings of three public speakers, 89 clips of five to
twenty seconds from podium speeches and press conferences, all 3916 pairs:

* separability of same-speaker from different-speaker pairs, AUC 0.718, bootstrap
  interval over clips 0.671 to 0.773;
* nearest neighbour is the same speaker in 84 of 89 clips, 94.4%, against a chance
  level near 32%;
* medians 0.652 within a speaker against 0.587 between speakers.

On the same material and protocol, facial-expression manner separates at 0.832 and
body motion at 0.536: the head is the strongest motion channel a head-and-shoulders
framing offers, and the only one of the two that separates speakers at all. The
score is still weak as an absolute verdict -- the two medians sit a tenth apart --
so use it to rank candidates against one common reference, not to decide whether a
single clip depicts a given person.

Backend: MediaPipe Face Landmarker, the same pinned bundle as
``head_motion_dynamics``, with facial transformation matrices enabled. Values are
left unset when the backend is unavailable or the face is not found often enough.
"""

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

from ._blendshape_utils import (
    MODEL_FILENAME,
    MODEL_REPO_ID,
    MODEL_REVISION,
    MODEL_URL,
    BlendshapeExtractor,
    select_face_index,
)
from .head_motion_dynamics import pose_and_center

logger = logging.getLogger(__name__)


def _agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Agreement of two one-dimensional distributions, 1.0 when they coincide.

    The Wasserstein distance carries the units of the quantity compared, so it is
    divided by the pooled spread before being turned into a score: without that a
    fixed distance would read as large disagreement for a still speaker and small
    for an animated one.
    """
    from scipy.stats import wasserstein_distance

    distance = float(wasserstein_distance(left, right))
    pooled = float(
        np.mean(np.abs(left - np.mean(left))) + np.mean(np.abs(right - np.mean(right)))
    )
    if pooled <= 0:
        return 1.0 if distance <= 0 else 0.0
    return float(1.0 / (1.0 + distance / pooled))


class HeadPoseSimilarityModule(PipelineModule):
    name = "head_pose_similarity"
    description = "Similarity of head-motion manner to a reference clip, compared as distributions"
    default_config = {
        "models_dir": "models",
        "stride": 3,
        "min_samples": 8,
        "num_faces": 1,
        "face_index": None,
    }
    metric_groups = {
        "head_pose_similarity": "motion",
        "head_pose_angle_agreement": "motion",
        "head_pose_rate_agreement": "motion",
        "head_pose_similarity_coverage": "motion",
    }
    metric_info = {
        "head_pose_similarity": "Composite head-motion manner similarity to the reference (0-1, higher=better)",
        "head_pose_angle_agreement": "Agreement of the head-angle distributions (0-1, carries camera placement)",
        "head_pose_rate_agreement": "Agreement of the angular-rate distributions (0-1, survives a change of camera)",
        "head_pose_similarity_coverage": "Lower of the two per-clip shares of sampled frames with a head pose (0-1)",
    }
    models = [
        {
            "id": MODEL_REPO_ID,
            "type": "huggingface",
            "url": MODEL_URL,
            "revision": MODEL_REVISION,
            "task": f"Face pose and landmarks ({MODEL_FILENAME})",
        }
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self.stride = max(1, int(self.config.get("stride", 3)))
        self.min_samples = max(4, int(self.config.get("min_samples", 8)))
        self._extractor = BlendshapeExtractor(
            str(self.config.get("models_dir", "models")),
            num_faces=max(1, int(self.config.get("num_faces", 1))),
            face_index=self.config.get("face_index"),
            output_facial_transformation_matrixes=True,
        )
        self._available = False

    def setup(self) -> None:
        self._available = self._extractor.setup("HeadPoseSimilarity")
        if not self._available:
            logger.warning("head_pose_similarity: face landmarker unavailable; metric disabled")

    def process(self, sample: Sample) -> Sample:
        if not self._available or not sample.is_video or sample.reference_path is None:
            return sample

        try:
            result = self._compare(Path(sample.path), Path(sample.reference_path))
        except Exception as exc:  # pragma: no cover - depends on decoder/backend
            logger.warning("head_pose_similarity failed for %s: %s", sample.path, exc)
            return sample
        if result is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        qm = sample.quality_metrics
        qm.head_pose_similarity_coverage = result["coverage"]
        qm.head_pose_angle_agreement = result["angles"]
        qm.head_pose_rate_agreement = result["rates"]
        qm.head_pose_similarity = result["composite"]
        return sample

    def _series(self, video: Path) -> Tuple[Optional[np.ndarray], float]:
        """Head angles over time, and the share of sampled frames that yielded one."""
        import cv2

        capture = cv2.VideoCapture(str(video))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                return None, 0.0
            rows: List[Tuple[float, float, float, float]] = []
            sampled = 0
            previous_ms = -1
            index = 0
            with self._extractor.create_landmarker() as landmarker:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if index % self.stride == 0:
                        sampled += 1
                        timestamp = max(int(round(index / fps * 1000.0)), previous_ms + 1)
                        previous_ms = timestamp
                        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        image = self._extractor.mediapipe.Image(
                            image_format=self._extractor.mediapipe.ImageFormat.SRGB, data=rgb
                        )
                        result = landmarker.detect_for_video(image, timestamp)
                        selected = select_face_index(result, self._extractor.face_index)
                        matrices = result.facial_transformation_matrixes or []
                        if selected is not None and selected < len(matrices):
                            values = pose_and_center(
                                matrices[selected],
                                result.face_landmarks[selected],
                                frame.shape[1],
                                frame.shape[0],
                            )
                            if values is not None:
                                rows.append((index / fps, values[0], values[1], values[2]))
                    index += 1
            if len(rows) < self.min_samples or sampled == 0:
                return None, len(rows) / float(sampled) if sampled else 0.0
            return np.asarray(rows, dtype=np.float64), len(rows) / float(sampled)
        finally:
            capture.release()

    def _compare(self, generated: Path, reference: Path) -> Optional[Dict[str, Any]]:
        left, left_coverage = self._series(generated)
        right, right_coverage = self._series(reference)
        coverage = round(min(left_coverage, right_coverage), 4)
        if left is None or right is None:
            return None

        def rates(series: np.ndarray) -> Optional[np.ndarray]:
            step = np.diff(series[:, 0])
            step[step <= 0] = np.nan
            values = np.abs(np.diff(series[:, 1:], axis=0)) / step[:, None]
            values = values[np.isfinite(values).all(axis=1)]
            return values if len(values) >= self.min_samples // 2 else None

        left_rates, right_rates = rates(left), rates(right)
        angle_scores = [_agreement(left[:, axis + 1], right[:, axis + 1]) for axis in range(3)]
        angles = round(float(np.mean(angle_scores)), 4)

        result: Dict[str, Any] = {"coverage": coverage, "angles": angles, "rates": None}
        if left_rates is not None and right_rates is not None:
            rate_scores = [
                _agreement(left_rates[:, axis], right_rates[:, axis]) for axis in range(3)
            ]
            result["rates"] = round(float(np.mean(rate_scores)), 4)
        parts = [value for value in (result["angles"], result["rates"]) if value is not None]
        result["composite"] = round(float(np.mean(parts)), 4)
        return result

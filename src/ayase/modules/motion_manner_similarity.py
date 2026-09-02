"""Similarity of movement manner between two clips, without time alignment.

Answers a question ``pose_driver_fidelity`` cannot: does this clip move *like*
the person in the reference, when the two are not performing the same movement?
Pose fidelity compares matched moments and is only meaningful when the generated
clip was driven by the reference. A clip generated from a text prompt has no such
correspondence -- its content is deliberately different -- yet the manner of
movement is exactly what an identity adapter is supposed to carry over: how fast
the person moves, how widely, and which parts of the body carry the motion.

The comparison is therefore distributional. Both clips are reduced to speed
distributions per body region, and the distributions are compared with the
Wasserstein distance. Nothing is matched frame to frame, so the clips may differ
in length, frame rate and content.

Two normalisations make the numbers comparable across clips and models:

* speeds are expressed in body scales per **second**, not per frame, so a clip
  rendered at 16 fps is not reported as slower than the same motion at 24 fps;
* positions are divided by the body scale of their own frame, so a person filmed
  closer to the camera does not read as moving more.

Coverage is reported next to every score. Arm motion in particular is only
measurable when the wrists are in frame at all: a tight head-and-shoulders
framing yields no arm signal, and a zero there means "not measured", not
"motionless". The arm score is left unset in that case rather than averaged in.

Backend: RTMPose + YOLOX via the :mod:`ayase.pose` primitive, the same weights as
``pose_driver_fidelity``. Values are left unset when the backend is unavailable.

Measured behaviour, and it is a negative result that governs how this metric may be
used. On 91 clips of three speakers filmed head-and-shoulders from a tripod --
podium speeches and press conferences, five to twenty seconds each, roughly five
minutes per speaker -- an exhaustive sweep of 1337 same-speaker and 2758
different-speaker pairs gives:

* separability of same-speaker from different-speaker pairs, AUC 0.536;
* nearest neighbour is the same speaker in 43 of 91 clips, 47%, against a chance
  level near 32%;
* medians 0.765 same-speaker against 0.748 different-speaker, with a within-group
  spread of 0.110 -- the difference is a seventh of the noise.

**This metric therefore does not identify a person and must not be used as if it
did.** Two different speakers at a lectern move more alike than one speaker filmed
on two different occasions. On the same material and the same protocol
``expression_similarity`` reaches AUC 0.832 and picks the right speaker in 96.5% of
clips, so the material does carry an individual signal -- it is carried by the face,
not by the body. What this metric is for is describing how much and how fast a clip
moves relative to a reference, and for ranking generations of *one* person against
one common reference; it says nothing about whether that person is who they claim.

Pairs are not independent observations -- one clip enters dozens of them -- so the
effective sample size is the number of clips, or of filming sessions when several
clips come from one. Confidence intervals must be taken by bootstrapping clips.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

#: COCO-17 indices grouped by the part of the body they describe. Head keypoints
#: move with every nod and turn, torso keypoints carry posture, arm keypoints
#: carry gesticulation -- mixing them into one number hides which of the three a
#: model actually reproduced.
HEAD_JOINTS = (0, 1, 2, 3, 4)
TORSO_JOINTS = (5, 6, 11, 12)
ARM_JOINTS = (7, 8, 9, 10)
WRIST_JOINTS = (9, 10)


def _speed_distribution(
    tracks: Sequence[Tuple[float, np.ndarray, np.ndarray]],
    joints: Sequence[int],
    min_conf: float,
) -> Optional[np.ndarray]:
    """Speeds of the given joints, in body scales per second.

    Args:
        tracks (Sequence[Tuple[float, np.ndarray, np.ndarray]]): Per-moment
            timestamp in seconds, normalised keypoints and their scores.
        joints (Sequence[int]): Keypoint indices taking part.
        min_conf (float): Confidence below which a keypoint is not used.

    Returns:
        Optional[np.ndarray]: Speeds, or ``None`` when fewer than two
        consecutive moments held a confident reading of the same joint.
    """
    speeds: List[float] = []
    for (t0, pts0, sc0), (t1, pts1, sc1) in zip(tracks, tracks[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        for joint in joints:
            if sc0[joint] < min_conf or sc1[joint] < min_conf:
                continue
            speeds.append(float(np.linalg.norm(pts1[joint] - pts0[joint]) / dt))
    return np.asarray(speeds, dtype=np.float64) if speeds else None


def _distribution_agreement(left: np.ndarray, right: np.ndarray) -> float:
    """Agreement of two speed distributions, 1.0 when they coincide.

    The Wasserstein distance is expressed in the units of the distributions
    themselves, so it is divided by their pooled spread before being turned into
    a score. Without that division the same distance would read as a large
    disagreement for a calm person and a small one for an animated person.

    Args:
        left (np.ndarray): Speeds of the sample.
        right (np.ndarray): Speeds of the reference.

    Returns:
        float: Agreement in ``(0, 1]``, higher = more alike.
    """
    from scipy.stats import wasserstein_distance

    distance = float(wasserstein_distance(left, right))
    pooled = float(np.mean(np.abs(left - np.mean(left))) + np.mean(np.abs(right - np.mean(right))))
    if pooled <= 0:
        return 1.0 if distance <= 0 else 0.0
    return float(1.0 / (1.0 + distance / pooled))


class MotionMannerSimilarityModule(PipelineModule):
    name = "motion_manner_similarity"
    description = "Similarity of movement manner to a reference clip, compared as distributions"
    default_config = {
        "device": "auto",
        "models_dir": "models",
        "moments": 48,
        "min_conf": 0.3,
        "min_speeds": 8,
        "arm_coverage_floor": 0.25,
    }
    metric_groups = {
        "motion_manner_similarity": "motion",
        "motion_manner_speed_agreement": "motion",
        "motion_manner_head_agreement": "motion",
        "motion_manner_arm_agreement": "motion",
        "motion_manner_amplitude_ratio": "motion",
        "motion_manner_coverage": "motion",
        "motion_manner_arm_coverage": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.moments = int(self.config.get("moments", 48))
        self.min_conf = float(self.config.get("min_conf", 0.3))
        self.min_speeds = int(self.config.get("min_speeds", 8))
        self.arm_coverage_floor = float(self.config.get("arm_coverage_floor", 0.25))
        self._backend = None

    def setup(self) -> None:
        from ayase.pose import load_pose_backend

        self._backend = load_pose_backend(
            device=self.config.get("device", "auto"),
            models_dir=self.config.get("models_dir", "models"),
        )
        if self._backend is None:
            logger.warning(
                "motion_manner_similarity: pose backend unavailable; metric disabled"
            )

    def process(self, sample: Sample) -> Sample:
        if self._backend is None:
            return sample
        if not sample.is_video or sample.reference_path is None:
            return sample

        try:
            result = self._compare(str(sample.path), str(sample.reference_path))
        except Exception as exc:  # pragma: no cover - depends on decoder/backend
            logger.warning("motion_manner_similarity failed for %s: %s", sample.path, exc)
            return sample
        if result is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        # Written out one by one rather than through ``setattr`` in a loop: the
        # documentation generator infers a module's output fields by reading
        # ``quality_metrics.<field> =`` assignments from the source, and a loop
        # leaves every field of this module listed as orphaned in METRICS.md.
        qm = sample.quality_metrics
        qm.motion_manner_coverage = result.get("motion_manner_coverage")
        qm.motion_manner_arm_coverage = result.get("motion_manner_arm_coverage")
        qm.motion_manner_similarity = result.get("motion_manner_similarity")
        qm.motion_manner_speed_agreement = result.get("motion_manner_speed_agreement")
        qm.motion_manner_head_agreement = result.get("motion_manner_head_agreement")
        qm.motion_manner_arm_agreement = result.get("motion_manner_arm_agreement")
        qm.motion_manner_amplitude_ratio = result.get("motion_manner_amplitude_ratio")
        return sample

    def _track(self, path: str) -> Tuple[List[Tuple[float, np.ndarray, np.ndarray]], float, float]:
        """Normalised keypoint track of one clip.

        Args:
            path (str): Video path.

        Returns:
            Tuple[List[Tuple[float, np.ndarray, np.ndarray]], float, float]: The
            track as (seconds, keypoints in body scales, scores), the share of
            sampled moments where a person was found, and the share where both
            wrists-bearing arms were confidently visible.
        """
        import cv2

        from ayase.pose import body_origin, body_scale, pose_keypoints

        cap = cv2.VideoCapture(path)
        try:
            if not cap.isOpened():
                return [], 0.0, 0.0
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if count <= 1 or fps <= 0:
                return [], 0.0, 0.0

            track: List[Tuple[float, np.ndarray, np.ndarray]] = []
            found = 0
            with_wrist = 0
            for step in range(self.moments):
                index = min(count - 1, int(step / max(self.moments - 1, 1) * (count - 1)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    continue
                people = pose_keypoints(frame, backend=self._backend)
                if not people:
                    continue
                person = people[0]
                points = np.asarray(person["keypoints"], dtype=np.float64)
                scores = np.asarray(person["scores"], dtype=np.float64)
                scale = body_scale(points, scores, self.min_conf)
                origin = body_origin(points, scores, self.min_conf)
                if not scale or origin is None:
                    continue
                found += 1
                if any(scores[j] >= self.min_conf for j in WRIST_JOINTS):
                    with_wrist += 1
                track.append((index / fps, (points - origin) / scale, scores))
            return (
                track,
                found / float(self.moments),
                with_wrist / float(self.moments),
            )
        finally:
            cap.release()

    def _compare(self, generated: str, reference: str) -> Optional[Dict[str, Any]]:
        sample_track, sample_cov, sample_arm_cov = self._track(generated)
        reference_track, reference_cov, reference_arm_cov = self._track(reference)
        if len(sample_track) < 2 or len(reference_track) < 2:
            return None

        groups = {
            "motion_manner_head_agreement": HEAD_JOINTS,
            "motion_manner_arm_agreement": ARM_JOINTS,
        }
        whole_body = tuple(HEAD_JOINTS + TORSO_JOINTS + ARM_JOINTS)

        result: Dict[str, Any] = {
            "motion_manner_coverage": round(min(sample_cov, reference_cov), 4),
            "motion_manner_arm_coverage": round(min(sample_arm_cov, reference_arm_cov), 4),
        }

        sample_speeds = _speed_distribution(sample_track, whole_body, self.min_conf)
        reference_speeds = _speed_distribution(reference_track, whole_body, self.min_conf)
        if (
            sample_speeds is None
            or reference_speeds is None
            or len(sample_speeds) < self.min_speeds
            or len(reference_speeds) < self.min_speeds
        ):
            return result

        result["motion_manner_speed_agreement"] = round(
            _distribution_agreement(sample_speeds, reference_speeds), 4
        )
        # Amplitude is reported as a ratio rather than a score: a value above one
        # says the clip moves more widely than the person does, below one that it
        # is calmer. Folding it into a symmetric score would hide the direction,
        # which is the practically useful half of the observation.
        reference_spread = float(np.std(reference_speeds))
        if reference_spread > 0:
            result["motion_manner_amplitude_ratio"] = round(
                float(np.std(sample_speeds) / reference_spread), 4
            )

        for field, joints in groups.items():
            # Arm motion is only measurable when the wrists are in frame in both
            # clips. Below the floor the arms are simply outside the framing, and
            # reporting a number there would read as "gesticulation not
            # reproduced" when nothing was observed at all.
            if joints is ARM_JOINTS and result["motion_manner_arm_coverage"] < self.arm_coverage_floor:
                continue
            left = _speed_distribution(sample_track, joints, self.min_conf)
            right = _speed_distribution(reference_track, joints, self.min_conf)
            if (
                left is None
                or right is None
                or len(left) < self.min_speeds
                or len(right) < self.min_speeds
            ):
                continue
            result[field] = round(_distribution_agreement(left, right), 4)

        parts = [
            result[key]
            for key in (
                "motion_manner_speed_agreement",
                "motion_manner_head_agreement",
                "motion_manner_arm_agreement",
            )
            if key in result
        ]
        if parts:
            result["motion_manner_similarity"] = round(float(np.mean(parts)), 4)
        return result

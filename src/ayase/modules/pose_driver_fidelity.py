"""Body-pose fidelity to a driving video (pose-transfer / reenactment).

Measures how closely the body pose in a generated clip follows the pose in the
driving clip supplied as ``sample.reference_path``. Complements
``expression_following``, which answers the same question for facial expression
only: a clip can copy a face perfectly while inventing an unrelated body.

Score is PCK-style -- the share of keypoints whose position, after both skeletons
are brought to a common frame, lands within a tolerance expressed in body scales.
Range 0-1, higher = better. Reported together with the share of moments in which
both skeletons were found at all, without which a score computed on a handful of
easy frames is indistinguishable from one computed on the whole clip.

Backend: RTMPose + YOLOX via the :mod:`ayase.pose` primitive (same weights as
``rtmpose_fidelity``). Values are left unset when the backend is unavailable.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PoseDriverFidelityModule(PipelineModule):
    name = "pose_driver_fidelity"
    description = "Body-pose fidelity to a driving video (PCK over normalised skeletons)"
    default_config = {
        "device": "auto",
        "models_dir": "models",
        "moments": 16,
        "alpha": 0.2,
        "min_conf": 0.3,
    }
    metric_groups = {
        "pose_driver_fidelity": "motion",
        "pose_driver_fidelity_min": "motion",
        "pose_driver_fidelity_coverage": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.moments = int(self.config.get("moments", 16))
        self.alpha = float(self.config.get("alpha", 0.2))
        self.min_conf = float(self.config.get("min_conf", 0.3))
        self._backend = None

    def setup(self) -> None:
        from ayase.pose import load_pose_backend

        self._backend = load_pose_backend(
            device=self.config.get("device", "auto"),
            models_dir=self.config.get("models_dir", "models"),
        )
        if self._backend is None:
            logger.warning("pose_driver_fidelity: pose backend unavailable; metric disabled")

    def process(self, sample: Sample) -> Sample:
        if self._backend is None:
            return sample
        if not sample.is_video or sample.reference_path is None:
            return sample

        try:
            result = self._compare(str(sample.path), str(sample.reference_path))
        except Exception as exc:  # pragma: no cover - depends on decoder/backend
            logger.warning("pose_driver_fidelity failed for %s: %s", sample.path, exc)
            return sample
        if result is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.pose_driver_fidelity = result["mean"]
        sample.quality_metrics.pose_driver_fidelity_min = result["min"]
        sample.quality_metrics.pose_driver_fidelity_coverage = result["coverage"]
        return sample

    def _compare(self, generated: str, driver: str) -> Optional[Dict[str, float]]:
        import cv2

        from ayase.pose import body_origin, body_scale, pose_keypoints

        caps = {}
        try:
            for key, path in (("gen", generated), ("drv", driver)):
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    return None
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if count <= 0:
                    return None
                caps[key] = (cap, count)

            hits: List[float] = []
            matched = 0
            for step in range(self.moments):
                # Clips are aligned by RELATIVE position, not by frame number: the
                # driving clip has its own length and frame rate, so matching by
                # index would compare different moments of the movement.
                position = (step + 0.5) / self.moments
                poses = {}
                for key in ("gen", "drv"):
                    cap, count = caps[key]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, int(position * count)))
                    ok, frame = cap.read()
                    people = pose_keypoints(frame, backend=self._backend) if ok else []
                    poses[key] = people[0] if people else None

                left, right = poses["gen"], poses["drv"]
                if left is None or right is None:
                    continue
                left_scale = body_scale(left["keypoints"], left["scores"], self.min_conf)
                right_scale = body_scale(right["keypoints"], right["scores"], self.min_conf)
                left_origin = body_origin(left["keypoints"], left["scores"], self.min_conf)
                right_origin = body_origin(right["keypoints"], right["scores"], self.min_conf)
                if not left_scale or not right_scale:
                    continue
                if left_origin is None or right_origin is None:
                    continue

                matched += 1
                inside = 0
                total = 0
                for joint in range(min(len(left["keypoints"]), len(right["keypoints"]))):
                    # A joint counts only when confidently found in BOTH clips.
                    # Counting one-sided joints would score a missing limb as a
                    # position error and blur two different failures together.
                    if left["scores"][joint] < self.min_conf:
                        continue
                    if right["scores"][joint] < self.min_conf:
                        continue
                    total += 1
                    left_point = (left["keypoints"][joint] - left_origin) / left_scale
                    right_point = (right["keypoints"][joint] - right_origin) / right_scale
                    if float(np.linalg.norm(left_point - right_point)) <= self.alpha:
                        inside += 1
                if total:
                    hits.append(inside / total)

            if not hits:
                return None
            return {
                "mean": round(float(np.mean(hits)), 4),
                "min": round(float(np.min(hits)), 4),
                "coverage": round(matched / float(self.moments), 4),
            }
        finally:
            for cap, _ in caps.values():
                cap.release()

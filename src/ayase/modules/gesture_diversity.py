"""L1 Diversity — mean pairwise L1 distance between gesture sequences.

EMAGE (Liu et al., CVPR 2024) and TalkSHOW (Yi et al., CVPR 2023) report
"Diversity": the average L1 distance between pose feature vectors across the
generated set — how much the gestures actually vary. Each video contributes
its mean pose vector (33 BlazePose joints, xy, per-video normalised); the
score is the mean pairwise L1 over all sample pairs.

l1_diversity (dataset-level) -- higher = more diverse gesture set.
"""

import logging
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.models import Sample

logger = logging.getLogger(__name__)


class GestureDiversityModule(BatchMetricModule):
    name = "gesture_diversity"
    description = "L1 Diversity — mean pairwise L1 of per-video pose signatures (EMAGE)"
    provenance = "adapted"
    sources = {
        "l1_diversity": "Diversity, EMAGE (Liu et al., arXiv:2401.00374); TalkSHOW (arXiv:2212.04420) — https://github.com/PantoMatrix/PantoMatrix",
    }
    deviations = {
        "l1_diversity": "the source computes L1 diversity on SMPL-X body+hand joint rotations; here per-video mean MediaPipe BlazePose xy vectors are used, so absolute values are not comparable",
    }
    default_config = {
        "fps": 15,
        "max_frames": 300,
    }
    metric_info = {
        "l1_diversity": "Mean pairwise L1 between per-video pose signatures (higher=more diverse)",
    }
    metric_groups = {"l1_diversity": "motion"}

    def __init__(self, config=None):
        super().__init__(config)
        self._pose = None
        self._backend = None

    def setup(self) -> None:
        if self.config.get("test_mode"):
            self._backend = "unavailable"
            return
        try:
            import mediapipe as mp

            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False, model_complexity=1,
                enable_segmentation=False, min_detection_confidence=0.5,
            )
            self._backend = "mediapipe"
        except ImportError:
            logger.warning("gesture_diversity: mediapipe not installed, disabled")
            self._backend = "unavailable"
        except Exception as e:
            logger.warning("gesture_diversity: Pose init failed: %s", e)
            self._backend = "unavailable"

    def on_dispose(self) -> None:
        try:
            if self._pose is not None:
                self._pose.close()
        except Exception:
            pass
        self._pose = None
        try:
            self.compute_distribution_metric(self._feature_cache, None)
        except Exception as e:
            logger.error("gesture_diversity: failed: %s", e)
        finally:
            self._feature_cache = []

    def extract_features(self, sample: Sample):
        """Per-video signature: mean pose vector over frames (66-dim)."""
        if self._backend != "mediapipe":
            return None
        if not sample.is_video:
            return None
        from ._mp_seq import body_pose_seq

        res = body_pose_seq(Path(sample.path),
                            fps=float(self.config.get("fps", 15)),
                            max_frames=int(self.config.get("max_frames", 300)),
                            detector=self._pose)
        if res is None or len(res[0]) < 2:
            return None
        xy = res[0][:, :, :2]  # [T, 33, 2]
        # centre each frame on the hip midpoint (joints 23/24) for position invariance
        hip = (xy[:, 23] + xy[:, 24]) * 0.5  # [T, 2]
        xy = xy - hip[:, None, :]
        return xy.reshape(len(xy), -1).mean(axis=0)

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> Optional[float]:
        if len(features) < 2:
            return None
        feats = np.stack(features)
        dists = [np.abs(a - b).sum()
                 for a, b in combinations(feats, 2)]
        score = float(np.mean(dists))
        if not np.isfinite(score):
            return None
        if hasattr(self, "pipeline") and self.pipeline and hasattr(
                self.pipeline, "add_dataset_metric"):
            self.pipeline.add_dataset_metric("l1_diversity", score)
        return score

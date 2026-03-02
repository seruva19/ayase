import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class HumanFidelityModule(PipelineModule):
    name = "human_fidelity"
    description = "Pose detection quality via MediaPipe landmark visibility, proportions, and symmetry"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)
        self._mp_available = False
        self.mp_pose = None
        self.mp_drawing = None

    def setup(self) -> None:
        try:
            import mediapipe as mp

            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self._mp_available = True
        except ImportError:
            logger.warning("MediaPipe not installed. Human fidelity checks disabled.")

    # MediaPipe landmark indices for anatomy checks
    _LIMB_PAIRS = [
        # (start, end, name) — used for bilateral symmetry and proportion checks
        (11, 13, "left_upper_arm"),   # left shoulder → left elbow
        (12, 14, "right_upper_arm"),  # right shoulder → right elbow
        (13, 15, "left_forearm"),     # left elbow → left wrist
        (14, 16, "right_forearm"),    # right elbow → right wrist
        (23, 25, "left_thigh"),       # left hip → left knee
        (24, 26, "right_thigh"),      # right hip → right knee
        (25, 27, "left_shin"),        # left knee → left ankle
        (26, 28, "right_shin"),       # right knee → right ankle
    ]

    _SYMMETRY_PAIRS = [
        # (left_limb_idx, right_limb_idx) into _LIMB_PAIRS
        (0, 1),  # upper arms
        (2, 3),  # forearms
        (4, 5),  # thighs
        (6, 7),  # shins
    ]

    def process(self, sample: Sample) -> Sample:
        if not self._mp_available:
            return sample

        image = self._load_image(sample)
        if image is None:
            return sample

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            with self.mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
            ) as pose:
                results = pose.process(image_rgb)

                if not results.pose_landmarks:
                    logger.debug("No human detected for pose fidelity check.")
                    return sample

                landmarks = results.pose_landmarks.landmark
                h, w = image.shape[:2]

                # 1. Average visibility check
                avg_visibility = sum(lm.visibility for lm in landmarks) / len(landmarks)

                if avg_visibility < 0.3:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Low pose visibility: {avg_visibility:.2f} (possible anatomy distortion)",
                            details={"avg_visibility": avg_visibility},
                        )
                    )

                # 2. Limb proportion checks
                limb_lengths = self._compute_limb_lengths(landmarks, w, h)
                proportion_issues = self._check_proportions(limb_lengths)
                if proportion_issues:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Abnormal limb proportions detected ({len(proportion_issues)} issues)",
                            details={"proportion_issues": proportion_issues},
                            recommendation="Generated human may have distorted body proportions.",
                        )
                    )

                # 3. Bilateral symmetry check
                symmetry_ratio = self._check_symmetry(limb_lengths)
                if symmetry_ratio is not None and symmetry_ratio < 0.5:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"Poor bilateral symmetry: {symmetry_ratio:.2f}",
                            details={"symmetry_ratio": symmetry_ratio},
                            recommendation="Left/right limb lengths are highly asymmetric.",
                        )
                    )

        except Exception as e:
            logger.warning(f"Human fidelity check failed: {e}")

        return sample

    def _compute_limb_lengths(self, landmarks, img_w: int, img_h: int) -> dict:
        """Compute pixel-space limb lengths from landmarks."""
        lengths = {}
        for start_idx, end_idx, name in self._LIMB_PAIRS:
            lm_start = landmarks[start_idx]
            lm_end = landmarks[end_idx]
            # Only compute if both landmarks are reasonably visible
            if lm_start.visibility > 0.3 and lm_end.visibility > 0.3:
                dx = (lm_start.x - lm_end.x) * img_w
                dy = (lm_start.y - lm_end.y) * img_h
                lengths[name] = float(np.sqrt(dx * dx + dy * dy))
        return lengths

    def _check_proportions(self, limb_lengths: dict) -> list:
        """Check that limb proportions are within anthropometric norms.

        Uses approximate human body ratios:
        - Upper arm / forearm ~ 1.0-1.5
        - Thigh / shin ~ 0.9-1.4
        - Upper arm / thigh ~ 0.5-1.0
        """
        issues = []
        ratio_checks = [
            ("left_upper_arm", "left_forearm", 0.6, 2.0, "left arm ratio"),
            ("right_upper_arm", "right_forearm", 0.6, 2.0, "right arm ratio"),
            ("left_thigh", "left_shin", 0.6, 1.8, "left leg ratio"),
            ("right_thigh", "right_shin", 0.6, 1.8, "right leg ratio"),
        ]

        for limb_a, limb_b, min_ratio, max_ratio, label in ratio_checks:
            if limb_a in limb_lengths and limb_b in limb_lengths:
                len_b = limb_lengths[limb_b]
                if len_b > 1e-6:
                    ratio = limb_lengths[limb_a] / len_b
                    if ratio < min_ratio or ratio > max_ratio:
                        issues.append(f"{label}: {ratio:.2f} (expected {min_ratio}-{max_ratio})")

        return issues

    def _check_symmetry(self, limb_lengths: dict) -> Optional[float]:
        """Check bilateral symmetry. Returns ratio 0-1 (1 = perfectly symmetric)."""
        ratios = []
        for left_idx, right_idx in self._SYMMETRY_PAIRS:
            left_name = self._LIMB_PAIRS[left_idx][2]
            right_name = self._LIMB_PAIRS[right_idx][2]
            if left_name in limb_lengths and right_name in limb_lengths:
                a, b = limb_lengths[left_name], limb_lengths[right_name]
                if max(a, b) > 1e-6:
                    ratios.append(min(a, b) / max(a, b))
        return float(np.mean(ratios)) if ratios else None

    def _load_image(self, sample: Sample) -> Optional[np.ndarray]:
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            else:
                return cv2.imread(str(sample.path))
        except Exception:
            return None

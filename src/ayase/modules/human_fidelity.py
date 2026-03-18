"""Human fidelity module — VBench-2.0 dimension.

Assesses body, hand, and face quality in generated humans via pose
detection and landmark analysis.

Backend tiers:
  1. **DWPose** — Full-body + hand + face landmarks (dwpose / mmpose)
  2. **MediaPipe** — 33 body landmarks
  3. **Heuristic** — Haar cascades + skin segmentation
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class HumanFidelityModule(PipelineModule):
    name = "human_fidelity"
    description = "Human body/hand/face fidelity (DWPose / MediaPipe / heuristic)"
    default_config = {}

    # MediaPipe landmark indices for anatomy checks
    _LIMB_PAIRS = [
        (11, 13, "left_upper_arm"),
        (12, 14, "right_upper_arm"),
        (13, 15, "left_forearm"),
        (14, 16, "right_forearm"),
        (23, 25, "left_thigh"),
        (24, 26, "right_thigh"),
        (25, 27, "left_shin"),
        (26, 28, "right_shin"),
    ]

    _SYMMETRY_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "heuristic"
        self._mp_available = False
        self._dwpose_available = False
        self.mp_pose = None

    def setup(self) -> None:
        # Tier 1: DWPose
        try:
            from dwpose import DWposeDetector  # noqa: F401
            self._dwpose_available = True
            self._backend = "dwpose"
            logger.info("HumanFidelity using DWPose backend")
            return
        except ImportError:
            pass

        # Tier 2: MediaPipe
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self._mp_available = True
            self._backend = "mediapipe"
            logger.info("HumanFidelity using MediaPipe backend")
            return
        except ImportError:
            pass

        # Tier 3: Heuristic
        logger.info("HumanFidelity using heuristic backend (Haar + skin)")

    def process(self, sample: Sample) -> Sample:
        image = self._load_image(sample)
        if image is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        try:
            if self._backend == "dwpose":
                score, issues = self._compute_dwpose(image)
            elif self._backend == "mediapipe":
                score, issues = self._compute_mediapipe(image)
            else:
                score, issues = self._compute_heuristic(image)

            if score is not None:
                sample.quality_metrics.human_fidelity_score = score

            for issue in issues:
                sample.validation_issues.append(issue)

        except Exception as e:
            logger.warning("Human fidelity check failed: %s", e)

        return sample

    # ------------------------------------------------------------------ #
    # Tier 1: DWPose                                                       #
    # ------------------------------------------------------------------ #

    def _compute_dwpose(self, image: np.ndarray) -> tuple:
        from dwpose import DWposeDetector

        detector = DWposeDetector()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = detector(image_rgb)
        issues = []

        if result is None or len(result.get("bodies", {}).get("candidate", [])) == 0:
            return None, issues

        # Body score from landmark confidence
        bodies = result.get("bodies", {})
        body_candidates = bodies.get("candidate", [])
        body_scores = [c[2] for c in body_candidates if len(c) > 2]
        body_score = float(np.mean(body_scores)) if body_scores else 0.5

        # Hand score
        hands = result.get("hands", [])
        if hands and len(hands) > 0:
            hand_scores = []
            for hand in hands:
                if hand is not None and len(hand) > 0:
                    hand_confs = [p[2] for p in hand if len(p) > 2]
                    if hand_confs:
                        hand_scores.append(float(np.mean(hand_confs)))
            hand_score = float(np.mean(hand_scores)) if hand_scores else 0.3
        else:
            hand_score = 0.3

        # Face score
        faces = result.get("faces", [])
        if faces and len(faces) > 0:
            face_scores = []
            for face in faces:
                if face is not None and len(face) > 0:
                    face_confs = [p[2] for p in face if len(p) > 2]
                    if face_confs:
                        face_scores.append(float(np.mean(face_confs)))
            face_score = float(np.mean(face_scores)) if face_scores else 0.3
        else:
            face_score = 0.3

        score = 0.4 * body_score + 0.3 * hand_score + 0.3 * face_score
        return float(np.clip(score, 0.0, 1.0)), issues

    # ------------------------------------------------------------------ #
    # Tier 2: MediaPipe                                                    #
    # ------------------------------------------------------------------ #

    def _compute_mediapipe(self, image: np.ndarray) -> tuple:
        issues = []
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            results = pose.process(image_rgb)

            if not results.pose_landmarks:
                return None, issues

            landmarks = results.pose_landmarks.landmark
            h, w = image.shape[:2]

            # Body score: visibility × proportion correctness × symmetry
            avg_visibility = sum(lm.visibility for lm in landmarks) / len(landmarks)

            if avg_visibility < 0.3:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low pose visibility: {avg_visibility:.2f} (possible anatomy distortion)",
                        details={"avg_visibility": avg_visibility},
                    )
                )

            limb_lengths = self._compute_limb_lengths(landmarks, w, h)
            proportion_issues = self._check_proportions(limb_lengths)
            proportion_score = 1.0 - min(len(proportion_issues) * 0.2, 1.0)

            if proportion_issues:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Abnormal limb proportions detected ({len(proportion_issues)} issues)",
                        details={"proportion_issues": proportion_issues},
                        recommendation="Generated human may have distorted body proportions.",
                    )
                )

            symmetry_ratio = self._check_symmetry(limb_lengths)
            symmetry_score = symmetry_ratio if symmetry_ratio is not None else 0.7

            if symmetry_ratio is not None and symmetry_ratio < 0.5:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Poor bilateral symmetry: {symmetry_ratio:.2f}",
                        details={"symmetry_ratio": symmetry_ratio},
                        recommendation="Left/right limb lengths are highly asymmetric.",
                    )
                )

            body_score = (avg_visibility + proportion_score + symmetry_score) / 3.0

            # Hand score: check wrist/finger landmark visibility (indices 15-22)
            hand_indices = [15, 16, 17, 18, 19, 20, 21, 22]
            hand_vis = [landmarks[i].visibility for i in hand_indices if i < len(landmarks)]
            hand_score = float(np.mean(hand_vis)) if hand_vis else 0.3

            # Face score: check face landmark visibility (indices 0-10)
            face_indices = list(range(11))
            face_vis = [landmarks[i].visibility for i in face_indices if i < len(landmarks)]
            face_score = float(np.mean(face_vis)) if face_vis else 0.3

            score = 0.4 * body_score + 0.3 * hand_score + 0.3 * face_score
            return float(np.clip(score, 0.0, 1.0)), issues

    # ------------------------------------------------------------------ #
    # Tier 3: Heuristic                                                    #
    # ------------------------------------------------------------------ #

    def _compute_heuristic(self, image: np.ndarray) -> tuple:
        issues = []
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Face detection via Haar cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_score = min(len(faces) * 0.5, 1.0) if len(faces) > 0 else 0.0

        # Skin detection via HSV range
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_ratio = float(np.sum(skin_mask > 0)) / (h * w)

        # Body score based on skin presence
        body_score = min(skin_ratio * 5.0, 1.0)  # ~20% skin = 1.0

        # Hand region approximation (skin regions in lower-middle area)
        hand_region = skin_mask[h // 2:, w // 4: 3 * w // 4]
        hand_ratio = float(np.sum(hand_region > 0)) / (hand_region.size + 1)
        hand_score = min(hand_ratio * 8.0, 1.0)

        if len(faces) == 0 and skin_ratio < 0.05:
            # No human detected — return None
            return None, issues

        score = 0.4 * body_score + 0.3 * hand_score + 0.3 * face_score
        return float(np.clip(score, 0.0, 1.0)), issues

    # ------------------------------------------------------------------ #
    # Helpers (preserved from original)                                    #
    # ------------------------------------------------------------------ #

    def _compute_limb_lengths(self, landmarks, img_w: int, img_h: int) -> dict:
        lengths = {}
        for start_idx, end_idx, name in self._LIMB_PAIRS:
            lm_start = landmarks[start_idx]
            lm_end = landmarks[end_idx]
            if lm_start.visibility > 0.3 and lm_end.visibility > 0.3:
                dx = (lm_start.x - lm_end.x) * img_w
                dy = (lm_start.y - lm_end.y) * img_h
                lengths[name] = float(np.sqrt(dx * dx + dy * dy))
        return lengths

    def _check_proportions(self, limb_lengths: dict) -> list:
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

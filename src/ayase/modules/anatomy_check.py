"""Human anatomy plausibility check for AI-generated content.

Flags the AIGC human-anatomy (AIGC-HA) artifact taxonomy — extra/missing
limbs, duplicated hands, duplicated heads, and grossly non-human limb
proportions — using a real pose detector plus published rule scoring.

Backends (real detectors only — no pixel-statistics fallback):
  1. **DWPose** — multi-person body (COCO-18) + per-hand + per-face keypoints.
     Best backend: its separate hand/face lists let the count-based rules fire.
  2. **MediaPipe Pose** — 33 single-person body landmarks. Supports the
     limb-ratio / asymmetry rules only (Pose exposes no separate hand list, so
     it structurally cannot report "extra hands").

Loading conventions mirror ``human_fidelity`` (DWPose first, then MediaPipe),
and the detector is built once per pipeline via ``shared_runtime_resource`` so
the pose model is not loaded twice across samples/frames.

Scoring: for every detected person the keypoints are grouped by body part and
the anatomy rules below are checked. ``anatomy_score`` is the fraction of
person-frames whose anatomy is plausible (0-1, higher = better). No detector
available -> ``_backend = "unavailable"`` and the metric is left ``None``. A
detector that finds no people in any frame also leaves the metric ``None``
(``no humans`` != ``good anatomy``).
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Normalized keypoint schema                                                   #
# --------------------------------------------------------------------------- #
# COCO-18 / OpenPose BODY_18 joint order (DWPose body-candidate order). A
# normalized "person" is a dict:
#   {
#     "body":  {joint_name: (x, y, conf)},   # subset of BODY_18, high-conf only
#     "hands": [(cx, cy, conf), ...],         # hand clusters for THIS person
#     "heads": [(cx, cy, conf), ...],         # head clusters for THIS person
#   }
BODY_18: List[str] = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

Person = Dict[str, object]

# --------------------------------------------------------------------------- #
# Rule thresholds — each documented at its check site below                    #
# --------------------------------------------------------------------------- #
KEYPOINT_CONF = 0.30          # min keypoint confidence to treat a joint as present
HAND_CONF = 0.30              # min confidence for a hand cluster to count
HEAD_CONF = 0.30              # min confidence for a head cluster to count
MAX_HANDS_PER_PERSON = 2      # a human has at most 2 hands
MAX_HEADS_PER_PERSON = 1      # a human has at most 1 head
# forearm / upper-arm length ratio. Anthropometry puts the forearm at roughly
# 0.7-1.0x the upper arm; the band is widened to tolerate foreshortening and
# separate x/y normalisation, so only gross violations (fused / duplicated /
# stretched limbs) trip it.
ARM_RATIO_MIN, ARM_RATIO_MAX = 0.55, 1.90
# shin / thigh length ratio (similar reasoning to the arm ratio).
LEG_RATIO_MIN, LEG_RATIO_MAX = 0.55, 1.90
# bilateral symmetry: min(left, right) / max(left, right) for a paired limb.
# Below this the two sides differ by more than ~2.2x — a hallmark of a merged
# or spurious extra limb.
SYMMETRY_MIN = 0.45


def _seg_len(body: Dict[str, tuple], a: str, b: str, conf: float = KEYPOINT_CONF) -> Optional[float]:
    """Euclidean length of a limb segment, or None if either end is low-conf/absent."""
    pa = body.get(a)
    pb = body.get(b)
    if pa is None or pb is None:
        return None
    if len(pa) < 3 or len(pb) < 3 or pa[2] < conf or pb[2] < conf:
        return None
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def check_person_anatomy(person: Person) -> Tuple[bool, List[str]]:
    """Return ``(plausible, reasons)`` for one normalized person.

    Rules (each with its threshold):
      A. Extra hands  — > MAX_HANDS_PER_PERSON (2) high-conf hand clusters.
      B. Extra heads  — > MAX_HEADS_PER_PERSON (1) high-conf head clusters.
      C. Arm ratio    — forearm/upper-arm outside [ARM_RATIO_MIN, ARM_RATIO_MAX].
      D. Leg ratio    — shin/thigh outside [LEG_RATIO_MIN, LEG_RATIO_MAX].
      E. Asymmetry    — min/max of a paired limb (forearm, upper-arm, thigh,
                        shin) below SYMMETRY_MIN.
    A person is plausible only if NO rule fires.
    """
    body: Dict[str, tuple] = person.get("body", {}) or {}  # type: ignore[assignment]
    hands = person.get("hands", []) or []  # type: ignore[assignment]
    heads = person.get("heads", []) or []  # type: ignore[assignment]
    reasons: List[str] = []

    # Rule A — extra hands (duplicated-hand artifact)
    n_hands = sum(1 for h in hands if len(h) > 2 and h[2] >= HAND_CONF)
    if n_hands > MAX_HANDS_PER_PERSON:
        reasons.append(f"extra_hands:{n_hands}>{MAX_HANDS_PER_PERSON}")

    # Rule B — extra heads (duplicated-head artifact)
    n_heads = sum(1 for h in heads if len(h) > 2 and h[2] >= HEAD_CONF)
    if n_heads > MAX_HEADS_PER_PERSON:
        reasons.append(f"extra_heads:{n_heads}>{MAX_HEADS_PER_PERSON}")

    # Rule C — arm length ratio (forearm / upper arm) per side
    for side in ("l", "r"):
        upper = _seg_len(body, f"{side}_shoulder", f"{side}_elbow")
        fore = _seg_len(body, f"{side}_elbow", f"{side}_wrist")
        if upper and fore and upper > 1e-6:
            ratio = fore / upper
            if ratio < ARM_RATIO_MIN or ratio > ARM_RATIO_MAX:
                reasons.append(f"{side}_arm_ratio:{ratio:.2f}")

    # Rule D — leg length ratio (shin / thigh) per side
    for side in ("l", "r"):
        thigh = _seg_len(body, f"{side}_hip", f"{side}_knee")
        shin = _seg_len(body, f"{side}_knee", f"{side}_ankle")
        if thigh and shin and thigh > 1e-6:
            ratio = shin / thigh
            if ratio < LEG_RATIO_MIN or ratio > LEG_RATIO_MAX:
                reasons.append(f"{side}_leg_ratio:{ratio:.2f}")

    # Rule E — bilateral symmetry of paired limbs
    pairs = [
        ("forearm", "l_elbow", "l_wrist", "r_elbow", "r_wrist"),
        ("upper_arm", "l_shoulder", "l_elbow", "r_shoulder", "r_elbow"),
        ("thigh", "l_hip", "l_knee", "r_hip", "r_knee"),
        ("shin", "l_knee", "l_ankle", "r_knee", "r_ankle"),
    ]
    for name, la, lb, ra, rb in pairs:
        left = _seg_len(body, la, lb)
        right = _seg_len(body, ra, rb)
        if left and right and max(left, right) > 1e-6:
            ratio = min(left, right) / max(left, right)
            if ratio < SYMMETRY_MIN:
                reasons.append(f"{name}_asymmetry:{ratio:.2f}")

    return (len(reasons) == 0), reasons


def score_person_frames(person_frames: List[List[Person]]) -> Optional[float]:
    """Fraction of person-frames with plausible anatomy.

    ``person_frames`` is a list (one entry per frame) of person lists. Returns
    ``None`` when no person was detected in any frame — "no humans" is not the
    same as "good anatomy".
    """
    total = 0
    plausible = 0
    for persons in person_frames:
        for person in persons:
            total += 1
            ok, _reasons = check_person_anatomy(person)
            if ok:
                plausible += 1
    if total == 0:
        return None
    return plausible / total


# --------------------------------------------------------------------------- #
# Detector-output adapters -> normalized persons                              #
# --------------------------------------------------------------------------- #

# MediaPipe Pose landmark indices -> BODY_18 names (subset we need).
_MP_MAP: Dict[str, int] = {
    "nose": 0,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
}


def _person_from_mediapipe(landmarks) -> Person:
    """Build one normalized person from MediaPipe Pose landmarks (single person)."""
    body: Dict[str, tuple] = {}
    for name, idx in _MP_MAP.items():
        if idx < len(landmarks):
            lm = landmarks[idx]
            body[name] = (float(lm.x), float(lm.y), float(getattr(lm, "visibility", 1.0)))
    nose = body.get("nose")
    heads = [nose] if (nose is not None and nose[2] >= HEAD_CONF) else []
    # MediaPipe Pose has no separate multi-hand list -> extra-hand rule cannot
    # fire; ratio/asymmetry rules still apply.
    return {"body": body, "hands": [], "heads": heads}


def _cluster_from_part(part) -> Optional[tuple]:
    """Collapse a detected hand/face (many points) into (cx, cy, mean_conf)."""
    pts = np.asarray(part, dtype=float)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    confs = pts[:, 2] if pts.shape[1] > 2 else np.ones(pts.shape[0])
    valid = confs > 0.0  # DWPose marks missing keypoints with 0 confidence
    if not np.any(valid):
        return None
    cx = float(pts[valid, 0].mean())
    cy = float(pts[valid, 1].mean())
    conf = float(confs[valid].mean())
    return (cx, cy, conf)


def _person_centroid(person: Person) -> Optional[tuple]:
    body: Dict[str, tuple] = person.get("body", {}) or {}  # type: ignore[assignment]
    pts = [(v[0], v[1]) for v in body.values() if len(v) >= 3 and v[2] >= KEYPOINT_CONF]
    if not pts:
        return None
    arr = np.asarray(pts, dtype=float)
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


def _assign_clusters(persons: List[Person], clusters: List[tuple], key: str) -> None:
    """Assign each hand/head cluster to the nearest person by body centroid."""
    if not persons:
        return
    centroids = [_person_centroid(p) for p in persons]
    for cluster in clusters:
        best_i = 0
        best_d: Optional[float] = None
        for i, cent in enumerate(centroids):
            if cent is None:
                continue
            d = (cluster[0] - cent[0]) ** 2 + (cluster[1] - cent[1]) ** 2
            if best_d is None or d < best_d:
                best_d = d
                best_i = i
        persons[best_i][key].append(cluster)  # type: ignore[union-attr]


def _persons_from_dwpose(result: dict) -> List[Person]:
    """Build normalized persons from a DWPose detector result dict.

    Uses the OpenPose BODY_18 ``candidate`` + ``subset`` layout when present,
    falling back to the first 18 candidates for a single detected person. The
    global hand/face lists are collapsed to clusters and assigned to the nearest
    person. Any structural surprise raises and is caught by the caller (that
    frame then contributes no persons).
    """
    bodies = result.get("bodies", {}) or {}
    raw_candidate = bodies.get("candidate", [])
    raw_subset = bodies.get("subset", [])
    candidate = np.asarray(raw_candidate, dtype=float) if len(raw_candidate) else np.zeros((0, 0))
    subset = np.asarray(raw_subset, dtype=float) if len(raw_subset) else np.zeros((0, 0))

    person_bodies: List[Dict[str, tuple]] = []
    if candidate.ndim == 2 and candidate.shape[0] > 0 and subset.ndim == 2 and subset.shape[1] >= 18:
        for row in subset:
            body: Dict[str, tuple] = {}
            for j, name in enumerate(BODY_18):
                idx = int(row[j])
                if 0 <= idx < candidate.shape[0]:
                    x = float(candidate[idx, 0])
                    y = float(candidate[idx, 1])
                    conf = float(candidate[idx, 2]) if candidate.shape[1] > 2 else 1.0
                    body[name] = (x, y, conf)
            if body:
                person_bodies.append(body)
    elif candidate.ndim == 2 and candidate.shape[0] >= 18:
        body = {}
        for j, name in enumerate(BODY_18):
            x = float(candidate[j, 0])
            y = float(candidate[j, 1])
            conf = float(candidate[j, 2]) if candidate.shape[1] > 2 else 1.0
            body[name] = (x, y, conf)
        person_bodies.append(body)

    if not person_bodies:
        return []

    persons: List[Person] = [{"body": b, "hands": [], "heads": []} for b in person_bodies]

    hand_clusters = [c for c in (_cluster_from_part(h) for h in (result.get("hands", []) or [])) if c]
    head_clusters = [c for c in (_cluster_from_part(f) for f in (result.get("faces", []) or [])) if c]
    _assign_clusters(persons, hand_clusters, "hands")
    _assign_clusters(persons, head_clusters, "heads")
    return persons


class AnatomyCheckModule(PipelineModule):
    name = "anatomy_check"
    description = "Human anatomy plausibility (extra/duplicated limbs) via DWPose/MediaPipe (0-1, higher=better)"
    default_config = {
        "subsample": 8,        # frames sampled per video
        "warn_threshold": 0.5,  # emit a WARNING issue below this score
        "device": "auto",
    }
    metric_info = {
        "anatomy_score": "Keypoint-based limb-count/anatomy plausibility (0-1, higher=better)",
    }
    metric_groups = {
        "anatomy_score": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "unavailable"
        self._ml_available = False
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        from ayase.runtime import resolve_torch_device
        self._device = resolve_torch_device(self.config.get("device", "auto"))

        # Tier 1: DWPose (mirrors human_fidelity loading order/conventions)
        try:
            from dwpose import DWposeDetector  # noqa: F401
            self._backend = "dwpose"
            self._ml_available = True
            logger.info("AnatomyCheck using DWPose backend")
            return
        except ImportError:
            pass

        # Tier 2: MediaPipe Pose
        try:
            import mediapipe as mp  # noqa: F401
            self._backend = "mediapipe"
            self._ml_available = True
            logger.info("AnatomyCheck using MediaPipe backend")
            return
        except ImportError:
            pass

        logger.warning("AnatomyCheck: no pose backend available (install dwpose or mediapipe)")

    # ------------------------------------------------------------------ #
    # Shared detector (loaded once per pipeline)                          #
    # ------------------------------------------------------------------ #

    def _get_detector(self):
        from ayase.runtime import shared_runtime_resource

        if self._backend == "dwpose":
            def build_dwpose():
                from dwpose import DWposeDetector
                return DWposeDetector()

            return shared_runtime_resource(self, ("pose_detector", "dwpose", self._device), build_dwpose)

        if self._backend == "mediapipe":
            def build_mediapipe():
                import mediapipe as mp
                return mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                )

            return shared_runtime_resource(self, ("pose_detector", "mediapipe"), build_mediapipe)

        return None

    def _detect_persons(self, detector, frame_bgr: np.ndarray) -> List[Person]:
        # cvtColor copies, so passing a read-only shared frame view is safe.
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self._backend == "dwpose":
            result = detector(rgb)
            if not isinstance(result, dict):
                return []
            return _persons_from_dwpose(result)
        if self._backend == "mediapipe":
            results = detector.process(rgb)
            if not getattr(results, "pose_landmarks", None):
                return []
            return [_person_from_mediapipe(results.pose_landmarks.landmark)]
        return []

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        from ayase.image import sample_frames

        max_frames = self.config.get("subsample", 8) if sample.is_video else 1
        return list(sample_frames(sample.path, max_frames=max_frames, color="bgr"))

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        frames = self._load_frames(sample)
        if not frames:
            return sample

        detector = self._get_detector()
        if detector is None:
            return sample

        person_frames: List[List[Person]] = []
        for frame in frames:
            try:
                person_frames.append(self._detect_persons(detector, frame))
            except Exception as e:  # detector/format surprises must not crash the run
                logger.debug("anatomy_check: pose detection failed on a frame: %s", e)
                person_frames.append([])

        score = score_person_frames(person_frames)
        if score is None:
            logger.debug(
                "anatomy_check: no persons detected in %s; leaving anatomy_score unset",
                sample.path,
            )
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.anatomy_score = float(score)

        if score < self.config.get("warn_threshold", 0.5):
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Implausible anatomy in {(1.0 - score) * 100:.0f}% of person-frames",
                    details={"anatomy_score": score, "backend": self._backend},
                    recommendation=(
                        "Generated humans show extra/duplicated limbs, extra heads, "
                        "or non-human limb proportions."
                    ),
                )
            )

        return sample

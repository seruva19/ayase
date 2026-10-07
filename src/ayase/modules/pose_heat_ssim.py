"""PoseHeat-SSIM for pose fidelity against a frame-corresponding reference.

This reference-based metric is intended for paired images or strictly aligned
videos. Video inputs must have the same frame count, frame rate, and resolution;
frames are compared by index without temporal resampling. RTMLib Wholebody
supplies 133 COCO-WholeBody keypoints per detected person — the same DWPose
keypoint set the paper uses.

Published definition (DanceTogether, arXiv:2505.18078, Track-2 metrics):
keypoints are "isotropically scale–shift aligned via a similarity transform",
then each frame's keypoint arrays ``X_t, X̂_t ∈ R^{P×J×2}`` are rasterised into
Gaussian heatmaps of size H×W with σ=4 px, and the score is the mean over frames
of ``SSIM(H(X̂_t), H(X_t))``. The published protocol covers P persons per frame;
no reference implementation was released, so the correspondence and fit details
below are documented local choices:

  * persons are matched across the two frames greedily by confident-joint
    centroid distance (gated at ``person_match_frac`` of the frame diagonal);
  * the similarity transform (proper rotation + isotropic scale + translation,
    Umeyama least-squares) is fitted on the union of jointly confident joints
    of matched person pairs, aligning the reference into the generated frame;
  * heatmaps rasterise each side's own confident joints of ALL detected
    persons (missing confident joints penalise the score — as published);
  * heatmap composition is pixelwise max.

``pose_heat_ssim`` and ``pose_heat_ssim_coverage`` are in [0, 1], higher is
better; coverage is the fraction of scanned frames that were scored. Frames
with no detected person on either side, or too few corresponding confident
joints to fit the transform, are excluded. It is not an identity or handedness
metric: the aggregate heatmap gives the 68 face landmarks more samples than any
other body part, and sparse-map SSIM can remain high after a mirror flip.
"""

from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

WHOLEBODY_KEYPOINTS = 133
DWPose_REVISION = "f7c16a3d45ad3783db41471848c80fbc281cabac"
DWPose_REPO = "yzd-v/DWPose"
DWPose_BASE_URL = (
    "https://huggingface.co/yzd-v/DWPose/resolve/"
    "f7c16a3d45ad3783db41471848c80fbc281cabac/"
)
DWPose_DETECTOR = "yolox_l.onnx"
DWPose_POSE = "dw-ll_ucoco_384.onnx"
DWPose_DETECTOR_SHA256 = "7860ae79de6c89a3c1eb72ae9a2756c0ccfbe04b7791bb5880afabd97855a411"
DWPose_POSE_SHA256 = "724f4ff2439ed61afb86fb8a1951ec39c6220682803b4a8bd4f598cd913b1843"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verified_dwpose_asset(filename: str, expected_sha256: str, models_dir: str) -> Path:
    """Download one pinned official DWPose asset and enforce its published hash."""

    from ayase.config import download_model_file

    relative = f"dwpose/{filename}"
    path = Path(download_model_file(relative, DWPose_BASE_URL + filename, models_dir))
    if _sha256(path) == expected_sha256:
        return path
    # A stale/partial cache entry must not be trusted merely because it exists.
    path.unlink(missing_ok=True)
    path = Path(download_model_file(relative, DWPose_BASE_URL + filename, models_dir))
    actual = _sha256(path)
    if actual != expected_sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"DWPose asset hash mismatch for {filename}: expected {expected_sha256}, got {actual}"
        )
    return path


def jointly_valid_keypoints(
    reference_points: np.ndarray,
    reference_scores: np.ndarray,
    generated_points: np.ndarray,
    generated_scores: np.ndarray,
    confidence_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return corresponding finite joints confident in both poses."""

    reference_points = np.asarray(reference_points, dtype=np.float64)
    generated_points = np.asarray(generated_points, dtype=np.float64)
    reference_scores = np.asarray(reference_scores, dtype=np.float64).reshape(-1)
    generated_scores = np.asarray(generated_scores, dtype=np.float64).reshape(-1)
    if reference_points.ndim != 2 or generated_points.ndim != 2:
        return np.empty((0, 2)), np.empty((0, 2))
    if reference_points.shape[1:] != (2,) or generated_points.shape[1:] != (2,):
        return np.empty((0, 2)), np.empty((0, 2))

    count = min(
        len(reference_points),
        len(reference_scores),
        len(generated_points),
        len(generated_scores),
        WHOLEBODY_KEYPOINTS,
    )
    reference_points = reference_points[:count]
    generated_points = generated_points[:count]
    valid = (
        (reference_scores[:count] >= confidence_threshold)
        & (generated_scores[:count] >= confidence_threshold)
        & np.isfinite(reference_points).all(axis=1)
        & np.isfinite(generated_points).all(axis=1)
    )
    return reference_points[valid], generated_points[valid]


def fit_isotropic_scale_translation(
    source: np.ndarray, target: np.ndarray
) -> Optional[Tuple[float, np.ndarray]]:
    """Least-squares fit of ``target ~= scale * source + translation``.

    Scale is constrained to be positive so the fit cannot introduce a hidden
    reflection/180-degree rotation. Degenerate source configurations are not
    scoreable.
    """

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        return None
    if len(source) < 2 or not np.isfinite(source).all() or not np.isfinite(target).all():
        return None

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    denominator = float(np.sum(centered_source * centered_source))
    if denominator <= np.finfo(np.float64).eps:
        return None
    scale = float(np.sum(centered_source * (target - target_center)) / denominator)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    translation = target_center - scale * source_center
    return scale, translation


def render_pose_heatmap(
    points: np.ndarray, frame_shape: Sequence[int], sigma: float = 4.0
) -> np.ndarray:
    """Render unit-peak Gaussian joints with deterministic max composition."""

    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0 or sigma <= 0:
        raise ValueError("frame dimensions and sigma must be positive")
    heatmap = np.zeros((height, width), dtype=np.float32)
    radius = max(1, int(np.ceil(4.0 * sigma)))
    for x, y in np.asarray(points, dtype=np.float64):
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x0 = max(0, int(np.floor(x)) - radius)
        x1 = min(width, int(np.floor(x)) + radius + 1)
        y0 = max(0, int(np.floor(y)) - radius)
        y1 = min(height, int(np.floor(y)) + radius + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
        gaussian = np.exp(-((grid_x - x) ** 2 + (grid_y - y) ** 2) / (2.0 * sigma**2))
        np.maximum(heatmap[y0:y1, x0:x1], gaussian, out=heatmap[y0:y1, x0:x1])
    return heatmap


def _confident_joints(
    points: np.ndarray, scores: np.ndarray, confidence_threshold: float
) -> np.ndarray:
    """One person's joints that are confident and finite."""
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    count = min(len(points), len(scores), WHOLEBODY_KEYPOINTS)
    if count == 0 or points.ndim != 2 or points.shape[1] != 2:
        return np.empty((0, 2))
    mask = (scores[:count] >= confidence_threshold) & np.isfinite(points[:count]).all(axis=1)
    return points[:count][mask]


def fit_similarity_2d(
    source: np.ndarray, target: np.ndarray
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """Umeyama least-squares similarity fit ``target ~= s * R @ source + t``.

    Returns ``(R, s, t)`` with ``R`` a proper rotation (det=+1, no reflection)
    and ``s > 0``, or ``None`` for degenerate input. This is the SIM3 alignment
    the paper applies to all Track-2 keypoints.
    """

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        return None
    n = len(source)
    if n < 2 or not np.isfinite(source).all() or not np.isfinite(target).all():
        return None

    src_c = source.mean(axis=0)
    tgt_c = target.mean(axis=0)
    x = (source - src_c).T  # (2, N)
    y = (target - tgt_c).T  # (2, N)
    var = float((x * x).sum())
    if var <= np.finfo(np.float64).eps:
        return None

    cov = (y @ x.T) / n  # (2, 2)
    try:
        u, svals, vt = np.linalg.svd(cov)
    except np.linalg.LinAlgError:
        return None
    d = np.diag([1.0, 1.0 if np.linalg.det(u @ vt) >= 0 else -1.0])
    rotation = u @ d @ vt
    # Umeyama: s = tr(Σ D) / var_normalized, with var_normalized = var / n.
    scale = float((svals * np.diag(d)).sum() * n / var)
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    translation = tgt_c - scale * (rotation @ src_c)
    return rotation, scale, translation


def match_persons(
    reference_conf: Sequence[np.ndarray],
    generated_conf: Sequence[np.ndarray],
    frame_shape: Sequence[int],
    match_frac: float = 0.25,
) -> List[Tuple[int, int]]:
    """Greedy centroid matching of detected persons across the two frames.

    Returns ``[(ref_index, gen_index), ...]`` — each generated person paired
    with the nearest unused reference person within ``match_frac`` of the
    frame diagonal. The paper's association procedure is unpublished; centroid
    matching is the documented local choice for the alignment fit only (the
    heatmap rasterises all persons regardless of matching).
    """

    def centroid(joints: np.ndarray) -> Optional[np.ndarray]:
        return joints.mean(axis=0) if len(joints) else None

    diag = float(np.hypot(*frame_shape[:2])) + 1e-9
    gate = diag * float(match_frac)
    gen_centroids = [centroid(p) for p in generated_conf]
    used: set = set()
    pairs: List[Tuple[int, int]] = []
    for gi, gc in enumerate(gen_centroids):
        if gc is None:
            continue
        best_i, best_d = -1, gate
        for ri, rp in enumerate(reference_conf):
            if ri in used:
                continue
            rc = centroid(rp)
            if rc is None:
                continue
            d = float(np.linalg.norm(gc - rc))
            if d <= best_d:
                best_i, best_d = ri, d
        if best_i >= 0:
            used.add(best_i)
            pairs.append((best_i, gi))
    return pairs


def pose_heat_ssim_frame(
    reference_persons: Sequence[Tuple[np.ndarray, np.ndarray]],
    generated_persons: Sequence[Tuple[np.ndarray, np.ndarray]],
    frame_shape: Sequence[int],
    *,
    confidence_threshold: float = 0.3,
    sigma: float = 4.0,
    min_joints: int = 3,
    match_frac: float = 0.25,
) -> Optional[float]:
    """Published PoseHeat-SSIM for one frame over P detected persons.

    ``reference_persons`` / ``generated_persons`` are lists of
    ``(keypoints, scores)`` — one ``(133, 2)``/``(133,)`` pair per detected
    person. Returns ``None`` when no person or no alignment correspondence
    exists on either side.
    """

    ref_conf = [
        _confident_joints(p, s, confidence_threshold) for p, s in reference_persons
    ]
    gen_conf = [
        _confident_joints(p, s, confidence_threshold) for p, s in generated_persons
    ]
    if not any(len(p) for p in ref_conf) or not any(len(p) for p in gen_conf):
        return None

    pairs = match_persons(ref_conf, gen_conf, frame_shape, match_frac)
    corr_ref: List[np.ndarray] = []
    corr_gen: List[np.ndarray] = []
    for ri, gi in pairs:
        both_ref, both_gen = jointly_valid_keypoints(
            reference_persons[ri][0],
            reference_persons[ri][1],
            generated_persons[gi][0],
            generated_persons[gi][1],
            confidence_threshold,
        )
        corr_ref.append(both_ref)
        corr_gen.append(both_gen)
    if not corr_ref:
        return None
    source = np.concatenate(corr_ref)
    target = np.concatenate(corr_gen)
    if len(source) < max(2, int(min_joints)):
        return None
    fit = fit_similarity_2d(source, target)
    if fit is None:
        return None
    rotation, scale, translation = fit

    ref_all = np.concatenate([p for p in ref_conf if len(p)])
    gen_all = np.concatenate([p for p in gen_conf if len(p)])
    aligned_ref = scale * (ref_all @ rotation.T) + translation

    reference_heatmap = render_pose_heatmap(aligned_ref, frame_shape, sigma)
    generated_heatmap = render_pose_heatmap(gen_all, frame_shape, sigma)

    from skimage.metrics import structural_similarity

    score = structural_similarity(reference_heatmap, generated_heatmap, data_range=1.0)
    return float(np.clip(score, 0.0, 1.0))


def pose_heat_ssim_score(
    reference_points: np.ndarray,
    reference_scores: np.ndarray,
    generated_points: np.ndarray,
    generated_scores: np.ndarray,
    frame_shape: Sequence[int],
    *,
    confidence_threshold: float = 0.3,
    sigma: float = 4.0,
    min_joints: int = 3,
) -> Optional[float]:
    """Compute one aligned PoseHeat-SSIM frame score, or ``None`` if invalid.

    Single-person convenience wrapper over :func:`pose_heat_ssim_frame`.
    """

    return pose_heat_ssim_frame(
        [(reference_points, reference_scores)],
        [(generated_points, generated_scores)],
        frame_shape,
        confidence_threshold=confidence_threshold,
        sigma=sigma,
        min_joints=min_joints,
    )


def all_wholebody_poses(output: Any) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Normalize RTMLib Wholebody output into ``[(keypoints, scores), ...]``.

    Returns every detected 133-joint person (empty list on malformed output).
    """

    if not isinstance(output, (tuple, list)) or len(output) < 2:
        return []
    keypoints = np.asarray(output[0], dtype=np.float64)
    scores = np.asarray(output[1], dtype=np.float64)
    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
    if scores.ndim == 1:
        scores = scores[None, ...]
    if keypoints.ndim != 3 or scores.ndim != 2:
        return []
    if keypoints.shape[0] != scores.shape[0]:
        return []
    if keypoints.shape[1:] != (WHOLEBODY_KEYPOINTS, 2):
        return []
    if scores.shape[1] != WHOLEBODY_KEYPOINTS:
        return []
    return [(keypoints[i], scores[i]) for i in range(keypoints.shape[0])]


def single_wholebody_pose(output: Any) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Normalize RTMLib output and accept exactly one 133-joint person.

    Kept for compatibility; the published P-person protocol scores every
    detected person via :func:`all_wholebody_poses`.
    """

    poses = all_wholebody_poses(output)
    if len(poses) != 1:
        return None
    return poses[0]


class PoseHeatSSIMModule(PipelineModule):
    """Frame-corresponding pose heatmap similarity over all detected persons."""

    name = "pose_heat_ssim"
    provenance = "adapted"
    sources = {
        "pose_heat_ssim": "DanceTogether PoseHeat (arXiv 2505.18078) — https://arxiv.org/abs/2505.18078",
        "pose_heat_ssim_coverage": "DanceTogether PoseHeat (arXiv 2505.18078) — https://arxiv.org/abs/2505.18078",
    }
    deviations = {
        "pose_heat_ssim": "clean-room from the paper's description (no official code released): Umeyama SIM3 alignment and max heatmap composition are local choices; person correspondence is greedy by centroid",
        "pose_heat_ssim_coverage": "clean-room from the paper's description (no official code released): fraction of frames that could be scored",
    }
    description = "PoseHeat-SSIM against an aligned reference (0-1, higher=better)"
    default_config = {
        "device": "auto",
        "confidence_threshold": 0.3,
        "sigma": 4.0,
        "min_joints": 3,
        "person_match_frac": 0.25,  # centroid gate as a fraction of frame diagonal
        "min_matched_frames": 1,
        "models_dir": "models",
        # Container FPS values may differ by tiny floating-point roundoff only.
        "fps_tolerance": 1e-3,
    }
    models = [
        {
            "id": DWPose_REPO,
            "type": "huggingface",
            "task": "Official DWPose detector and 133-keypoint COCO-WholeBody estimator",
            "revision": DWPose_REVISION,
            "files": [DWPose_DETECTOR, DWPose_POSE],
            "size": "351.1 MB total",
            "auto_download": True,
            "license": "Apache-2.0",
            "notes": (
                f"SHA-256 {DWPose_DETECTOR}={DWPose_DETECTOR_SHA256}; "
                f"{DWPose_POSE}={DWPose_POSE_SHA256}"
            ),
        },
        {
            "id": "rtmlib>=0.0.13",
            "type": "pip_package",
            "install": "pip install rtmlib",
            "task": "ONNX detector and pose runtime",
        },
    ]
    metric_info = {
        "pose_heat_ssim": "Mean aligned pose-heatmap SSIM (0-1, higher=better)",
        "pose_heat_ssim_coverage": "Fraction of corresponding frames scored (0-1)",
    }
    metric_groups = {
        "pose_heat_ssim": "fr_quality",
        "pose_heat_ssim_coverage": "fr_quality",
    }

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.3))
        self.sigma = float(self.config.get("sigma", 4.0))
        self.min_joints = max(2, int(self.config.get("min_joints", 3)))
        self.min_matched_frames = max(1, int(self.config.get("min_matched_frames", 1)))
        self.person_match_frac = float(self.config.get("person_match_frac", 0.25))
        self.fps_tolerance = max(0.0, float(self.config.get("fps_tolerance", 1e-3)))
        self._backend_available = False
        self._wholebody = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            from rtmlib import Wholebody  # noqa: F401

            requested = str(self.config.get("device", "auto"))
            if requested == "auto":
                from ayase.runtime import resolve_torch_device

                requested = str(resolve_torch_device("auto"))
            self._device = "cuda" if requested.startswith("cuda") else "cpu"
            self._backend_available = True
        except ImportError:
            logger.warning("PoseHeat-SSIM unavailable: install rtmlib")
        except Exception as exc:
            logger.warning("PoseHeat-SSIM backend check failed: %s", exc)

    def _get_backend(self):
        """Return an injected backend or a cached RTMLib Wholebody instance."""

        if self._wholebody is not None:
            return self._wholebody
        if not self._backend_available:
            return None

        from ayase.runtime import shared_runtime_resource

        def build():
            from rtmlib import Wholebody

            models_dir = str(self.config.get("models_dir", "models"))
            detector = verified_dwpose_asset(
                DWPose_DETECTOR, DWPose_DETECTOR_SHA256, models_dir
            )
            pose = verified_dwpose_asset(DWPose_POSE, DWPose_POSE_SHA256, models_dir)

            return Wholebody(
                det=str(detector),
                pose=str(pose),
                det_input_size=(640, 640),
                pose_input_size=(288, 384),
                to_openpose=False,
                backend="onnxruntime",
                device=self._device,
            )

        self._wholebody = shared_runtime_resource(
            self,
            (
                "pose_heat_ssim_dwpose",
                DWPose_REVISION,
                "onnxruntime",
                self._device,
                False,
            ),
            build,
        )
        return self._wholebody

    def _score_pair(self, reference_frame: np.ndarray, generated_frame: np.ndarray, backend) -> Optional[float]:
        if reference_frame.shape[:2] != generated_frame.shape[:2]:
            return None
        reference_persons = all_wholebody_poses(backend(np.ascontiguousarray(reference_frame)))
        generated_persons = all_wholebody_poses(backend(np.ascontiguousarray(generated_frame)))
        if not reference_persons or not generated_persons:
            return None
        return pose_heat_ssim_frame(
            reference_persons,
            generated_persons,
            generated_frame.shape[:2],
            confidence_threshold=self.confidence_threshold,
            sigma=self.sigma,
            min_joints=self.min_joints,
            match_frac=self.person_match_frac,
        )

    def _compare_images(self, generated: Path, reference: Path, backend) -> Optional[dict[str, float]]:
        import cv2

        generated_frame = cv2.imread(str(generated))
        reference_frame = cv2.imread(str(reference))
        if generated_frame is None or reference_frame is None:
            return None
        score = self._score_pair(reference_frame, generated_frame, backend)
        if score is None:
            return None
        return {"mean": score, "coverage": 1.0}

    def _compare_videos(self, generated: Path, reference: Path, backend) -> Optional[dict[str, float]]:
        import cv2

        generated_cap = cv2.VideoCapture(str(generated))
        reference_cap = cv2.VideoCapture(str(reference))
        try:
            if not generated_cap.isOpened() or not reference_cap.isOpened():
                return None
            generated_meta = self._video_metadata(generated_cap)
            reference_meta = self._video_metadata(reference_cap)
            if generated_meta is None or reference_meta is None:
                return None
            if generated_meta[:3] != reference_meta[:3]:
                return None
            if abs(generated_meta[3] - reference_meta[3]) > self.fps_tolerance:
                return None

            frame_count = generated_meta[0]
            scores = []
            for _ in range(frame_count):
                generated_ok, generated_frame = generated_cap.read()
                reference_ok, reference_frame = reference_cap.read()
                if not generated_ok or not reference_ok:
                    return None
                try:
                    score = self._score_pair(reference_frame, generated_frame, backend)
                except Exception as exc:
                    logger.debug("PoseHeat-SSIM skipped a frame after pose failure: %s", exc)
                    continue
                if score is not None:
                    scores.append(score)

            if len(scores) < self.min_matched_frames:
                return None
            return {
                "mean": float(np.mean(scores)),
                "coverage": len(scores) / float(frame_count),
            }
        finally:
            generated_cap.release()
            reference_cap.release()

    @staticmethod
    def _video_metadata(capture) -> Optional[Tuple[int, int, int, float]]:
        import cv2

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0 or width <= 0 or height <= 0 or fps <= 0.0:
            return None
        return frame_count, width, height, fps

    def process(self, sample: Sample) -> Sample:
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        backend = self._get_backend()
        if backend is None:
            return sample

        try:
            if sample.is_video:
                result = self._compare_videos(Path(sample.path), Path(reference), backend)
            else:
                result = self._compare_images(Path(sample.path), Path(reference), backend)
            if result is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.pose_heat_ssim = result["mean"]
            sample.quality_metrics.pose_heat_ssim_coverage = result["coverage"]
        except Exception as exc:
            logger.warning("PoseHeat-SSIM failed for %s: %s", sample.path, exc)
        return sample

"""CamI2V camera-trajectory adherence for camera-controlled video generation.

Re-estimates camera poses from a generated video and compares them against a
target camera trajectory using the metrics defined by CamI2V (arXiv 2410.15957,
https://github.com/ZGCTroy/CamI2V, file ``evaluation/glomap_evaluation.py``):

  * **RotErr**  — summed relative-rotation geodesic angle error (radians).
  * **TransErr** — summed relative-translation L2 error after scale alignment.
  * **CamMC**   — camera-motion consistency: summed L2 difference of the 3x4
                  relative pose matrices (rotation + translation combined).

The error definitions are taken verbatim from CamI2V (do not re-derive):

    calc_roterr(R1, R2)   = acos( clamp( (trace(R1^T @ R2) - 1) / 2, -1, 1 ) )   # per pose, radians
    calc_transerr(t1, t2) = || t2 - t1 ||_2                                       # per pose
    calc_cammc(RT1, RT2)  = || (RT2 - RT1).reshape(12) ||_2                        # per pose, 3x4

Relative poses follow CamI2V's ``relative_pose(c2w, mode="left")``: frame 0 is
the identity and frame i is ``inv(P_0) @ P_i``. Translations are scale-normalised
with CamI2V's ``normalize_t`` (each trajectory divided by its own maximum
per-frame translation norm) before TransErr / CamMC, which removes the global
monocular scale ambiguity of the estimated trajectory. The per-pose errors are
summed over the relative poses, in radians for RotErr exactly as upstream.

Pose estimation follows the upstream ``run_glomap`` pipeline — COLMAP
``feature_extractor`` + ``sequential_matcher`` feeding the ``glomap`` binary as
the global mapper — when both binaries are on PATH (upstream stage options are
reproduced verbatim). A plain COLMAP incremental ``mapper`` is used when only
``colmap`` is present (documented deviation: upstream always runs glomap).
``pose_backend="vggt"`` is an explicitly opt-in learned estimator
(facebook/VGGT-1B) — a documented deviation from upstream SfM; it is never
selected automatically unless ``pose_backend="auto"`` and no SfM binaries exist.

Upstream passes the ground-truth SIMPLE_PINHOLE intrinsics ``f,cx,cy`` into the
reconstruction. Ayase takes optional per-sample intrinsics from the trajectory
payload (``"intrinsics": [f, cx, cy]`` or ``{"fx","fy","cx","cy"}``); without
them COLMAP's default focal-length heuristic is used (documented deviation).
When SfM drops frames, the registered image names are mapped back to their frame
indices and only those target poses are compared — no prefix truncation.

Upstream additionally reports TransErr_abs/CamMC_abs between two re-estimated
videos with DepthAnything-V2 metric scaling; Ayase's input is a target
trajectory rather than a second video, so only the trajectory metrics are
computed.

Target-trajectory convention: a per-sample JSON list of camera-to-world (c2w)
4x4 (or 3x4) matrices, one per sampled frame. It is read from either
``getattr(sample, "metadata", {})[trajectory_key]`` or a sidecar file next to the
video named ``<stem>.camera.json`` (suffix configurable via ``trajectory_suffix``;
``<name>.camera.json`` is also accepted). The JSON may be a bare list of matrices
or a dict whose ``trajectory_key`` (default ``"camera_trajectory"``) holds the
list, optionally alongside an ``"intrinsics"`` entry. When no trajectory is
found the metrics are left unset.
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure trajectory math (module-level so it is unit-testable without a model).
# ---------------------------------------------------------------------------


def _pad_to_44(mat: np.ndarray) -> np.ndarray:
    """Return a 4x4 homogeneous matrix from a 3x4 or 4x4 pose."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape == (4, 4):
        return mat
    out = np.eye(4, dtype=np.float64)
    out[:3, :4] = mat[:3, :4]
    return out


def _relative_pose_left(poses: np.ndarray) -> np.ndarray:
    """CamI2V ``relative_pose(rt, mode="left")``: anchor every pose to frame 0.

    Frame 0 becomes the identity; frame ``i`` becomes ``inv(P_0) @ P_i``.
    """
    poses = np.asarray(poses, dtype=np.float64)
    n = poses.shape[0]
    inv0 = np.linalg.inv(poses[0])
    rel = np.empty((n, 4, 4), dtype=np.float64)
    rel[0] = np.eye(4, dtype=np.float64)
    for i in range(1, n):
        rel[i] = inv0 @ poses[i]
    return rel


def _normalize_translations(rel: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """CamI2V ``normalize_t``: divide translations by the max per-frame norm."""
    rel = np.array(rel, dtype=np.float64, copy=True)
    trans = rel[:, :3, 3]
    scale = float(np.max(np.linalg.norm(trans, axis=1))) + eps
    rel[:, :3, 3] = trans / scale
    return rel


def _rotation_geodesic(r1: np.ndarray, r2: np.ndarray) -> float:
    """CamI2V ``calc_roterr`` for a single pair: geodesic angle in radians.

    ``acos( clamp( (trace(R1^T @ R2) - 1) / 2, -1, 1 ) )``.
    """
    cos = (float(np.trace(np.asarray(r1).T @ np.asarray(r2))) - 1.0) / 2.0
    cos = min(1.0, max(-1.0, cos))
    return float(np.arccos(cos))


def compute_trajectory_errors(
    estimated_c2w: np.ndarray,
    target_c2w: np.ndarray,
    eps: float = 1e-9,
) -> Optional[dict]:
    """Compute CamI2V RotErr (rad) / TransErr / CamMC between two c2w trajectories.

    ``estimated_c2w`` and ``target_c2w`` are ``(T, 4, 4)`` camera-to-world stacks.
    Identical trajectories yield errors of ~0. Returns ``None`` if fewer than two
    corresponding poses are available.
    """
    est = np.asarray(estimated_c2w, dtype=np.float64)
    tgt = np.asarray(target_c2w, dtype=np.float64)
    t = min(est.shape[0], tgt.shape[0])
    if t < 2:
        return None
    est = est[:t]
    tgt = tgt[:t]

    est_rel = _normalize_translations(_relative_pose_left(est), eps)
    tgt_rel = _normalize_translations(_relative_pose_left(tgt), eps)

    rot_err = 0.0
    trans_err = 0.0
    cammc = 0.0
    for i in range(t):
        rot_err += _rotation_geodesic(est_rel[i, :3, :3], tgt_rel[i, :3, :3])
        trans_err += float(np.linalg.norm(tgt_rel[i, :3, 3] - est_rel[i, :3, 3]))
        cammc += float(np.linalg.norm((tgt_rel[i, :3, :4] - est_rel[i, :3, :4]).reshape(-1)))

    return {
        "rot_err": float(rot_err),  # radians, summed over poses — upstream unit
        "trans_err": float(trans_err),
        "cammc": float(cammc),
    }


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert a (w, x, y, z) unit quaternion to a 3x3 rotation matrix."""
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5 + 1e-12
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz + qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _parse_colmap_images_txt(path: str) -> dict:
    """Parse a COLMAP ``images.txt`` into ``{image_name: 4x4 c2w matrix}``.

    Each image occupies two lines; the first is
    ``IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME`` giving the world-to-camera
    pose, which is inverted to camera-to-world.
    """
    poses: dict = {}
    with open(path, "r", encoding="utf-8") as handle:
        lines = [ln for ln in handle.readlines() if ln.strip() and not ln.startswith("#")]
    i = 0
    while i < len(lines):
        parts = lines[i].split()
        if len(parts) >= 10:
            qw, qx, qy, qz, tx, ty, tz = (float(v) for v in parts[1:8])
            name = parts[9]
            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = _quat_to_rot(qw, qx, qy, qz)
            w2c[:3, 3] = (tx, ty, tz)
            poses[name] = np.linalg.inv(w2c)
        i += 2  # skip the 2D-point line that follows every pose line
    return poses


class CameraTrajectoryModule(PipelineModule):
    name = "camera_trajectory"
    provenance = "adapted"
    sources = {
        "camera_rot_error": "CamI2V RotErr/TransErr/CamMC — https://github.com/ZGCTroy/CamI2V/blob/main/evaluation/glomap_evaluation.py",
        "camera_traj_consistency": "CamI2V RotErr/TransErr/CamMC — https://github.com/ZGCTroy/CamI2V/blob/main/evaluation/glomap_evaluation.py",
        "camera_trans_error": "CamI2V RotErr/TransErr/CamMC — https://github.com/ZGCTroy/CamI2V/blob/main/evaluation/glomap_evaluation.py",
    }
    deviations = {
        "camera_rot_error": "CamI2V: GLOMAP reconstruction with GT SIMPLE_PINHOLE intrinsics; without a sidecar intrinsics COLMAP estimates the focal itself. Upstream TransErr_abs/CamMC_abs (depth-scale from the second video) do not apply to the target trajectory",
        "camera_traj_consistency": "CamI2V: GLOMAP reconstruction with GT SIMPLE_PINHOLE intrinsics; without a sidecar intrinsics COLMAP estimates the focal itself. Upstream TransErr_abs/CamMC_abs do not apply to the target trajectory",
        "camera_trans_error": "CamI2V: GLOMAP reconstruction with GT SIMPLE_PINHOLE intrinsics; without a sidecar intrinsics COLMAP estimates the focal itself. Upstream TransErr_abs/CamMC_abs do not apply to the target trajectory",
    }
    description = (
        "CamI2V camera-trajectory adherence (RotErr/TransErr/CamMC) via GLOMAP "
        "pose re-estimation against a target trajectory"
    )
    default_config = {
        "num_frames": 0,  # 0 = all frames (upstream reads the whole video)
        "trajectory_key": "camera_trajectory",
        "trajectory_suffix": ".camera.json",
        "pose_backend": "auto",  # "auto" | "glomap" | "colmap" | "vggt"
        "model_id": "facebook/VGGT-1B",
        "sfm_timeout": 600,
    }
    metric_groups = {
        "camera_rot_error": "motion",
        "camera_trans_error": "motion",
        "camera_traj_consistency": "motion",
    }
    metric_info = {
        "camera_rot_error": (
            "CamI2V RotErr: summed relative-rotation geodesic error (radians) between "
            "the re-estimated and target camera trajectories (lower is better)"
        ),
        "camera_trans_error": (
            "CamI2V TransErr: summed relative-translation L2 error after scale alignment "
            "(lower is better)"
        ),
        "camera_traj_consistency": (
            "CamI2V CamMC: summed L2 difference of the 3x4 relative pose matrices "
            "(lower is better)"
        ),
    }
    models = [
        {
            "id": "facebook/VGGT-1B",
            "type": "huggingface",
            "task": "Learned camera pose estimation (pose_backend='vggt' opt-in)",
        },
        {
            "id": "glomap",
            "type": "other",
            "task": "Global structure-from-motion mapper (upstream backend, external binary)",
        },
        {
            "id": "colmap",
            "type": "other",
            "task": "SfM feature extraction/matching frontend (external binary)",
        },
    ]

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._ml_available = False
        self._vggt = None
        self._sfm_bin: Optional[dict] = None
        self._device = "cpu"
        self._intrinsics: Optional[Tuple[float, float, float]] = None

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        from ayase.runtime import resolve_torch_device

        self._device = resolve_torch_device(self.config.get("device", "auto"))
        backend = str(self.config.get("pose_backend", "auto")).lower()

        # Canonical tier: upstream's colmap feature/match + glomap mapper.
        sfm = self._detect_sfm_binaries()
        if backend in ("auto", "glomap") and sfm and sfm.get("glomap"):
            self._sfm_bin = sfm
            self._backend = "glomap"
            self._ml_available = True
            logger.info("camera_trajectory: using upstream GLOMAP pipeline %s", sfm)
            return
        if backend in ("auto", "colmap") and sfm:
            self._sfm_bin = sfm
            self._backend = "colmap"
            self._ml_available = True
            logger.info("camera_trajectory: glomap absent; using colmap mapper %s", sfm)
            return
        if backend in ("glomap", "colmap"):
            logger.info(
                "camera_trajectory: requested backend %r unavailable (missing binaries)",
                backend,
            )
            self._backend = "unavailable"
            return

        # Opt-in learned backend: VGGT (documented deviation from upstream SfM).
        if backend in ("auto", "vggt"):
            try:
                model = self._load_vggt()
                if model is not None:
                    self._vggt = model
                    self._backend = "vggt"
                    self._ml_available = True
                    logger.info(
                        "camera_trajectory: loaded VGGT (%s) on %s",
                        self.config.get("model_id"),
                        self._device,
                    )
                    return
            except ImportError as exc:
                logger.info("camera_trajectory: VGGT unavailable (missing dependency): %s", exc)
            except Exception as exc:  # pragma: no cover - depends on optional weights
                logger.info("camera_trajectory: VGGT load failed: %s", exc)

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "camera_trajectory unavailable: no GLOMAP/COLMAP binaries on PATH and VGGT "
            "not installed; camera_rot_error / camera_trans_error / "
            "camera_traj_consistency will not be populated by this module."
        )

    def _load_vggt(self):
        """Load and cache the shared VGGT model (raises ImportError if absent)."""
        import torch  # noqa: F401  (ensures torch is present before touching vggt)
        from vggt.models.vggt import VGGT

        from ayase.runtime import shared_runtime_resource

        model_id = self.config.get("model_id", "facebook/VGGT-1B")

        def build():
            return VGGT.from_pretrained(model_id).to(self._device).eval()

        return shared_runtime_resource(self, ("vggt", self._device), build)

    @staticmethod
    def _detect_sfm_binaries() -> Optional[dict]:
        """Return ``{"colmap": path, "glomap": path|None}`` or ``None``."""
        import shutil

        colmap = shutil.which("colmap")
        if not colmap:
            return None
        return {"colmap": colmap, "glomap": shutil.which("glomap")}

    # -- processing ---------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            target = self._load_target_trajectory(sample)
            if target is None or len(target) < 2:
                return sample

            # Upstream scores every frame of the video; ``num_frames`` caps how
            # many positions are compared. Estimated and target poses are aligned
            # by uniform sampling to the SAME count at matched fractional
            # positions, so estimated[i] <-> target[i] refer to the same fraction
            # of each sequence.
            cap = int(self.config.get("num_frames", 0))
            n_compare = len(target) if cap <= 1 else min(len(target), cap)
            if n_compare < len(target):
                idx = np.linspace(0, len(target) - 1, n_compare).round().astype(int)
                target = target[idx]

            estimated, reg_idx = self._estimate_poses(sample, n_compare)
            if estimated is None or len(estimated) < 2:
                return sample

            # SfM may fail to register some frames; align by the registered
            # frame indices instead of truncating a prefix (upstream compares
            # the same positional indices of the GT trajectory).
            if reg_idx is not None and len(reg_idx) != len(target):
                target = target[reg_idx]
                if len(target) < 2:
                    return sample

            errs = compute_trajectory_errors(estimated, target)
            if errs is None:
                return sample

            sample.quality_metrics.camera_rot_error = round(errs["rot_err"], 6)
            sample.quality_metrics.camera_trans_error = round(errs["trans_err"], 6)
            sample.quality_metrics.camera_traj_consistency = round(errs["cammc"], 6)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("camera_trajectory processing failed: %s", exc)

        return sample

    # -- target trajectory --------------------------------------------------

    def _load_target_trajectory(self, sample: Sample) -> Optional[np.ndarray]:
        key = self.config.get("trajectory_key", "camera_trajectory")

        raw = None
        self._intrinsics = None
        meta = getattr(sample, "metadata", None)
        if isinstance(meta, dict) and key in meta:
            raw = meta[key]
            self._intrinsics = self._parse_intrinsics(meta.get("intrinsics"))
        if raw is None:
            raw = self._read_trajectory_sidecar(sample, key)
        if raw is None:
            return None
        return self._parse_trajectory(raw)

    def _read_trajectory_sidecar(self, sample: Sample, key: str):
        import json
        from pathlib import Path

        path = Path(sample.path)
        suffix = self.config.get("trajectory_suffix", ".camera.json")
        candidates = [
            path.parent / (path.stem + suffix),  # clip.camera.json
            path.parent / (path.name + suffix),  # clip.mp4.camera.json
        ]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    data = json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(data, dict):
                        self._intrinsics = self._parse_intrinsics(data.get("intrinsics"))
                        return data.get(key, data.get("trajectory"))
                    return data
            except Exception as exc:  # pragma: no cover - malformed sidecar
                logger.debug("camera_trajectory: failed to read sidecar %s: %s", candidate, exc)
        return None

    @staticmethod
    def _parse_intrinsics(raw) -> Optional[Tuple[float, float, float]]:
        """Coerce ``[f, cx, cy]`` / ``{"f","cx","cy"}`` / ``{"fx","fy","cx","cy"}``
        into the ``(f, cx, cy)`` triple upstream feeds SIMPLE_PINHOLE."""
        if raw is None:
            return None
        try:
            if isinstance(raw, dict):
                f = float(raw.get("f", raw.get("fx")))
                cx = float(raw["cx"])
                cy = float(raw["cy"])
                if f <= 0:
                    return None
                return (f, cx, cy)
            arr = np.asarray(raw, dtype=np.float64).reshape(-1)
            if arr.size == 3 and arr[0] > 0:
                return (float(arr[0]), float(arr[1]), float(arr[2]))
        except Exception:
            return None
        return None

    @staticmethod
    def _parse_trajectory(raw) -> Optional[np.ndarray]:
        """Coerce a raw trajectory into a ``(T, 4, 4)`` c2w stack, or ``None``."""
        try:
            arr = np.asarray(raw, dtype=np.float64)
        except Exception:
            return None
        if arr.ndim == 3 and arr.shape[1:] == (4, 4):
            poses = arr
        elif arr.ndim == 3 and arr.shape[1:] == (3, 4):
            poses = np.stack([_pad_to_44(m) for m in arr])
        elif arr.ndim == 2 and arr.shape[1] == 16:
            poses = arr.reshape(-1, 4, 4)
        elif arr.ndim == 2 and arr.shape[1] == 12:
            poses = np.stack([_pad_to_44(m.reshape(3, 4)) for m in arr])
        else:
            return None
        if poses.shape[0] < 2:
            return None
        return poses

    # -- pose estimation ----------------------------------------------------

    def _estimate_poses(
        self, sample: Sample, num_frames: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Return ``(c2w_poses, registered_frame_indices)``.

        ``registered_frame_indices`` is ``None`` when every sampled frame has a
        pose (dense estimators like VGGT); for SfM it lists the frame indices
        that were successfully registered.
        """
        from ayase.image import sample_frames

        # ``num_frames`` is already the decided compare count (>= 2): every
        # sampled position maps 1:1 onto a target pose.
        frames = sample_frames(sample.path, max_frames=max(2, num_frames), color="rgb")
        if len(frames) < 2:
            return None, None
        # sample_frames returns read-only views over the shared cache; make
        # writable contiguous copies before handing them to torch / cv2.
        frames = [np.ascontiguousarray(f) for f in frames]

        if self._backend == "vggt":
            poses = self._estimate_poses_vggt(frames)
            if poses is None:
                return None, None
            return poses, None
        if self._backend in ("glomap", "colmap"):
            return self._estimate_poses_sfm(frames)
        return None, None

    def _estimate_poses_vggt(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        import os
        import tempfile

        import cv2
        import torch
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for idx, frame in enumerate(frames):
                fpath = os.path.join(tmp, f"frame_{idx:04d}.png")
                cv2.imwrite(fpath, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                paths.append(fpath)

            images = load_and_preprocess_images(paths).to(self._device)
            with torch.inference_mode():
                preds = self._vggt(images)
                pose_enc = preds["pose_enc"] if isinstance(preds, dict) else preds
                extrinsic, _intrinsic = pose_encoding_to_extri_intri(
                    pose_enc, images.shape[-2:]
                )
            extr = np.asarray(extrinsic.detach().cpu().float().numpy())

        if extr.ndim == 4:  # (B, S, 3, 4) -> (S, 3, 4)
            extr = extr[0]
        # VGGT extrinsics are OpenCV world-to-camera; invert to camera-to-world.
        c2w = [np.linalg.inv(_pad_to_44(e)) for e in extr]
        return np.stack(c2w) if len(c2w) >= 2 else None

    def _estimate_poses_sfm(
        self, frames: List[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Upstream ``run_glomap``: colmap feature_extractor + sequential_matcher
        + glomap mapper with CamI2V's verbatim stage options."""
        import os
        import subprocess
        import tempfile

        import cv2

        bins = self._sfm_bin or {}
        colmap = bins.get("colmap")
        glomap = bins.get("glomap")
        if not colmap:
            return None, None
        timeout = int(self.config.get("sfm_timeout", 600))

        def run(cmd: List[str]) -> None:
            subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)

        with tempfile.TemporaryDirectory() as tmp:
            img_dir = os.path.join(tmp, "images")
            os.makedirs(img_dir)
            names = []
            for idx, frame in enumerate(frames):
                name = f"frame_{idx:04d}.png"
                cv2.imwrite(os.path.join(img_dir, name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                names.append(name)

            db = os.path.join(tmp, "database.db")
            sparse = os.path.join(tmp, "sparse")
            os.makedirs(sparse)

            # Upstream feature_extractor options (CamI2V evaluation): single
            # SIMPLE_PINHOLE camera + affine-shape/domain-size-pooled SIFT.
            feat_cmd = [
                colmap,
                "feature_extractor",
                "--database_path", db,
                "--image_path", img_dir,
                "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", "SIMPLE_PINHOLE",
                "--SiftExtraction.estimate_affine_shape", "1",
                "--SiftExtraction.domain_size_pooling", "1",
            ]
            if self._intrinsics is not None:
                f, cx, cy = self._intrinsics
                feat_cmd += ["--ImageReader.camera_params", f"{f},{cx},{cy}"]
            run(feat_cmd)
            run(
                [
                    colmap,
                    "sequential_matcher",
                    "--database_path", db,
                    "--SiftMatching.guided_matching", "1",
                    "--SiftMatching.max_num_matches", "65536",
                ]
            )
            if glomap and self._backend == "glomap":
                run(
                    [
                        glomap,
                        "mapper",
                        "--database_path", db,
                        "--image_path", img_dir,
                        "--output_path", sparse,
                        "--output_format", "txt",
                        "--RelPoseEstimation.max_epipolar_error", "4",
                        "--BundleAdjustment.optimize_intrinsics", "0",
                    ]
                )
            else:
                run(
                    [
                        colmap,
                        "mapper",
                        "--database_path", db,
                        "--image_path", img_dir,
                        "--output_path", sparse,
                    ]
                )

            model_dir = os.path.join(sparse, "0")
            if not os.path.isdir(model_dir):
                return None, None
            if self._backend == "colmap" or not os.path.isfile(
                os.path.join(model_dir, "images.txt")
            ):
                run(
                    [
                        colmap,
                        "model_converter",
                        "--input_path", model_dir,
                        "--output_path", model_dir,
                        "--output_type", "TXT",
                    ]
                )
            poses_by_name = _parse_colmap_images_txt(os.path.join(model_dir, "images.txt"))

        ordered = []
        reg_idx = []
        for i, name in enumerate(names):
            if name in poses_by_name:
                ordered.append(poses_by_name[name])
                reg_idx.append(i)
        if len(ordered) < 2:
            return None, None
        return np.stack(ordered), np.asarray(reg_idx, dtype=np.int64)

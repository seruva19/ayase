"""CamI2V-style camera-trajectory adherence for camera-controlled video generation.

Re-estimates camera poses from a generated video with a *real* pose estimator and
compares them against a target camera trajectory using the three metrics defined
by CamI2V (arXiv 2410.15957, https://github.com/ZGCTroy/CamI2V, file
``evaluation/glomap_evaluation.py``):

  * **RotErr**  — summed relative-rotation geodesic angle error (degrees).
  * **TransErr** — summed relative-translation L2 error after scale alignment.
  * **CamMC**   — camera-motion consistency: summed L2 difference of the 3x4
                  relative pose matrices (rotation + translation combined).

The error definitions are taken verbatim from CamI2V (do not re-derive):

    calc_roterr(R1, R2)   = acos( clamp( (trace(R1^T @ R2) - 1) / 2, -1, 1 ) )   # per pose, radians
    calc_transerr(t1, t2) = || t2 - t1 ||_2                                       # per pose
    calc_cammc(RT1, RT2)  = || (RT2 - RT1).reshape(12) ||_2                        # per pose, 3x4

Relative poses follow CamI2V's ``relative_pose(c2w, mode="left")``: frame 0 is the
identity and frame i is ``inv(P_0) @ P_i``. Translations are scale-normalised with
CamI2V's ``normalize_t`` (divide each trajectory's translations by its own maximum
per-frame translation norm) before TransErr / CamMC, which removes the global
monocular scale ambiguity of the estimated trajectory. The per-pose errors are
summed over the relative poses. CamI2V's ``calc_roterr`` returns radians; this
module converts the summed rotation error to degrees to match the documented unit
of the ``camera_rot_error`` field.

Pose-estimation backends (a real model or nothing — the repo policy forbids
heuristic/homography proxy stand-ins):

  * **VGGT** — facebook/VGGT-1B (CVPR 2025), loaded via the ``vggt`` package;
    predicts OpenCV world-to-camera extrinsics that are inverted to camera-to-world.
  * **COLMAP / GLOMAP** — classical structure-from-motion via the ``colmap``
    (and optionally ``glomap``) command-line binaries, when they are on PATH.

If neither backend is available the module sets ``_backend = "unavailable"`` and
leaves the three metrics unset.

Target-trajectory convention: a per-sample JSON list of camera-to-world (c2w)
4x4 (or 3x4) matrices, one per sampled frame. It is read from either
``getattr(sample, "metadata", {})[trajectory_key]`` or a sidecar file next to the
video named ``<stem>.camera.json`` (suffix configurable via ``trajectory_suffix``;
``<name>.camera.json`` is also accepted). The JSON may be a bare list of matrices
or a dict whose ``trajectory_key`` (default ``"camera_trajectory"``) holds the list.
When no trajectory is found the metrics are left unset.
"""

from __future__ import annotations

import logging
from typing import List, Optional

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
    """Compute CamI2V RotErr (deg) / TransErr / CamMC between two c2w trajectories.

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
        "rot_err": float(np.degrees(rot_err)),  # radians -> degrees (camera_rot_error is documented in deg)
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
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
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
        lines = [ln for ln in handle if ln.strip() and not ln.startswith("#")]
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
    description = (
        "CamI2V camera-trajectory adherence (RotErr/TransErr/CamMC) via real pose "
        "re-estimation (VGGT or COLMAP/GLOMAP) against a target trajectory"
    )
    default_config = {
        "num_frames": 16,
        "trajectory_key": "camera_trajectory",
        "trajectory_suffix": ".camera.json",
        "model_id": "facebook/VGGT-1B",
        "colmap_matcher": "sequential",  # or "exhaustive"
        "sfm_timeout": 600,
    }
    metric_groups = {
        "camera_rot_error": "motion",
        "camera_trans_error": "motion",
        "camera_traj_consistency": "motion",
    }
    metric_info = {
        "camera_rot_error": (
            "CamI2V RotErr: summed relative-rotation geodesic error (deg) between the "
            "re-estimated and target camera trajectories (lower is better)"
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

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend: Optional[str] = None
        self._ml_available = False
        self._vggt = None
        self._sfm_bin: Optional[dict] = None
        self._device = "cpu"

    # -- lifecycle ----------------------------------------------------------

    def setup(self) -> None:
        from ayase.runtime import resolve_torch_device

        self._device = resolve_torch_device(self.config.get("device", "auto"))

        # Tier 1: VGGT (learned feed-forward pose estimator).
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

        # Tier 2: COLMAP / GLOMAP command-line structure-from-motion.
        sfm = self._detect_sfm_binaries()
        if sfm is not None:
            self._sfm_bin = sfm
            self._backend = "colmap"
            self._ml_available = True
            logger.info("camera_trajectory: using SfM binaries %s", sfm)
            return

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "camera_trajectory unavailable: neither VGGT nor a COLMAP/GLOMAP binary "
            "was found; camera_rot_error / camera_trans_error / camera_traj_consistency "
            "will not be populated by this module."
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

            estimated = self._estimate_poses(sample, len(target))
            if estimated is None or len(estimated) < 2:
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
        meta = getattr(sample, "metadata", None)
        if isinstance(meta, dict) and key in meta:
            raw = meta[key]
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
                        return data.get(key, data.get("trajectory"))
                    return data
            except Exception as exc:  # pragma: no cover - malformed sidecar
                logger.debug("camera_trajectory: failed to read sidecar %s: %s", candidate, exc)
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

    def _estimate_poses(self, sample: Sample, num_frames: int) -> Optional[np.ndarray]:
        from ayase.image import sample_frames

        frames = sample_frames(sample.path, max_frames=num_frames, color="rgb")
        if len(frames) < 2:
            return None
        # sample_frames returns read-only views over the shared cache; make
        # writable contiguous copies before handing them to torch / cv2.
        frames = [np.ascontiguousarray(f) for f in frames]

        if self._backend == "vggt":
            return self._estimate_poses_vggt(frames)
        if self._backend == "colmap":
            return self._estimate_poses_colmap(frames)
        return None

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

    def _estimate_poses_colmap(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        import os
        import subprocess
        import tempfile

        import cv2

        bins = self._sfm_bin or {}
        colmap = bins.get("colmap")
        glomap = bins.get("glomap")
        if not colmap:
            return None
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

            run([colmap, "feature_extractor", "--database_path", db,
                 "--image_path", img_dir, "--ImageReader.single_camera", "1"])
            matcher = (
                "sequential_matcher"
                if self.config.get("colmap_matcher", "sequential") == "sequential"
                else "exhaustive_matcher"
            )
            run([colmap, matcher, "--database_path", db])
            if glomap:
                run([glomap, "mapper", "--database_path", db,
                     "--image_path", img_dir, "--output_path", sparse])
            else:
                run([colmap, "mapper", "--database_path", db,
                     "--image_path", img_dir, "--output_path", sparse])

            model_dir = os.path.join(sparse, "0")
            if not os.path.isdir(model_dir):
                return None
            run([colmap, "model_converter", "--input_path", model_dir,
                 "--output_path", model_dir, "--output_type", "TXT"])
            poses_by_name = _parse_colmap_images_txt(os.path.join(model_dir, "images.txt"))

        ordered = [poses_by_name[n] for n in names if n in poses_by_name]
        return np.stack(ordered) if len(ordered) >= 2 else None

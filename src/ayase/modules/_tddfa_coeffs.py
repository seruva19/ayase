"""Shared TDDFA (3DDFA_V2) per-frame 3DMM coefficient extraction.

Wraps the vendored ``third_party/idreveal`` stack (RetinaFace detection +
TDDFA MobileNet-1 regression) to produce a per-frame 62-dim coefficient
vector: ``12 pose + 40 shape + 10 expression`` (upstream split). Used by
``aed_apd``, ``head_pose_diversity``, ``head_beat_align``, ``fd_3dmm`` and
``id_reveal``. Weights live in one shared directory under
``models/id_reveal/`` (sha256-verified).
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TDDFA_WEIGHTS = {
    "Resnet50_Final.pth": (
        "https://huggingface.co/akhaliq/RetinaFace-R50/resolve/main/RetinaFace-R50.pth",
        "6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d",
    ),
    "mb1_120x120.pth": (
        "https://huggingface.co/Stable-Human/3ddfa_v2/resolve/main/mb1_120x120.pth",
        "a45a946c6e9b16f8d3cf2e69376da9560a7cf9afae671bebceb7e437a405ea79",
    ),
    "model_idreveal.th": (
        "https://raw.githubusercontent.com/grip-unina/id-reveal/main/model.th",
        "f77f2073ae18e49cc2a8f607bb36b7c4fde98aba95231f35abfb4b67a0b4d421",
    ),
}

# upstream 62-dim split: 12 pose + 40 shape + 10 expression
POSE_SLICE = slice(0, 12)
SHAPE_SLICE = slice(12, 52)
EXPR_SLICE = slice(52, 62)

WEIGHTS_DIR_NAME = "id_reveal"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_tddfa_weights(models_dir: str, names=None) -> Optional[Path]:
    """Download + verify the shared TDDFA/RetinaFace/ID-Reveal weights.

    Returns the resources directory, or None if any required file is missing
    or fails checksum verification.
    """
    from ayase.config import download_model_file

    names = names or list(_TDDFA_WEIGHTS)
    out = None
    for name in names:
        url, digest = _TDDFA_WEIGHTS[name]
        try:
            path = download_model_file(f"{WEIGHTS_DIR_NAME}/{name}", url, models_dir)
        except Exception as e:
            logger.warning("tddfa_coeffs: %s download failed: %s", name, e)
            return None
        if not path.exists() or _sha256(path) != digest:
            logger.warning("tddfa_coeffs: checksum mismatch or missing %s", path)
            return None
        out = path.parent
    return out


class TDDFACoeffExtractor:
    """Per-frame 62-dim 3DMM coefficients for the largest detected face.

    Replicates the upstream detection path: the video is resampled to
    ``fps`` and RetinaFace runs on every frame (upstream ``extract_boxes``
    behaviour); the largest box per frame goes through TDDFA. No tracking —
    per-frame metrics only need the dominant face's coefficients.
    """

    def __init__(self, resources: Path, device: str = "cpu",
                 fps: int = 25, read_stride: int = 96, rec_stride: int = 32,
                 det_size_threshold: int = 75, det_score_threshold: float = 0.7,
                 det_target_size: int = 1280):
        from ayase.third_party.idreveal.grip_unina.id_reveal.util_3dmm import (
            Compute3DMMtracked,
        )
        from ayase.third_party.idreveal.grip_unina.util_face import DetectFace

        self.fps = fps
        self.read_stride = read_stride
        self._det = DetectFace(
            device, str(Path(resources) / "Resnet50_Final.pth"),
            size_threshold=det_size_threshold,
            target_size=det_target_size,
            batch_size=rec_stride,
            score_threshold=det_score_threshold,
            return_frame=True,
        )
        self._mm = Compute3DMMtracked(device, str(resources), return_frame=False)

    def extract(self, video_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Return ``(params, frame_inds)``: one 62-dim row per detected frame.

        Per resampled frame the largest-face detection is kept — the standard
        single-subject convention of the talking-head evals these metrics come
        from.
        """
        from ayase.third_party.idreveal.grip_unina.util_read import (
            ReadingResampledVideo,
        )

        rows = {}  # resampled frame index -> 62-dim params (largest face)
        try:
            with ReadingResampledVideo(str(video_path), self.fps, self.read_stride) as rd:
                ops = [rd, self._det.reset(), self._mm.reset()]
                count = 0
                while True:
                    try:
                        out = count
                        for op in ops:
                            out = op(out)
                    except StopIteration:
                        break
                    count += 1
                    boxes = out.get("boxes") or []
                    coeffs = out.get("3dmm")
                    if coeffs is None or len(coeffs) == 0:
                        continue
                    coeffs = np.asarray(coeffs)
                    for frame_ind, box, coef in zip(out["image_inds"], boxes, coeffs):
                        area = (box[2] - box[0]) * (box[3] - box[1])
                        prev = rows.get(frame_ind)
                        if prev is None or area > prev[0]:
                            rows[frame_ind] = (area, coef)
        except Exception as e:
            logger.warning("tddfa_coeffs: extraction failed on %s: %s",
                           video_path.name, e)
            return None
        if not rows:
            return None
        inds = np.asarray(sorted(rows), dtype=np.int64)
        params = np.stack([rows[i][1] for i in inds], 0).astype(np.float32)
        return params, inds

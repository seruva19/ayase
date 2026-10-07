"""AUCON and PRMSE — action-unit agreement and head-pose RMSE vs driver.

MarioNETte's reenactment evaluation: the driver video and the generated video
are decoded per frame; AUCON is the fraction of frames whose set of active
facial action units coincides between generated and driven frame, and PRMSE is
the RMSE of the head-pose angles (pitch, yaw, roll). AU presence and pose are
estimated with Py-Feat's Detector (Jia et al., arXiv:2104.03509), the AU/pose
backend named alongside OpenFace for this evaluation.

aucon -- fraction of frames with coincident AU sets (0-1, higher=better).
prmse -- pose-angle RMSE vs the driver (lower=better).
Requires ``sample.reference_path`` pointing at the driver video.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_AU_THRESHOLD = 0.5  # Py-Feat AU intensities binarised to presence/absence


def _frames(video_path: Path, fps: float, max_frames: int) -> list:
    """Uniformly subsample RGB frames at ~fps."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    stride = max(1, int(round(src_fps / fps)))
    out = []
    idx = -1
    try:
        while len(out) < max_frames:
            ok = cap.grab()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    return out


class AuconPrmseModule(PipelineModule):
    name = "aucon_prmse"
    description = "AUCON/PRMSE — action-unit and pose agreement with a driver video"
    provenance = "adapted"
    sources = {
        "aucon": "AUCON, MarioNETte (Ha et al., arXiv:1911.08139); Py-Feat backend (arXiv:2104.03509) — https://github.com/cosanlab/py-feat",
        "prmse": "PRMSE, MarioNETte (Ha et al., arXiv:1911.08139); Py-Feat backend (arXiv:2104.03509) — https://github.com/cosanlab/py-feat",
    }
    deviations = {
        "aucon": "AU presence comes from Py-Feat intensities thresholded at 0.5 rather than OpenFace's binary AU labels",
        "prmse": "head-pose angles come from Py-Feat's face model rather than OpenFace",
    }
    requires_reference = True
    default_config = {
        "fps": 5.0,
        "max_frames": 300,
        "au_threshold": _AU_THRESHOLD,
        "device": "auto",
    }
    models = [
        {
            "id": "py-feat",
            "type": "pip_package",
            "install": "pip install py-feat",
            "task": "Py-Feat AU and head-pose detector (weights auto-download)",
        },
    ]
    metric_info = {
        "aucon": "Fraction of frames with coincident active-AU sets vs driver (0-1, higher=better)",
        "prmse": "RMSE of head-pose angles vs driver (lower=better)",
    }
    metric_groups = {"aucon": "face", "prmse": "face"}

    def __init__(self, config=None):
        super().__init__(config)
        self._detector = None
        self._backend = None

    def setup(self) -> None:
        if self.config.get("test_mode"):
            self._backend = "unavailable"
            return
        try:
            from feat import Detector
        except ImportError:
            logger.warning(
                "aucon_prmse: py-feat not installed (pip install py-feat), disabled"
            )
            self._backend = "unavailable"
            return
        device = self.config.get("device", "auto")
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        try:
            self._detector = Detector(device=device)
            self._backend = "pyfeat"
        except Exception as e:
            logger.warning("aucon_prmse: Py-Feat init failed: %s", e)
            self._backend = "unavailable"

    def _detect(self, video_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Per-frame (active-AU boolean vector, pose angles) for the largest face."""
        if self._detector is None:
            return None
        frames = _frames(video_path,
                         float(self.config.get("fps", 5.0)),
                         int(self.config.get("max_frames", 300)))
        if not frames:
            return None
        aus, poses = [], []
        for frame in frames:
            try:
                res = self._detector.detect_image(frame)
            except Exception:
                continue
            if res is None or res.empty:
                continue
            i = 0  # largest/primary detected face
            au_row = res.aus.iloc[i].values.astype(np.float32)
            pose_row = res.poses.iloc[i].values.astype(np.float32)
            aus.append(au_row > float(self.config.get("au_threshold", _AU_THRESHOLD)))
            poses.append(pose_row)
        if not aus:
            return None
        return np.stack(aus), np.stack(poses)

    def process(self, sample: Sample) -> Sample:
        if self._backend != "pyfeat" or self._detector is None:
            return sample
        if not sample.is_video:
            return sample
        ref = sample.reference_path
        if ref is None or not Path(ref).is_file():
            return sample
        try:
            cand = self._detect(Path(sample.path))
            drv = self._detect(Path(ref))
            if cand is None or drv is None:
                return sample
            n = min(len(cand[0]), len(drv[0]))
            if n == 0:
                return sample
            aucon = float((cand[0][:n] == drv[0][:n]).all(axis=1).mean())
            prmse = float(np.sqrt(((cand[1][:n] - drv[1][:n]) ** 2).mean()))
            if not np.isfinite(aucon) or not np.isfinite(prmse):
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.aucon = aucon
            sample.quality_metrics.prmse = prmse
        except Exception as e:
            logger.warning("aucon_prmse: failed on %s: %s",
                           Path(sample.path).name, e)
        return sample

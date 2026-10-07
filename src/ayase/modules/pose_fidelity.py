"""AKD / MKR / MPJPE / PCK — 2D pose fidelity against the source video.

FOMM's pose-evaluation protocol (Siarohin et al., NeurIPS 2019/2020) reports
AKD — the mean distance between corresponding keypoints normalised by the
source bounding-box diagonal — and MKR — the fraction of keypoints missing in
the generated frame that are present in the source. Ginosar et al. (2019)
report MPJPE, the mean per-joint position error, and PCK, the share of joints
within ``pck_threshold`` of the source position (same normalisation).

Keypoints are the 33 BlazePose joints from MediaPipe; a joint counts as
missing when the pose is absent or its visibility is below
``visibility_threshold``.

akd / mkr / mpjpe / pck -- AKD, MKR, MPJPE lower = closer; PCK higher = closer.
Requires ``sample.reference_path`` pointing at the source video.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PoseFidelityModule(PipelineModule):
    name = "pose_fidelity"
    description = "AKD/MKR (FOMM) and MPJPE/PCK (Ginosar 2019) pose distance to source"
    provenance = "adapted"
    sources = {
        "akd": "AKD, FOMM pose-evaluation (Siarohin et al., NeurIPS 2019, arXiv:2003.00196) — https://github.com/AliaksandrSiarohin/pose-evaluation",
        "mkr": "MKR, FOMM pose-evaluation (Siarohin et al., NeurIPS 2019, arXiv:2003.00196) — https://github.com/AliaksandrSiarohin/pose-evaluation",
        "mpjpe": "MPJPE, Ginosar et al. (https://arxiv.org/abs/1906.04160)",
        "pck": "PCK, Ginosar et al. (https://arxiv.org/abs/1906.04160)",
    }
    deviations = {
        "akd": "keypoints come from MediaPipe BlazePose (33 joints, normalised coords) instead of the official learned 10-kp detector; normalisation by source bbox diagonal matches",
        "mkr": "a MediaPipe joint with visibility < threshold counts as missing; the source uses its detector's own detection failure",
        "mpjpe": "computed on MediaPipe 33-joint landmarks in image-normalised coords (proportional to pixels) rather than the paper's pose detector",
        "pck": "computed on MediaPipe 33-joint landmarks with threshold on bbox-normalised coords rather than the paper's pose detector",
    }
    requires_reference = True
    default_config = {
        "fps": 25,
        "max_frames": 600,
        "visibility_threshold": 0.5,
        "pck_threshold": 0.1,
    }
    metric_info = {
        "akd": "Average Keypoint Distance vs source, bbox-normalised (lower=better)",
        "mkr": "Missing Keypoint Rate vs source (0-1, lower=better)",
        "mpjpe": "Mean per-joint position error vs source, bbox-normalised (lower=better)",
        "pck": "Fraction of joints within threshold of source pose (0-1, higher=better)",
    }
    metric_groups = {"akd": "pose", "mkr": "pose", "mpjpe": "pose", "pck": "pose"}

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
            logger.warning("pose_fidelity: mediapipe not installed, disabled")
            self._backend = "unavailable"
        except Exception as e:
            logger.warning("pose_fidelity: Pose init failed: %s", e)
            self._backend = "unavailable"

    def on_dispose(self) -> None:
        if self._pose is not None:
            try:
                self._pose.close()
            except Exception:
                pass
            self._pose = None

    def _bbox_diag(self, xy: np.ndarray) -> float:
        lo = xy.min(axis=0)
        hi = xy.max(axis=0)
        return float(max(np.linalg.norm(hi - lo), 1e-6))

    def process(self, sample: Sample) -> Sample:
        if self._backend != "mediapipe":
            return sample
        if not sample.is_video:
            return sample
        ref = sample.reference_path
        if ref is None or not Path(ref).is_file():
            return sample
        try:
            from ._mp_seq import body_pose_seq

            fps = float(self.config.get("fps", 25))
            max_frames = int(self.config.get("max_frames", 600))
            cand = body_pose_seq(Path(sample.path), fps=fps,
                                 max_frames=max_frames, detector=self._pose)
            src = body_pose_seq(Path(ref), fps=fps,
                                max_frames=max_frames, detector=self._pose)
            if cand is None or src is None:
                return sample
            # align on common source-frame indices
            cand_map = {i: f for i, f in zip(cand[1], cand[0])}
            src_map = {i: f for i, f in zip(src[1], src[0])}
            common = sorted(set(cand_map) & set(src_map))
            if not common:
                return sample
            vis_thr = float(self.config.get("visibility_threshold", 0.5))
            pck_thr = float(self.config.get("pck_threshold", 0.1))

            dists, errs, missing, present_total, correct = [], [], [], [], []
            for fi in common:
                c, s = cand_map[fi], src_map[fi]
                diag = self._bbox_diag(s[:, :2])
                s_ok = s[:, 3] >= vis_thr
                c_ok = c[:, 3] >= vis_thr
                d_raw = np.linalg.norm(c[:, :2] - s[:, :2], axis=-1)
                d = d_raw / diag
                dists.append(d[s_ok].mean() if s_ok.any() else np.nan)
                errs.append(d_raw[s_ok].mean() if s_ok.any() else np.nan)
                missing.append(int((s_ok & ~c_ok).sum()))
                present_total.append(int(s_ok.sum()))
                correct.append(int((s_ok & c_ok & (d <= pck_thr)).sum()))
            present = sum(present_total)
            if present == 0:
                return sample
            akd = float(np.nanmean(dists))
            mkr = float(sum(missing) / present)
            mpjpe = float(np.nanmean(errs))  # unnormalised, image-normalised coords
            pck = float(sum(correct) / present)
            vals = [akd, mkr, mpjpe, pck]
            if not all(np.isfinite(v) for v in vals):
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            (sample.quality_metrics.akd, sample.quality_metrics.mkr,
             sample.quality_metrics.mpjpe, sample.quality_metrics.pck) = vals
        except Exception as e:
            logger.warning("pose_fidelity: failed on %s: %s",
                           Path(sample.path).name, e)
        return sample

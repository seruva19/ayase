"""RTMPose pose/gesture fidelity — keypoint-confidence plausibility.

Assesses how plausible the human pose(s) in generated content are, using
RTMPose (via the lightweight ``rtmlib`` wrapper — no full mmpose install).

Backend:
  1. person detection: ``rtmlib.YOLOX`` (ONNX);
  2. pose estimation: ``rtmlib.RTMPose`` (ONNX).

The ONNX weights are fetched from the ayase-models mirror on first use (via
``download_model_file``, cached under ``models_dir/rtmpose_fidelity/``) — the
same convention as the other weight-backed modules. ``rtmlib`` is an OPTIONAL
dependency: its import is guarded in ``setup()``; if the package is missing or
the weights cannot be fetched, the module stays registered but
``_backend = "unavailable"`` and never crashes discovery.

Defaults match the ``rtmlib`` "balanced" pair: YOLOX-m detector + RTMPose-m
body (256x192, COCO-17).

Scoring (no ground-truth pose available, so this is a plausibility proxy):
for every detected person we combine the fraction of confidently-localised
joints with their mean confidence — ``0.5 * valid_joint_ratio + 0.5 * mean_conf``
— and average over all person-frames. ``rtmpose_score`` is 0-1, higher = better.
No detector -> metric left ``None``. A detector that finds no people in any
frame also leaves the metric ``None`` ("no humans" != "good pose").
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Min per-joint confidence for a keypoint to count as confidently localised.
KEYPOINT_CONF = 0.30

# Weights are mirrored under <models_dir>/rtmpose_fidelity/ and fetched on first
# use, matching the other weight-backed modules.
_MODELS_BASE = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
_DET_REL = "rtmpose_fidelity/yolox_m.onnx"
_POSE_REL = "rtmpose_fidelity/rtmpose_m.onnx"


class RTMPoseFidelityModule(PipelineModule):
    name = "rtmpose_fidelity"
    description = "RTMPose keypoint-confidence pose/gesture plausibility (rtmlib, local ONNX; 0-1, higher=better)"
    default_config = {
        "subsample": 8,                    # frames sampled per video
        "models_dir": "models",            # weights land under models_dir/rtmpose_fidelity/
        "det_input_size": [640, 640],      # YOLOX input (w, h)
        "pose_input_size": [192, 256],     # RTMPose input (w, h) for the 256x192 body model
        "warn_threshold": 0.4,             # emit an INFO issue below this score
        "device": "auto",
    }
    metric_info = {
        "rtmpose_score": "RTMPose keypoint-confidence pose plausibility (0-1, higher=better)",
    }
    metric_groups = {
        "rtmpose_score": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "unavailable"
        self._ml_available = False
        self._device = "cpu"
        self._det_path: Optional[str] = None
        self._pose_path: Optional[str] = None

    def setup(self) -> None:
        if self.test_mode:
            return

        from ayase.runtime import resolve_torch_device

        self._device = resolve_torch_device(self.config.get("device", "auto"))

        # rtmlib is optional/heavy: guard the import so a missing package does
        # not break discovery — the module simply stays unavailable.
        try:
            from rtmlib import YOLOX, RTMPose  # noqa: F401
        except ImportError:
            logger.warning("RTMPoseFidelity: rtmlib not available (pip install rtmlib)")
            return
        except Exception as e:
            logger.debug("rtmlib import check failed: %s", e)
            return

        # Fetch the ONNX weights from the ayase-models mirror on first use
        # (cached afterwards), same as the other weight-backed modules.
        from ayase.config import download_model_file

        models_dir = str(self.config.get("models_dir", "models"))
        try:
            self._det_path = str(download_model_file(_DET_REL, _MODELS_BASE + _DET_REL, models_dir))
            self._pose_path = str(download_model_file(_POSE_REL, _MODELS_BASE + _POSE_REL, models_dir))
        except Exception as e:
            logger.warning("RTMPoseFidelity: could not fetch ONNX weights (%s); staying unavailable", e)
            return

        self._backend = "rtmlib"
        self._ml_available = True
        logger.info("RTMPoseFidelity using rtmlib (RTMPose) with mirrored ONNX weights")

    # ------------------------------------------------------------------ #
    # Shared detector + pose estimator (loaded once per pipeline)         #
    # ------------------------------------------------------------------ #

    def _get_detpose(self):
        from ayase.runtime import shared_runtime_resource

        if self._backend != "rtmlib":
            return None

        # rtmlib runs on onnxruntime; map torch device string to its convention.
        rt_device = "cuda" if "cuda" in str(self._device) else "cpu"
        det_size = tuple(self.config.get("det_input_size", [640, 640]))
        pose_size = tuple(self.config.get("pose_input_size", [192, 256]))

        def build_detpose():
            from rtmlib import YOLOX, RTMPose
            det = YOLOX(
                onnx_model=self._det_path,
                model_input_size=det_size,
                backend="onnxruntime",
                device=rt_device,
            )
            pose = RTMPose(
                onnx_model=self._pose_path,
                model_input_size=pose_size,
                backend="onnxruntime",
                device=rt_device,
            )
            return (det, pose)

        return shared_runtime_resource(
            self, ("rtmpose_detpose", self._det_path, self._pose_path, rt_device), build_detpose
        )

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        from ayase.image import sample_frames

        max_frames = self.config.get("subsample", 8) if sample.is_video else 1
        return list(sample_frames(sample.path, max_frames=max_frames, color="bgr"))

    def _score_frame(self, detpose, frame_bgr: np.ndarray) -> List[float]:
        """Return per-person plausibility scores for one frame."""
        det, pose = detpose
        # sample_frames yields read-only views; rtmlib/onnxruntime need a
        # writable, contiguous array.
        img = np.ascontiguousarray(frame_bgr)
        bboxes = det(img)
        if bboxes is None or len(bboxes) == 0:
            return []
        keypoints, scores = pose(img, bboxes)
        if scores is None or len(scores) == 0:
            return []

        person_scores: List[float] = []
        for person_scores_arr in np.asarray(scores, dtype=float):
            if person_scores_arr.size == 0:
                continue
            valid_ratio = float(np.mean(person_scores_arr >= KEYPOINT_CONF))
            mean_conf = float(np.mean(person_scores_arr))
            person_scores.append(0.5 * valid_ratio + 0.5 * mean_conf)
        return person_scores

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        frames = self._load_frames(sample)
        if not frames:
            return sample

        detpose = self._get_detpose()
        if detpose is None:
            return sample

        all_person_scores: List[float] = []
        for frame in frames:
            try:
                all_person_scores.extend(self._score_frame(detpose, frame))
            except Exception as e:  # detector/format surprises must not crash the run
                logger.debug("rtmpose_fidelity: pose detection failed on a frame: %s", e)

        if not all_person_scores:
            logger.debug(
                "rtmpose_fidelity: no persons detected in %s; leaving rtmpose_score unset",
                sample.path,
            )
            return sample

        score = float(np.clip(np.mean(all_person_scores), 0.0, 1.0))

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.rtmpose_score = score

        if score < self.config.get("warn_threshold", 0.4):
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Low RTMPose keypoint confidence: {score:.2f} (implausible/occluded pose)",
                    details={"rtmpose_score": score, "backend": self._backend},
                    recommendation=(
                        "Detected human pose has few confidently-localised joints; "
                        "the generated pose may be distorted or heavily occluded."
                    ),
                )
            )

        return sample

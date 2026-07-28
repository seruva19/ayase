"""VMBench Object Integrity Score (OIS) — human anatomy temporal integrity.

Faithful port of VMBench's Object Integrity Score (AMAP-ML, ICCV 2025,
arXiv:2503.10076). OIS penalises implausible changes in a person's body over
time — bones that stretch/shrink and joints that bend impossibly frame-to-frame
(the tell-tale signature of extra/warped limbs in generated video).

The upstream metric runs an mmdet person detector + mmpose RTMPose, then scores
COCO-17 keypoint tracks with two pure-NumPy checks (bone-length consistency and
joint-angle consistency), combined 50/50. This module keeps that scoring math
verbatim (``ayase.vendor.vmbench.pose_utils``) but sources the keypoints from
the toolkit's own rtmlib RTMPose backend (same RTMPose-m body model), so no
mmpose/mmcv/CUDA build is needed. Weights come from the model mirror on first
use (shared with ``rtmpose_fidelity``).

Scoring: for the top-detected person, per consecutive frame collect COCO-17
keypoints + confidences, then
    OIS = 0.5 * bone_length_score + 0.5 * joint_angle_score        (0-1, higher=better)
Both sub-scores are the fraction of body parts / joints whose size / angle stays
within the upstream anomaly thresholds across the clip. No person detected in
enough frames -> metric left ``None`` ("no human" != "intact anatomy").
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Weights are the same rtmlib ONNX pair as rtmpose_fidelity, mirrored and fetched
# on first use (shared cache directory).
_MODELS_BASE = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
_DET_REL = "rtmpose_fidelity/yolox_m.onnx"
_POSE_REL = "rtmpose_fidelity/rtmpose_m.onnx"


class ObjectIntegrityModule(PipelineModule):
    name = "object_integrity"
    description = "VMBench Object Integrity Score — human bone-length/joint-angle temporal integrity (0-1, higher=better)"
    default_config = {
        "max_frames": 120,                 # consecutive frames read from the start
        "models_dir": "models",            # weights land under models_dir/rtmpose_fidelity/
        "det_input_size": [640, 640],
        "pose_input_size": [192, 256],
        "warn_threshold": 0.6,
        "device": "auto",
    }
    metric_info = {
        "object_integrity_score": "VMBench OIS: human bone-length/joint-angle temporal integrity (0-1, higher=better)",
    }
    metric_groups = {
        "object_integrity_score": "motion",
    }
    models = [
        {"id": "yolox_m.onnx", "type": "local",
         "url": "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rtmpose_fidelity/yolox_m.onnx",
         "task": "YOLOX person detector (rtmlib backend)", "notes": "Shared with rtmpose_fidelity"},
        {"id": "rtmpose_m.onnx", "type": "local",
         "url": "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rtmpose_fidelity/rtmpose_m.onnx",
         "task": "RTMPose keypoint estimator (rtmlib backend)", "notes": "Shared with rtmpose_fidelity"},
    ]

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

        try:
            from rtmlib import YOLOX, RTMPose  # noqa: F401
        except ImportError:
            logger.warning("ObjectIntegrity: rtmlib not available (pip install rtmlib)")
            return
        except Exception as e:
            logger.debug("rtmlib import check failed: %s", e)
            return

        from ayase.config import download_model_file

        models_dir = str(self.config.get("models_dir", "models"))
        try:
            self._det_path = str(download_model_file(_DET_REL, _MODELS_BASE + _DET_REL, models_dir))
            self._pose_path = str(download_model_file(_POSE_REL, _MODELS_BASE + _POSE_REL, models_dir))
        except Exception as e:
            logger.warning("ObjectIntegrity: could not fetch ONNX weights (%s); staying unavailable", e)
            return

        self._backend = "rtmlib"
        self._ml_available = True
        logger.info("ObjectIntegrity using rtmlib (RTMPose) with mirrored ONNX weights")

    def _get_detpose(self):
        from ayase.runtime import shared_runtime_resource

        if self._backend != "rtmlib":
            return None

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

        # Shared with rtmpose_fidelity (same weights + device) to avoid a second load.
        return shared_runtime_resource(
            self, ("rtmpose_detpose", self._det_path, self._pose_path, rt_device), build_detpose
        )

    def _read_consecutive_frames(self, path: str, max_frames: int) -> List[np.ndarray]:
        """Read up to ``max_frames`` consecutive BGR frames (VMBench reads the clip
        frame-by-frame; the OIS thresholds are calibrated on adjacent frames)."""
        frames: List[np.ndarray] = []
        cap = cv2.VideoCapture(path)
        try:
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        return frames

    def _instance_info(self, detpose, frames: List[np.ndarray]) -> list:
        """Build VMBench's instance_info: top-person COCO-17 keypoints+scores per frame."""
        det, pose = detpose
        info = []
        for frame in frames:
            img = np.ascontiguousarray(frame)
            try:
                bboxes = det(img)
            except Exception as e:
                logger.debug("object_integrity: detection failed on a frame: %s", e)
                continue
            if bboxes is None or len(bboxes) == 0:
                continue
            # Top detection (rtmlib returns boxes highest-score first); score just it.
            top_box = np.asarray(bboxes)[:1]
            try:
                keypoints, scores = pose(img, top_box)
            except Exception as e:
                logger.debug("object_integrity: pose failed on a frame: %s", e)
                continue
            if scores is None or len(scores) == 0:
                continue
            info.append(
                {"instances": [{
                    "keypoints": np.asarray(keypoints[0], dtype=float),
                    "keypoint_scores": np.asarray(scores[0], dtype=float),
                }]}
            )
        return info

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        detpose = self._get_detpose()
        if detpose is None:
            return sample

        frames = self._read_consecutive_frames(str(sample.path), self.config.get("max_frames", 120))
        if len(frames) < 2:
            return sample

        instance_info = self._instance_info(detpose, frames)
        if len(instance_info) < 2:
            logger.debug(
                "object_integrity: fewer than 2 frames with a detected person in %s; leaving unset",
                sample.path,
            )
            return sample

        from ayase.vendor.vmbench.pose_utils import analyze_lengths_over_time, analyze_joint_angles

        try:
            _, length_score = analyze_lengths_over_time(instance_info)
            _, angle_score = analyze_joint_angles(instance_info)
        except Exception as e:
            logger.warning("object_integrity: scoring failed for %s: %s", sample.path, e)
            return sample

        score = float(length_score) * 0.5 + float(angle_score) * 0.5

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.object_integrity_score = score

        if score < self.config.get("warn_threshold", 0.6):
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Low object integrity: {score:.2f} (implausible limb-length/joint-angle changes over time)",
                    details={"object_integrity_score": score, "backend": self._backend},
                    recommendation=(
                        "The person's bones/joints change implausibly between frames — "
                        "a sign of warped, extra, or disappearing limbs in the generated motion."
                    ),
                )
            )

        return sample

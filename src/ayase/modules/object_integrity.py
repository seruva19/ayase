"""VMBench Object Integrity Score (OIS) — human anatomy temporal integrity.

Faithful port of VMBench's Object Integrity Score (AMAP-ML, ICCV 2025,
arXiv:2503.10076, ``object_integrity_score.py``). OIS penalises implausible
changes in a person's body over time — bones that stretch/shrink and joints
that bend impossibly frame-to-frame (the tell-tale signature of extra/warped
limbs in generated video).

Upstream pipeline (reproduced verbatim for ``backend="mmpose"``):

  * person detector: mmdet **RTMDet-m 640×640** trained on person
    (``rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth``, config
    ``demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py`` from the mmpose
    package), ``det_cat_id=0``, ``bbox_thr=0.3``, NMS ``nms_thr=0.3``;
  * pose estimator: mmpose **RTMPose-m body8 256×192**
    (``rtmpose-m_8xb256-420e_body8-256x192.py``,
    ``rtmpose-m_simcc-body7_pt-body7_420e-256x192``) via ``inference_topdown``;
  * **every frame** of the video is processed frame-by-frame;
  * the COCO-17 keypoint tracks feed the pure-NumPy bone-length and
    joint-angle consistency checks, combined 50/50.

Both sub-scores are the fraction of body parts / joints whose size / angle
stays within the upstream anomaly thresholds across the clip; the scoring math
itself is the vendored upstream code (``ayase.vendor.vmbench.pose_utils``),
which scores ``instances[0]`` of every frame. No person detected in enough
frames -> metric left ``None`` ("no human" != "intact anatomy").

``backend="rtmlib"`` is an opt-in ONNX alternative (YOLOX-m + RTMPose-m via
onnxruntime — same model family, different runtime and detector weights);
``backend="auto"`` prefers the canonical mmpose/mmdet stack and falls back to
rtmlib when the OpenMMLab packages are not installed. If neither is available
the metric stays unset.
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# rtmlib fallback weights (mirror), shared with rtmpose_fidelity.
_MODELS_BASE = "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
_DET_REL = "rtmpose_fidelity/yolox_m.onnx"
_POSE_REL = "rtmpose_fidelity/rtmpose_m.onnx"

# Upstream model defaults (object_integrity_score.py).
_DET_CFG_REL = "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py"
_POSE_CFG_REL = "configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py"
_DET_CKPT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/"
    "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)
_POSE_CKPT = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth"
)


def _mmpose_config_path(rel_path: str) -> Optional[str]:
    """Resolve an mmpose repo-relative config inside the installed package.

    pip ``mmpose`` ships the repo tree under ``mmpose/.mim/``.
    """
    import os

    try:
        import mmpose
    except ImportError:
        return None
    base = os.path.dirname(mmpose.__file__)
    for root in (os.path.join(base, ".mim"), base):
        candidate = os.path.join(root, rel_path.replace("/", os.sep))
        if os.path.isfile(candidate):
            return candidate
    return None


class ObjectIntegrityModule(PipelineModule):
    name = "object_integrity"
    provenance = "adapted"
    sources = {
        "object_integrity_score": "VMBench OIS (Ling et al., ICCV 2025, arXiv 2503.10076) — https://github.com/AMAP-ML/VMBench",
    }
    deviations = {
        "object_integrity_score": "backend='mmpose' replicates upstream (RTMDet-m person + RTMPose-m body8, all frames); backend='rtmlib' uses ONNX YOLOX-m/RTMPose-m instead of mmdet/mmpose (documented runtime substitution)",
    }
    description = "VMBench Object Integrity Score — human bone-length/joint-angle temporal integrity (0-1, higher=better)"
    default_config = {
        "backend": "auto",  # "auto" | "mmpose" | "rtmlib"
        "max_frames": 0,  # 0 = all frames (upstream reads the whole video)
        "models_dir": "models",  # rtmlib weights land under models_dir/rtmpose_fidelity/
        "det_input_size": [640, 640],
        "pose_input_size": [192, 256],
        "det_config": None,  # default: upstream RTMDet-m person config from mmpose
        "det_checkpoint": _DET_CKPT,
        "pose_config": None,  # default: upstream RTMPose-m body8 config from mmpose
        "pose_checkpoint": _POSE_CKPT,
        "det_cat_id": 0,
        "bbox_thr": 0.3,
        "nms_thr": 0.3,
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
        {
            "id": "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
            "type": "other",
            "url": _DET_CKPT,
            "task": "RTMDet-m person detector (upstream mmdet backend)",
        },
        {
            "id": "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth",
            "type": "other",
            "url": _POSE_CKPT,
            "task": "RTMPose-m body8 keypoint estimator (upstream mmpose backend)",
        },
        {
            "id": "mmpose+mmdet",
            "type": "pip_package",
            "install": "pip install mmpose mmdet mmcv",
            "task": "OpenMMLab pose pipeline (canonical upstream backend)",
        },
        {"id": "yolox_m.onnx", "type": "local",
         "url": "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rtmpose_fidelity/yolox_m.onnx",
         "task": "YOLOX person detector (rtmlib opt-in backend)", "notes": "Shared with rtmpose_fidelity"},
        {"id": "rtmpose_m.onnx", "type": "local",
         "url": "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/rtmpose_fidelity/rtmpose_m.onnx",
         "task": "RTMPose keypoint estimator (rtmlib opt-in backend)", "notes": "Shared with rtmpose_fidelity"},
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
        backend = str(self.config.get("backend", "auto")).lower()

        # Canonical tier: upstream mmdet detector + mmpose RTMPose.
        if backend in ("auto", "mmpose") and self._mmpose_ready():
            self._backend = "mmpose"
            self._ml_available = True
            logger.info("ObjectIntegrity using upstream mmdet+mmpose pipeline")
            return
        if backend == "mmpose":
            logger.warning("ObjectIntegrity: mmpose/mmdet requested but unavailable")
            return

        # Opt-in ONNX fallback: rtmlib YOLOX + RTMPose (same model family).
        if backend in ("auto", "rtmlib") and self._rtmlib_ready():
            self._backend = "rtmlib"
            self._ml_available = True
            logger.info("ObjectIntegrity using rtmlib (RTMPose) ONNX fallback")
            return
        if backend == "rtmlib":
            logger.warning("ObjectIntegrity: rtmlib requested but unavailable")

    @staticmethod
    def _mmpose_ready() -> bool:
        try:
            import mmdet  # noqa: F401
            import mmpose  # noqa: F401
        except Exception:
            return False
        return (
            _mmpose_config_path(_DET_CFG_REL) is not None
            and _mmpose_config_path(_POSE_CFG_REL) is not None
        )

    def _rtmlib_ready(self) -> bool:
        try:
            from rtmlib import YOLOX, RTMPose  # noqa: F401
        except Exception as e:
            logger.debug("rtmlib import check failed: %s", e)
            return False

        from ayase.config import download_model_file

        models_dir = str(self.config.get("models_dir", "models"))
        try:
            self._det_path = str(download_model_file(_DET_REL, _MODELS_BASE + _DET_REL, models_dir))
            self._pose_path = str(download_model_file(_POSE_REL, _MODELS_BASE + _POSE_REL, models_dir))
        except Exception as e:
            logger.warning("ObjectIntegrity: could not fetch ONNX weights (%s); staying unavailable", e)
            return False
        return True

    def _get_detpose(self):
        from ayase.runtime import shared_runtime_resource

        rt_device = "cuda" if "cuda" in str(self._device) else "cpu"
        det_size = tuple(self.config.get("det_input_size", [640, 640]))
        pose_size = tuple(self.config.get("pose_input_size", [192, 256]))

        if self._backend == "mmpose":
            det_cfg = self.config.get("det_config") or _mmpose_config_path(_DET_CFG_REL)
            pose_cfg = self.config.get("pose_config") or _mmpose_config_path(_POSE_CFG_REL)
            det_ckpt = self.config.get("det_checkpoint", _DET_CKPT)
            pose_ckpt = self.config.get("pose_checkpoint", _POSE_CKPT)
            device = str(self._device)

            def build_mmpose():
                from mmdet.apis import init_detector
                from mmpose.apis import init_model as init_pose_estimator
                from mmpose.utils import adapt_mmdet_pipeline

                detector = init_detector(det_cfg, det_ckpt, device=device)
                detector.cfg = adapt_mmdet_pipeline(detector.cfg)
                pose_estimator = init_pose_estimator(pose_cfg, pose_ckpt, device=device)
                return ("mmpose", detector, pose_estimator)

            return shared_runtime_resource(
                self, ("vmbench_mmpose", det_cfg, pose_cfg, device), build_mmpose
            )

        if self._backend != "rtmlib":
            return None

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
            return ("rtmlib", det, pose)

        # Shared with rtmpose_fidelity (same weights + device) to avoid a second load.
        return shared_runtime_resource(
            self, ("rtmpose_detpose", self._det_path, self._pose_path, rt_device), build_detpose
        )

    def _read_consecutive_frames(self, path: str, max_frames: int) -> List[np.ndarray]:
        """Read BGR frames sequentially (upstream decodes the whole video; the
        OIS thresholds are calibrated on adjacent frames)."""
        frames: List[np.ndarray] = []
        cap = cv2.VideoCapture(path)
        try:
            while max_frames <= 0 or len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        return frames

    def _instance_info(self, detpose, frames: List[np.ndarray]) -> list:
        """Build VMBench's instance_info: COCO-17 keypoints+scores per frame."""
        kind, det, pose = detpose
        info = []
        for frame in frames:
            img = np.ascontiguousarray(frame)
            try:
                if kind == "mmpose":
                    instance = self._mmpose_frame_instance(det, pose, img)
                else:
                    instance = self._rtmlib_frame_instance(det, pose, img)
            except Exception as e:
                logger.debug("object_integrity: %s failed on a frame: %s", kind, e)
                continue
            if instance is not None:
                info.append({"instances": [instance]})
        return info

    def _mmpose_frame_instance(self, detector, pose_estimator, img):
        """Upstream ``process_one_image``: det -> threshold -> NMS -> topdown pose."""
        from mmdet.apis import inference_detector
        from mmpose.apis import inference_topdown
        from mmpose.evaluation.functional import nms
        from mmpose.structures import merge_data_samples

        det_result = inference_detector(detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
        )
        bboxes = bboxes[
            np.logical_and(
                pred_instance.labels == self.config.get("det_cat_id", 0),
                pred_instance.scores > self.config.get("bbox_thr", 0.3),
            )
        ]
        bboxes = bboxes[nms(bboxes, self.config.get("nms_thr", 0.3)), :4]
        if len(bboxes) == 0:
            return None
        pose_results = inference_topdown(pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)
        pred_instances = data_samples.get("pred_instances", None)
        if pred_instances is None or len(pred_instances) == 0:
            return None
        inst = pred_instances[0]  # instances[0] — what upstream scoring reads
        return {
            "keypoints": np.asarray(inst.keypoints.cpu().numpy() if hasattr(inst.keypoints, "cpu") else inst.keypoints, dtype=float),
            "keypoint_scores": np.asarray(inst.keypoint_scores.cpu().numpy() if hasattr(inst.keypoint_scores, "cpu") else inst.keypoint_scores, dtype=float),
        }

    @staticmethod
    def _rtmlib_frame_instance(det, pose, img):
        bboxes = det(img)
        if bboxes is None or len(bboxes) == 0:
            return None
        # Top detection (rtmlib returns boxes highest-score first); score just it.
        top_box = np.asarray(bboxes)[:1]
        keypoints, scores = pose(img, top_box)
        if scores is None or len(scores) == 0:
            return None
        return {
            "keypoints": np.asarray(keypoints[0], dtype=float),
            "keypoint_scores": np.asarray(scores[0], dtype=float),
        }

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        detpose = self._get_detpose()
        if detpose is None:
            return sample

        frames = self._read_consecutive_frames(str(sample.path), self.config.get("max_frames", 0))
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

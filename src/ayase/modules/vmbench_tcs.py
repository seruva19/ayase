"""VMBench Temporal Coherence Score (TCS) — implausible object vanish/emerge.

Faithful port of VMBench's Temporal Coherence Score (AMAP-ML, ICCV 2025,
arXiv:2503.10076). Subjects are grounded on keyframes, segmented, and their masks
are propagated across the clip; objects are given consistent ids by IoU matching.
An object that disappears (or appears) mid-clip is counted as an *error* only when
it is not explained by a benign cause — leaving the frame edge, shrinking below a
size floor, or a tracker detection error (judged from tracked-point visibility).
The score is the fraction of objects free of such errors:

    vanish_score = (objects_count - disappear_objects_count) / objects_count
    emerge_score = (objects_count - appear_objects_count) / objects_count
    temporal_coherence_score = (vanish_score + emerge_score) / 2   (0-1, higher=better)

Backends (all in-tree, weights from HF, pure ``pip install ayase``):
  * subject grounding — vendored GroundingDINO SwinB (``ayase.vendor.groundingdino``).
  * per-frame masks + video propagation — vendored SAM 2.1 Hiera-L
    (``ayase.vendor.sam2``), in-memory frames, weight from GD-ML/VMBench.
  * point tracking + visibility — vendored CoTracker3 offline (``ayase.vendor.cotracker``,
    weight ``scaled_offline.pth`` from GD-ML/VMBench on HF), the tracker VMBench uses:
    it lays a grid on the object mask and reports per-point tracks and visibility,
    which the verbatim vanish/emerge classifiers (``temporal_coherence_utils``) consume.

The subject noun defaults to ``"person"`` (human-centric) and can be overridden per
sample via ``sample.metadata["subject_noun"]`` or the module config.
"""

import copy
import logging
from typing import List, Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VMBenchTemporalCoherenceModule(PipelineModule):
    name = "vmbench_tcs"
    description = "VMBench Temporal Coherence — implausible object vanish/emerge over tracked masks (0-1, higher=better)"
    default_config = {
        "device": "auto",
        "max_frames": 48,          # frames read (VMBench clips ~49f)
        "keyframe_step": None,     # grounding interval; None -> derived from fps (fps-1)
        "grid_size": 30,           # VMBench grid density per axis
        "box_threshold": 0.35,     # VMBench TCS grounding thresholds
        "text_threshold": 0.35,
        "iou_threshold": 0.75,     # cross-keyframe object-id IoU match
        "subject_noun": None,      # override; else sample.metadata['subject_noun'] or 'person'
        "long_side": 640,          # cap resolution (aspect preserved)
        "query_chunk_size": 64,    # TAPIR query points per forward
        "models_dir": "models",
    }
    metric_info = {
        "temporal_coherence_score": "VMBench Temporal Coherence (implausible object vanish/emerge; 0-1, higher=better)",
    }
    metric_groups = {
        "temporal_coherence_score": "motion",
    }
    models = [
        {"id": "GD-ML/VMBench", "type": "huggingface",
         "url": "https://huggingface.co/GD-ML/VMBench",
         "task": "GroundingDINO SwinB + SAM2 Hiera-L + CoTracker3 offline weights",
         "notes": "groundingdino_swinb_cogcoor.pth, sam2.1_hiera_large.pt, scaled_offline.pth via ayase.vendor.groundingdino/sam2/cotracker"},
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self._device = None
        self._grounder = None
        self._sam2 = None
        self._cotracker = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            self._device = torch.device(resolve_torch_device(self.config.get("device", "auto")))
            device_str = "cuda" if self._device.type == "cuda" else "cpu"

            from ayase.vendor.groundingdino import load_grounding_dino
            from ayase.vendor.sam2 import load_sam2

            models_dir = self.config.get("models_dir", "models")
            self._grounder = load_grounding_dino(models_dir=models_dir, device=device_str)
            self._sam2 = load_sam2(models_dir=models_dir, device=device_str)

            # In-tree vendored CoTracker3 (the tracker VMBench TCS uses; weight
            # scaled_offline.pth from GD-ML/VMBench on HF).
            from ayase.vendor.cotracker import load_cotracker

            self._cotracker = load_cotracker(models_dir=models_dir, device=device_str)

            self._ml_available = True
            logger.info("VMBench TCS initialised (GroundingDINO + SAM2 + CoTracker3)")
        except ImportError as e:
            logger.warning("VMBench TCS unavailable (missing backend): %s", e)
        except Exception as e:
            logger.warning("Failed to setup VMBench TCS: %s", e)

    # --------------------------------------------------------------- helpers
    def _resolve_subject_noun(self, sample: Sample) -> str:
        noun = self.config.get("subject_noun")
        if noun:
            return str(noun)
        meta_noun = (sample.metadata or {}).get("subject_noun")
        if meta_noun:
            return str(meta_noun)
        return "person"

    def _load_frames(self, sample: Sample):
        """Sampled RGB frames, long side capped (aspect preserved). Returns
        (frames uint8 [T,H,W,3], width, height, fps)."""
        import cv2

        raw = sample_frames(sample.path, max_frames=int(self.config.get("max_frames", 48)),
                            color="rgb")
        if len(raw) < 4:
            return None, 0, 0, 0.0
        h0, w0 = raw[0].shape[:2]
        long_side = int(self.config.get("long_side", 640))
        scale = min(1.0, long_side / float(max(h0, w0)))
        if scale < 1.0:
            w, h = int(round(w0 * scale)), int(round(h0 * scale))
            frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in raw]
        else:
            w, h = w0, h0
            frames = list(raw)
        fps = 0.0
        try:
            cap = cv2.VideoCapture(str(sample.path))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
        except Exception:
            fps = 0.0
        return np.stack([np.ascontiguousarray(f) for f in frames]).astype(np.uint8), w, h, fps

    def _track_masked_grid(self, frames_u8, segm_mask, grid_query_frame, width, height):
        """Grid tracking on ``segm_mask`` with CoTracker3 (VMBench's tracker):
        the ``grid_size`` grid points inside the mask are tracked across the clip
        from ``grid_query_frame`` (backward + forward). Returns
        (pred_tracks [1, T, N, 2] pixel xy, pred_visibility [1, T, N] bool) or
        (None, None) if the mask holds no grid point.
        """
        mask = np.asarray(segm_mask)
        if mask.ndim != 2:
            mask = mask.reshape(mask.shape[-2], mask.shape[-1])
        mask = mask.astype(bool)
        if not mask.any():
            return None, None

        n_frames = frames_u8.shape[0]
        gqf = int(max(0, min(grid_query_frame, n_frames - 1)))
        pred_tracks, pred_visibility = self._cotracker.track(
            frames_u8, grid_size=int(self.config.get("grid_size", 30)),
            grid_query_frame=gqf, segm_mask=mask, backward_tracking=True)
        if pred_tracks is None or pred_tracks.shape[2] < 1:
            return None, None
        # Classifiers run on CPU tensors ([1, T, N, 2] and [1, T, N] bool).
        return pred_tracks.float().cpu(), pred_visibility.bool().cpu()

    def _build_tracking(self, frames_u8, width, height, subject_noun, step):
        """Reproduce VMBench's keyframe grounding + SAM2 propagation, returning
        (video_object_data, objects_count)."""
        import cv2
        import torch

        from ayase.vendor.vmbench.mask_dictionary_model import (
            MaskDictionaryModel, ObjectInfo,
        )

        n_frames = frames_u8.shape[0]
        state = self._sam2.init_state(frames_u8)
        sam2_masks = MaskDictionaryModel()
        objects_count = 0
        video_object_data: List[dict] = []

        for start_frame_idx in range(0, n_frames, step):
            frame_rgb = frames_u8[start_frame_idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            mask_dict = MaskDictionaryModel(promote_type="mask")

            grounding = self._grounder.ground(
                frame_bgr, subject_noun,
                box_threshold=float(self.config.get("box_threshold", 0.35)),
                text_threshold=float(self.config.get("text_threshold", 0.35)))
            boxes = grounding.boxes

            if boxes is not None and len(boxes) > 0:
                masks = [self._sam2.image_mask(frame_rgb, b) for b in boxes]
                mask_list = torch.from_numpy(np.stack(masks)).to(torch.bool)
                box_list = boxes.detach().cpu() if hasattr(boxes, "detach") else torch.as_tensor(boxes)
                labels = [subject_noun] * len(masks)
                mask_dict.add_new_frame_annotation(
                    mask_list=mask_list, box_list=box_list, label_list=labels)
                objects_count = mask_dict.update_masks(
                    tracking_annotation_dict=sam2_masks,
                    iou_threshold=float(self.config.get("iou_threshold", 0.75)),
                    objects_count=objects_count)
            else:
                mask_dict = sam2_masks

            if len(mask_dict.labels) == 0:
                video_object_data.extend([{} for _ in range(step + 1)])
                continue

            self._sam2.reset_state(state)
            for object_id, object_info in mask_dict.labels.items():
                self._sam2.add_new_mask(state, start_frame_idx, object_id,
                                        np.asarray(object_info.mask))

            for out_frame_idx, per_obj in self._sam2.propagate(
                    state, max_frame_num_to_track=step, start_frame_idx=start_frame_idx):
                frame_masks = MaskDictionaryModel()
                object_data = {}
                for out_obj_id, out_mask in per_obj.items():
                    info = ObjectInfo(instance_id=out_obj_id,
                                      mask=torch.from_numpy(np.asarray(out_mask)).to(torch.bool),
                                      class_name=mask_dict.get_target_class_name(out_obj_id)
                                      if out_obj_id in mask_dict.labels else subject_noun)
                    info.update_box()
                    frame_masks.labels[out_obj_id] = info
                    object_data[out_obj_id] = {"mask": np.asarray(out_mask).astype(bool)}
                sam2_masks = copy.deepcopy(frame_masks)
                video_object_data.append(object_data)

        return video_object_data, objects_count

    # --------------------------------------------------------------- process
    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        try:
            from ayase.vendor.vmbench.temporal_coherence_utils import (
                get_disappear_objects, get_appear_objects,
                is_edge_vanish, is_small_vanish, is_vanish_detect_error,
                is_edge_emerge, is_small_emerge, is_emerge_detect_error,
            )

            frames_u8, width, height, fps = self._load_frames(sample)
            if frames_u8 is None:
                return sample

            step = self.config.get("keyframe_step")
            if not step:
                step = int(fps) - 1 if fps and fps > 1 else 8
            step = max(1, int(step))
            subject_noun = self._resolve_subject_noun(sample)

            video_object_data, objects_count = self._build_tracking(
                frames_u8, width, height, subject_noun, step)
            if objects_count <= 0:
                return sample

            # VMBench subsamples the overlapping propagation segments to ~one
            # object-presence dict per keyframe interval.
            tracking_result = [item for i, item in enumerate(video_object_data, 1)
                               if i % (step + 1) != 0]
            tracking_result = tracking_result[::step]
            if len(tracking_result) < 2:
                return sample

            # -- vanish evaluation --
            disappear_objects = get_disappear_objects(tracking_result)
            if len(disappear_objects) == 0:
                vanish_score = 1.0
            else:
                disappear_objects_count = 0
                for obj_info in disappear_objects:
                    mask = obj_info['mask']
                    if mask is None or np.asarray(mask).ndim != 2:
                        continue
                    gqf = obj_info['first_appearance']
                    pred_tracks, pred_vis = self._track_masked_grid(frames_u8, mask, gqf, width, height)
                    if pred_tracks is None:
                        continue
                    edge = is_edge_vanish(pred_tracks, pred_vis, gqf, width, height)
                    small = is_small_vanish(pred_tracks, pred_vis, gqf, width, height)
                    derr = is_vanish_detect_error(pred_tracks, pred_vis, gqf)
                    if not edge and not small and not derr:
                        disappear_objects_count += 1
                vanish_score = (objects_count - disappear_objects_count) / objects_count

            # -- emerge evaluation --
            appear_objects = get_appear_objects(tracking_result)
            if len(appear_objects) == 0:
                emerge_score = 1.0
            else:
                appear_objects_count = 0
                for obj_info in appear_objects:
                    mask = obj_info['mask']
                    if mask is None or np.asarray(mask).ndim != 2:
                        continue
                    gqf = obj_info['first_appearance']
                    pred_tracks, pred_vis = self._track_masked_grid(frames_u8, mask, gqf, width, height)
                    if pred_tracks is None:
                        continue
                    edge = is_edge_emerge(pred_tracks, pred_vis, gqf, width, height)
                    small = is_small_emerge(pred_tracks, pred_vis, gqf, width, height)
                    derr = is_emerge_detect_error(pred_tracks, pred_vis, gqf)
                    if not edge and not small and not derr:
                        appear_objects_count += 1
                emerge_score = (objects_count - appear_objects_count) / objects_count

            tcs = float((vanish_score + emerge_score) / 2)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.temporal_coherence_score = tcs
            logger.debug("VMBench TCS for %s: %.4f (objects=%d, vanish=%.3f, emerge=%.3f)",
                         sample.path.name, tcs, objects_count, vanish_score, emerge_score)
        except Exception as e:
            logger.warning("VMBench TCS processing failed for %s: %s", sample.path, e)
        return sample

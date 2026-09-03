"""VMBench Perceptible Amplitude Score (PAS) — subject-vs-background motion.

Faithful port of VMBench's Perceptible Amplitude Score (AMAP-ML, ICCV 2025,
arXiv:2503.10076). PAS captures how much the *subject* actually moves once the
camera/background motion is discounted: the subject is grounded from a noun
phrase, segmented, and grid points inside/outside its mask are tracked over the
clip; each point's frame-to-frame displacement is summed over time, normalised
by the frame diagonal, and averaged over points to a motion degree; the score is
the subject's degree with the background (camera) degree subtracted off.

    PAS = subject_motion_degree - background_motion_degree     (0-1, subject motion)

Backends (all in-tree, weights from HF, pure ``pip install ayase``):
  * subject grounding — vendored GroundingDINO SwinB (``ayase.vendor.groundingdino``),
    weight ``groundingdino_swinb_cogcoor.pth`` from GD-ML/VMBench on HF.
  * subject mask — vendored SAM ViT-H (``ayase.vendor.sam``), weight
    ``sam_vit_h_4b8939.pth`` from GD-ML/VMBench on HF.
  * point tracking — vendored CoTracker3 offline (``ayase.vendor.cotracker``,
    weight ``scaled_offline.pth`` from GD-ML/VMBench on HF), the tracker VMBench
    uses: a grid is laid on the first frame and the points inside the mask are
    tracked; the diagonal-normalised aggregation (``calculate_motion_degree``,
    kept verbatim) consumes the per-point trajectories.

The subject noun defaults to ``"person"`` (human-centric) and can be overridden
per sample via ``sample.metadata["subject_noun"]`` or the module config.
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VMBenchPerceptibleAmplitudeModule(PipelineModule):
    name = "vmbench_pas"
    description = "VMBench Perceptible Amplitude — subject-vs-background tracked-point motion (0-1)"
    default_config = {
        "device": "auto",
        "max_frames": 60,          # frames tracked (VMBench clips ~49f)
        "grid_size": 30,           # VMBench grid density per axis
        "box_threshold": 0.3,      # GroundingDINO box filter (VMBench default)
        "text_threshold": 0.25,    # GroundingDINO text filter (VMBench default)
        "subject_noun": None,      # override; else sample.metadata['subject_noun'] or 'person'
        "long_side": 512,          # cap tracking resolution (aspect preserved)
        "query_chunk_size": 64,    # TAPIR query points per forward (VRAM cap)
        "models_dir": "models",
    }
    metric_info = {
        "perceptible_amplitude_score": "VMBench Perceptible Amplitude (subject-vs-background tracked motion; 0-1)",
    }
    metric_groups = {
        "perceptible_amplitude_score": "motion",
    }
    models = [
        {"id": "GD-ML/VMBench", "type": "huggingface",
         "url": "https://huggingface.co/GD-ML/VMBench",
         "task": "GroundingDINO SwinB + SAM ViT-H + CoTracker3 offline weights",
         "notes": "groundingdino_swinb_cogcoor.pth, sam_vit_h_4b8939.pth, scaled_offline.pth via ayase.vendor.groundingdino/sam/cotracker"},
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self._device = None
        self._grounder = None
        self._sam = None
        self._cotracker = None
        self._ml_available = False

    #: Vendored components whose licence differs from Ayase's own.
    vendor_components = ('cotracker',)

    def setup(self) -> None:
        from ayase.licenses import announce

        announce(self.vendor_components)
        if self.test_mode:
            return
        try:
            import torch
            from ayase.runtime import resolve_torch_device

            self._device = torch.device(resolve_torch_device(self.config.get("device", "auto")))
            device_str = "cuda" if self._device.type == "cuda" else "cpu"

            # In-tree vendored grounding + segmentation (weights from GD-ML/VMBench on HF).
            from ayase.vendor.groundingdino import load_grounding_dino
            from ayase.vendor.sam import load_sam

            models_dir = self.config.get("models_dir", "models")
            self._grounder = load_grounding_dino(models_dir=models_dir, device=device_str)
            self._sam = load_sam(models_dir=models_dir, device=device_str)

            # In-tree vendored CoTracker3 (the tracker VMBench PAS uses; weight
            # scaled_offline.pth from GD-ML/VMBench on HF).
            from ayase.vendor.cotracker import load_cotracker

            self._cotracker = load_cotracker(models_dir=models_dir, device=device_str)

            self._ml_available = True
            logger.info("VMBench PAS initialised (GroundingDINO + SAM + CoTracker3)")
        except ImportError as e:
            logger.warning("VMBench PAS unavailable (missing backend): %s", e)
        except Exception as e:
            logger.warning("Failed to setup VMBench PAS: %s", e)

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
        (frames uint8 [T,H,W,3], width, height)."""
        import cv2

        raw = sample_frames(sample.path, max_frames=int(self.config.get("max_frames", 60)),
                            color="rgb")
        if len(raw) < 4:
            return None, 0, 0
        h0, w0 = raw[0].shape[:2]
        long_side = int(self.config.get("long_side", 512))
        scale = min(1.0, long_side / float(max(h0, w0)))
        if scale < 1.0:
            w, h = int(round(w0 * scale)), int(round(h0 * scale))
            frames = [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in raw]
        else:
            w, h = w0, h0
            frames = list(raw)
        return np.stack([np.ascontiguousarray(f) for f in frames]).astype(np.uint8), w, h

    def _track_masked_grid(self, frames_u8, mask_bool, width, height):
        """Track a grid of points inside ``mask_bool`` with CoTracker3 (VMBench's
        tracker; ``segm_mask`` keeps only the grid points inside the mask, or the
        whole grid when ``mask_bool`` is None). Returns tracks as ``[1, T, N, 2]``
        (pixel xy) or None if the mask holds no point / the clip is too short."""
        if mask_bool is not None and not np.asarray(mask_bool).any():
            return None
        tracks, _vis = self._cotracker.track(
            frames_u8, grid_size=int(self.config.get("grid_size", 30)),
            grid_query_frame=0, segm_mask=mask_bool, backward_tracking=True)
        if tracks is None or tracks.shape[2] < 1 or tracks.shape[1] < 2:
            return None
        # [1, T, N, 2] pixel xy for the verbatim motion-degree formula (CPU to
        # match the CPU diagonal constant).
        return tracks.float().cpu()

    # --------------------------------------------------------------- process
    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._ml_available:
            return sample

        try:
            import cv2

            from ayase.vendor.vmbench.perceptible_amplitude_utils import (
                calculate_motion_degree, combine_subject_background,
            )

            frames_u8, width, height = self._load_frames(sample)
            if frames_u8 is None:
                return sample

            frame0_rgb = frames_u8[0]
            frame0_bgr = cv2.cvtColor(frame0_rgb, cv2.COLOR_RGB2BGR)
            subject_noun = self._resolve_subject_noun(sample)

            # Ground the subject on the first frame, then segment it.
            grounding = self._grounder.ground(
                frame0_bgr, subject_noun,
                box_threshold=float(self.config.get("box_threshold", 0.3)),
                text_threshold=float(self.config.get("text_threshold", 0.25)))
            boxes = grounding.boxes
            subject_detected = boxes is not None and len(boxes) > 0

            subject_mask = None
            if subject_detected:
                subject_mask = self._sam.subject_mask(frame0_rgb, boxes)
                if subject_mask is None or not bool(subject_mask.any()):
                    subject_detected = False
                    subject_mask = None

            background_mask = None
            if subject_mask is not None:
                background_mask = np.logical_not(subject_mask)

            # Background (camera) motion from points outside the subject.
            bg_kp = self._track_masked_grid(frames_u8, background_mask, width, height)
            if bg_kp is None:
                return sample
            background_degree = float(calculate_motion_degree(bg_kp, width, height)[0])

            subject_degree = float("nan")
            if subject_detected:
                subj_kp = self._track_masked_grid(frames_u8, subject_mask, width, height)
                if subj_kp is not None:
                    subject_degree = float(calculate_motion_degree(subj_kp, width, height)[0])

            pas = combine_subject_background(subject_degree, background_degree, subject_detected)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.perceptible_amplitude_score = float(pas)
            logger.debug(
                "VMBench PAS for %s: %.4f (subject=%s '%s', subj_deg=%.4f, bg_deg=%.4f)",
                sample.path.name, pas, subject_detected, subject_noun,
                subject_degree, background_degree)
        except Exception as e:
            logger.warning("VMBench PAS processing failed for %s: %s", sample.path, e)
        return sample

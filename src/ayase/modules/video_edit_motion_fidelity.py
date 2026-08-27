"""Source-to-edited-video motion fidelity using dense CoTracker trajectories.

Independently implements MTBench's trajectory protocol: positions are normalized
by frame size, trajectories are aligned in time, converted to initial position
plus per-step velocity, and matched by mean cosine similarity. Higher is better.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import resolve_torch_device
from ayase.image import sample_frames

logger = logging.getLogger(__name__)

COTRACKER_REPO = "GD-ML/VMBench"
COTRACKER_REVISION = "437cf3b7c667cd23e3f1e24e19e0af7868088907"


def trajectory_motion_similarity(generated, source) -> Optional[float]:
    """Compute MTBench's best-match cosine trajectory similarity."""
    import torch

    generated = torch.as_tensor(generated, dtype=torch.float32)
    source = torch.as_tensor(source, dtype=torch.float32)
    if generated.ndim != 3 or source.ndim != 3 or generated.shape[-1] != 2 or source.shape[-1] != 2:
        return None
    if generated.shape[0] == 0 or source.shape[0] == 0 or min(generated.shape[1], source.shape[1]) < 2:
        return None
    target_frames = min(generated.shape[1], source.shape[1])
    if generated.shape[1] != target_frames:
        generated = generated[:, torch.linspace(0, generated.shape[1] - 1, target_frames).long()]
    if source.shape[1] != target_frames:
        source = source[:, torch.linspace(0, source.shape[1] - 1, target_frames).long()]

    def motion_vectors(tracks):
        return torch.cat([tracks[:, :1], tracks[:, 1:] - tracks[:, :-1]], dim=1)

    generated_motion = torch.nn.functional.normalize(motion_vectors(generated), dim=-1, eps=1e-8)
    source_motion = torch.nn.functional.normalize(motion_vectors(source), dim=-1, eps=1e-8)
    pairwise = torch.einsum("ntc,mtc->nmt", generated_motion, source_motion).mean(dim=-1)
    score = float(pairwise.max(dim=1).values.mean().item())
    return score if np.isfinite(score) else None


class VideoEditMotionFidelityModule(PipelineModule):
    name = "video_edit_motion_fidelity"
    description = "MTBench dense-trajectory motion similarity between source and edited video"
    default_config = {"max_frames": 60, "long_side": 512, "grid_size": 30}
    models = [{"id": COTRACKER_REPO, "type": "huggingface", "revision": COTRACKER_REVISION,
               "task": "CoTracker3 offline dense point tracking"}]
    metric_info = {"video_edit_motion_fidelity": "Best-match dense trajectory-motion cosine similarity (higher=better)"}
    metric_groups = {"video_edit_motion_fidelity": "motion"}

    def __init__(self, config=None):
        super().__init__(config)
        self._tracker = None
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            from ayase.vendor.cotracker import load_cotracker
            device = resolve_torch_device(self.config.get("device", "auto"))
            self._tracker = load_cotracker(
                models_dir=self.config.get("models_dir", "models"), device=device
            )
            self._ml_available = True
            self._backend = "cotracker3_offline"
        except Exception as exc:
            logger.warning("VideoEditMotionFidelity setup failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        if sample.reference_path is None:
            return sample
        try:
            score = self._score_pair(sample.reference_path, sample.path)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.video_edit_motion_fidelity = score
        except Exception as exc:
            logger.warning("VideoEditMotionFidelity failed for %s: %s", sample.path, exc)
        return sample

    def _frames(self, path: Path) -> Optional[np.ndarray]:
        import cv2
        frames = sample_frames(path, max_frames=int(self.config.get("max_frames", 60)), color="rgb")
        if len(frames) < 2:
            return None
        height, width = frames[0].shape[:2]
        scale = min(1.0, float(self.config.get("long_side", 512)) / max(height, width))
        if scale < 1.0:
            size = (int(round(width * scale)), int(round(height * scale)))
            frames = [cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames]
        return np.stack([np.ascontiguousarray(frame) for frame in frames]).astype(np.uint8)

    def _tracks(self, frames: np.ndarray):
        tracks, _ = self._tracker.track(
            frames, grid_size=int(self.config.get("grid_size", 30)),
            grid_query_frame=0, backward_tracking=True,
        )
        tracks = tracks[0].permute(1, 0, 2).float().cpu()
        tracks[..., 0] /= frames.shape[2]
        tracks[..., 1] /= frames.shape[1]
        return tracks

    def _score_pair(self, source_path: Path, generated_path: Path) -> Optional[float]:
        source_frames = self._frames(source_path)
        generated_frames = self._frames(generated_path)
        if source_frames is None or generated_frames is None:
            return None
        return trajectory_motion_similarity(
            self._tracks(generated_frames), self._tracks(source_frames)
        )

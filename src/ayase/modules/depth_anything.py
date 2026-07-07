"""Depth Anything monocular depth estimation module.

From Data-Juicer's video_depth_estimation_mapper.
Uses Depth Anything V2 for monocular depth estimation and
temporal consistency scoring.
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class DepthAnythingModule(PipelineModule):
    name = "depth_anything"
    description = "Depth Anything V2 monocular depth estimation and consistency"
    default_config = {
        "model_name": "depth-anything/Depth-Anything-V2-Small-hf",
        "subsample": 8,
    }
    metric_groups = {
        "depth_anything_consistency": "spatial",
        "depth_anything_score": "spatial",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._pipe = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import torch
            from transformers import pipeline

            model_name = self.config.get(
                "model_name", "depth-anything/Depth-Anything-V2-Small-hf"
            )
            device = 0 if torch.cuda.is_available() else -1
            self._pipe = pipeline(
                "depth-estimation", model=model_name, device=device
            )
            self._device = device
            self._ml_available = True
            self._backend = "depth-anything-v2"
            logger.info("Depth Anything V2 loaded (device=%s)", device)
        except (ImportError, Exception) as e:
            logger.warning("Depth Anything unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample

        try:
            from PIL import Image

            subsample = self.config.get("subsample", 8)
            # Uniformly sampled RGB frames from the shared per-sample cache
            # (read-only views; PIL.fromarray only reads them).
            arrays = sample_frames(sample.path, max_frames=subsample, color="rgb")
            frames = [Image.fromarray(np.ascontiguousarray(a)) for a in arrays]

            if not frames:
                return sample

            depth_maps = []
            depth_scores = []
            for img in frames:
                result = self._pipe(img)
                depth_map = np.array(result["depth"])
                depth_maps.append(depth_map)
                # Quality: ratio of non-zero depth range (higher = more depth variation = richer scene)
                d_range = depth_map.max() - depth_map.min()
                depth_scores.append(float(d_range))

            sample.quality_metrics.depth_anything_score = float(np.mean(depth_scores))

            # Temporal consistency: correlation between consecutive depth maps
            if len(depth_maps) >= 2:
                consistencies = []
                for i in range(len(depth_maps) - 1):
                    d1 = depth_maps[i].flatten().astype(float)
                    d2 = depth_maps[i + 1].flatten().astype(float)
                    # Resize if needed
                    min_len = min(len(d1), len(d2))
                    d1 = d1[:min_len]
                    d2 = d2[:min_len]
                    if np.std(d1) > 0 and np.std(d2) > 0:
                        corr = np.corrcoef(d1, d2)[0, 1]
                        consistencies.append(max(0.0, corr))
                if consistencies:
                    sample.quality_metrics.depth_anything_consistency = float(
                        np.mean(consistencies)
                    )
        except Exception as e:
            logger.warning("Depth Anything processing failed: %s", e)
        return sample

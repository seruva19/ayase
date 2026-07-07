"""ptlflow optical flow motion module.

From Data-Juicer's video_motion_score_ptlflow_filter.
Uses ptlflow library with dpflow model for optical flow estimation.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PtlflowMotionModule(PipelineModule):
    name = "ptlflow_motion"
    description = "ptlflow optical flow motion scoring (dpflow model)"
    default_config = {
        "model_name": "dpflow",
        "ckpt_path": "things",
        "subsample": 8,
    }
    metric_groups = {
        "ptlflow_motion_score": "motion",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import ptlflow

            from ayase.runtime import resolve_torch_device

            model_name = self.config.get("model_name", "dpflow")
            ckpt_path = self.config.get("ckpt_path", "things")

            self._model = ptlflow.get_model(model_name, pretrained_ckpt=ckpt_path)
            device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = self._model.to(device).eval()
            self._device = device
            self._ml_available = True
            self._backend = "ptlflow"
            logger.info("ptlflow model (%s) loaded on %s", model_name, device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("ptlflow unavailable: ptlflow not installed")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("ptlflow unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        if not sample.is_video:
            return sample

        try:
            import torch
            from ptlflow.utils.io_adapter import IOAdapter

            from ayase.image import sample_frames

            subsample = self.config.get("subsample", 8)
            frames = sample_frames(sample.path, max_frames=subsample, color="bgr")

            if len(frames) < 2:
                return sample

            motion_scores = []
            adapter = IOAdapter()

            for i in range(len(frames) - 1):
                # sample_frames returns read-only views; copy so ptlflow's
                # tensor conversion / normalisation can write freely.
                inputs = adapter.prepare_inputs(
                    [np.ascontiguousarray(frames[i]), np.ascontiguousarray(frames[i + 1])]
                )
                inputs = {k: v.to(self._device) if hasattr(v, 'to') else v
                          for k, v in inputs.items()}

                with torch.no_grad():
                    predictions = self._model(inputs)

                flow = predictions["flows"].cpu().numpy()
                # ptlflow returns shape (B, 2, H, W) — channels are axis 1
                magnitude = np.sqrt(flow[0, 0] ** 2 + flow[0, 1] ** 2)
                motion_scores.append(float(np.mean(magnitude)))

            if motion_scores:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.ptlflow_motion_score = float(
                    np.mean(motion_scores)
                )
        except Exception as e:
            logger.warning("ptlflow motion scoring failed: %s", e)
        return sample

"""PaQ-2-PiQ (Patches to Pictures) quality module.

CVPR 2020. Trained on largest subjective database (40K images).
Maps perceptual space from patches to global quality predictions.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PaQ2PiQModule(PipelineModule):
    name = "paq2piq"
    description = "PaQ-2-PiQ patch-to-picture NR quality (CVPR 2020)"
    default_config = {"subsample": 4}
    metric_groups = {
        "paq2piq_score": "nr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            import torch

            from ayase.runtime import resolve_torch_device

            device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("paq2piq", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device(device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("PaQ-2-PiQ model loaded on %s", device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("PaQ-2-PiQ unavailable: pyiqa not installed")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("PaQ-2-PiQ unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        try:
            import cv2
            import torch

            frames = self._load_frames(sample)
            if not frames:
                return sample

            scores = []
            for frame in frames:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = (
                    torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                )
                tensor = tensor.to(self._device)
                with torch.no_grad():
                    score = self._model(tensor).item()
                scores.append(score)

            sample.quality_metrics.paq2piq_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("PaQ-2-PiQ processing failed: %s", e)
        return sample

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 4)
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames

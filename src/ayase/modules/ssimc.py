"""SSIM-C (Complex Wavelet SSIM variant) module.

SSIM-C is a variant of SSIM that operates in the complex wavelet domain,
providing better correlation with human perception than plain SSIM.

Range: 0-1 (higher = better quality, 1 = identical).

Uses the ``pyiqa`` package.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class SSIMCModule(PipelineModule):
    name = "ssimc"
    description = "SSIM-C complex wavelet structural similarity FR (higher=better)"
    default_config = {"subsample": 8}
    metric_groups = {
        "ssimc": "fr_quality",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._backend = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import pyiqa
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            self._model = pyiqa.create_metric("ssimc", device=self._device)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("SSIM-C model loaded on %s", self._device)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("SSIM-C unavailable: pyiqa not installed")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("SSIM-C unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        reference_path = getattr(sample, "reference_path", None)
        if reference_path is None:
            return sample
        try:
            import cv2
            import torch

            ref_frames = self._load_frames(str(reference_path), sample.is_video)
            dist_frames = self._load_frames(str(sample.path), sample.is_video)
            if not ref_frames or not dist_frames:
                return sample

            n = min(len(ref_frames), len(dist_frames))
            scores = []
            device = self._device
            for i in range(n):
                ref_rgb = cv2.cvtColor(ref_frames[i], cv2.COLOR_BGR2RGB)
                dist_rgb = cv2.cvtColor(dist_frames[i], cv2.COLOR_BGR2RGB)
                h = min(ref_rgb.shape[0], dist_rgb.shape[0])
                w = min(ref_rgb.shape[1], dist_rgb.shape[1])
                ref_rgb = cv2.resize(ref_rgb, (w, h))
                dist_rgb = cv2.resize(dist_rgb, (w, h))
                ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                dist_t = torch.from_numpy(dist_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    score = self._model(dist_t.to(device), ref_t.to(device)).item()
                scores.append(score)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.ssimc = float(np.mean(scores))
        except Exception as e:
            logger.warning("SSIM-C processing failed: %s", e)
        return sample

    def _load_frames(self, path: str, is_video: bool) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        frames = []
        if is_video:
            cap = cv2.VideoCapture(path)
            total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(path)
            if frame is not None:
                frames.append(frame)
        return frames

"""CKDN (Degraded-Reference IQA via Knowledge Distillation) module.

FR-IQA using knowledge distillation from teacher to student network.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CKDNModule(PipelineModule):
    name = "ckdn"
    description = "CKDN knowledge distillation FR image quality"
    default_config = {"subsample": 4}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None

    def setup(self) -> None:
        try:
            import pyiqa
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = pyiqa.create_metric("ckdn", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = torch.device("cpu")
            self._ml_available = True
            logger.info("CKDN model loaded on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("CKDN unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        # FR metric — needs reference
        reference = getattr(sample, "reference_path", None)
        if reference is None:
            return sample
        try:
            import cv2
            import torch

            ref_frames = self._load_frames(str(reference), sample.is_video)
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

            sample.quality_metrics.ckdn_score = float(np.mean(scores))
        except Exception as e:
            logger.warning("CKDN processing failed: %s", e)
        return sample

    def _load_frames(self, path: str, is_video: bool) -> list:
        import cv2

        subsample = self.config.get("subsample", 4)
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

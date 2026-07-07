"""Face-IQA module using TOPIQ face-specific variant."""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FaceIQAModule(PipelineModule):
    name = "face_iqa"
    description = "Face-specific IQA via TOPIQ-face (GFIQA-trained, higher=better)"
    default_config = {"subsample": 8}
    metric_groups = {
        "face_iqa_score": "face",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._model = None
        self._device = None
        self._face_cascade = None
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            import torch
            from ayase.runtime import resolve_torch_device

            device = torch.device(resolve_torch_device(self.config.get("device", "auto")))
            self._model = pyiqa.create_metric("topiq_nr-face", device=device)
            try:
                self._device = next(self._model.parameters()).device
            except StopIteration:
                self._device = device
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info("Face-IQA (topiq_nr-face) model loaded on %s", device)
        except (ImportError, Exception) as e:
            logger.warning("Face-IQA unavailable: %s", e)

        try:
            import cv2

            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            pass

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

            face_scores = []
            device = self._device

            for frame in frames:
                faces = self._detect_faces(frame)
                if not faces:
                    continue

                for (x, y, w, h) in faces:
                    # Pad face region by 20%
                    pad = int(max(w, h) * 0.2)
                    fy = max(0, y - pad)
                    fx = max(0, x - pad)
                    fh = min(frame.shape[0], y + h + pad) - fy
                    fw = min(frame.shape[1], x + w + pad) - fx
                    face_crop = frame[fy : fy + fh, fx : fx + fw]

                    if face_crop.size == 0:
                        continue

                    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    tensor = (
                        torch.from_numpy(rgb)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .float()
                        / 255.0
                    )
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        score = self._model(tensor).item()
                    face_scores.append(score)

            if face_scores:
                sample.quality_metrics.face_iqa_score = float(np.mean(face_scores))
        except Exception as e:
            logger.warning("Face-IQA processing failed: %s", e)
        return sample

    def _detect_faces(self, frame) -> list:
        if self._face_cascade is None:
            return []
        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
        return list(faces) if len(faces) > 0 else []

    def _load_frames(self, sample: Sample) -> list:
        # Uniformly sampled BGR frames served from the shared per-sample cache.
        subsample = self.config.get("subsample", 8)
        return sample_frames(sample.path, max_frames=subsample, color="bgr")

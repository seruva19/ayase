"""Estimate no-reference quality of faces with the TOPIQ face-specific model.

Faces are detected with RetinaFace (facexlib) and aligned by the canonical
5-landmark similarity warp to 512x512 — the alignment protocol of the GFIQA
training data — then scored with ``topiq_nr-face`` and averaged per sample.
Samples with no detected face are left unset. Higher is better; no fixed
range is assumed. Basis: https://github.com/chaofengc/IQA-PyTorch
"""

import logging
from typing import Optional

import numpy as np

from ayase.image import sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FaceIQAModule(PipelineModule):
    name = "face_iqa"
    provenance = "adapted"
    sources = {
        "face_iqa_score": "TOPIQ-face (pyiqa topiq_nr-face, Chen et al.) — https://github.com/chaofengc/IQA-PyTorch",
    }
    deviations = {
        "face_iqa_score": "input is a 5-landmark-aligned face as in the training data (GFIQA); the video extension (mean over frames) is not from the source",
    }
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
        self._detector = None
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
            from facexlib.detection import init_detection_model

            self._detector = init_detection_model(
                "retinaface_resnet50", device=self._device
            )
        except Exception as e:
            self._ml_available = False
            self._backend = "unavailable"
            logger.warning(
                "Face-IQA unavailable: facexlib retinaface detector failed (%s). "
                "Aligned-face input is required by the GFIQA protocol.", e
            )

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
                faces = self._aligned_faces(frame)
                if not faces:
                    continue

                for aligned_bgr in faces:
                    rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
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

    def _aligned_faces(self, frame) -> list:
        """RetinaFace detect + 5-landmark alignment to 512x512 (GFIQA-style)."""
        if self._detector is None:
            return []
        import cv2
        from facexlib.utils import align_crop_face_landmarks

        try:
            bboxes = self._detector.detect_faces(frame)
        except Exception:
            return []
        if bboxes is None or len(bboxes) == 0:
            return []
        faces = []
        for det in bboxes:
            landmarks = np.asarray(det[5:15], dtype=np.float32).reshape(5, 2)
            try:
                aligned = align_crop_face_landmarks(frame, landmarks, output_size=512)
            except Exception:
                continue
            if aligned is not None and aligned.size:
                faces.append(aligned)
        return faces

    def _load_frames(self, sample: Sample) -> list:
        # Uniformly sampled BGR frames served from the shared per-sample cache.
        subsample = self.config.get("subsample", 8)
        return sample_frames(sample.path, max_frames=subsample, color="bgr")

"""Localized mouth-region quality using THEval's cropped MUSIQ-SPAQ protocol.

Each detected lip region receives ten pixels of padding. Crops are zero-padded
to a common batch shape and scored by MUSIQ (SPAQ-pretrained, as in THEval);
their mean is reported. Higher is better and complements whole-frame MUSIQ by
exposing localized mouth defects.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.runtime import resolve_torch_device

from ._blendshape_utils import (
    MODEL_FILENAME, MODEL_REPO_ID, MODEL_REVISION, MODEL_URL,
    BlendshapeExtractor, select_face_index,
)
from .lip_dynamics import LIP_INDICES

logger = logging.getLogger(__name__)


def mouth_bbox(
    landmarks: Sequence[object], width: int, height: int, padding: int = 10
) -> Optional[tuple[int, int, int, int]]:
    """Return THEval's padded, frame-clamped lip bounding box."""
    if len(landmarks) <= max(LIP_INDICES) or width <= 0 or height <= 0:
        return None
    xs = [int(float(getattr(landmarks[index], "x")) * width) for index in LIP_INDICES]
    ys = [int(float(getattr(landmarks[index], "y")) * height) for index in LIP_INDICES]
    box = (
        max(min(xs) - padding, 0), max(min(ys) - padding, 0),
        min(max(xs) + padding, width), min(max(ys) + padding, height),
    )
    return box if box[2] > box[0] and box[3] > box[1] else None


class MouthQualityModule(PipelineModule):
    name = "mouth_quality"
    provenance = 'adapted'
    deviations = {'*': 'The cited score is defined per image. Ayase also writes this field for video by selecting decoded frames and aggregating their image scores; frame selection and pooling follow this module, not a published native video protocol. Image inputs use the image backend.'}
    sources = {
        "mouth_quality_score": "THEval (Quignon et al., arXiv 2511.04520) — https://arxiv.org/abs/2511.04520",
    }
    description = "THEval localized mouth-crop MUSIQ-SPAQ quality"
    default_config = {"batch_size": 64, "padding": 10, "num_faces": 1, "face_index": None}
    models = [
        {"id": "musiq-spaq", "type": "pyiqa",
         "task": "MUSIQ pretrained on SPAQ — the THEval mouth-crop scorer"},
        {"id": MODEL_REPO_ID, "type": "huggingface", "url": MODEL_URL,
         "revision": MODEL_REVISION, "task": f"Lip landmarks ({MODEL_FILENAME})"},
    ]
    metric_info = {"mouth_quality_score": "Mean MUSIQ-SPAQ score over detected mouth crops (higher=better)"}
    metric_groups = {"mouth_quality_score": "nr_quality"}

    def __init__(self, config=None):
        super().__init__(config)
        self.batch_size = max(1, int(self.config.get("batch_size", 64)))
        self.padding = max(0, int(self.config.get("padding", 10)))
        self._device = "cpu"
        self._metric = None
        self._extractor = BlendshapeExtractor(
            self.config.get("models_dir", "models"),
            config=self.config,
            num_faces=int(self.config.get("num_faces", 1)),
            face_index=self.config.get("face_index"),
        )
        self._ml_available = False
        self._backend = "unavailable"

    def setup(self) -> None:
        try:
            import pyiqa
            if not self._extractor.setup("MouthQuality"):
                return
            self._device = resolve_torch_device(self.config.get("device", "auto"))
            # THEval uses MUSIQ trained on SPAQ — pyiqa's musiq-spaq is the
            # official Google checkpoint port.
            self._metric = pyiqa.create_metric("musiq-spaq", device=self._device)
            self._ml_available = True
            self._backend = "mediapipe_face_landmarker+pyiqa_musiq_spaq"
        except ImportError:
            logger.warning("MouthQuality requires mediapipe and pyiqa")
        except Exception as exc:
            logger.warning("MouthQuality setup failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample
        try:
            score = self._score_video(sample.path)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.mouth_quality_score = score
        except Exception as exc:
            logger.warning("MouthQuality failed for %s: %s", sample.path, exc)
        return sample

    def _score_crops(self, crops: Sequence[np.ndarray]) -> list[float]:
        import torch
        scores = []
        for start in range(0, len(crops), self.batch_size):
            batch_crops = crops[start : start + self.batch_size]
            max_height = max(crop.shape[0] for crop in batch_crops)
            max_width = max(crop.shape[1] for crop in batch_crops)
            tensors = []
            for crop in batch_crops:
                canvas = np.zeros((max_height, max_width, 3), dtype=np.uint8)
                canvas[: crop.shape[0], : crop.shape[1]] = crop
                tensors.append(torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0)
            batch = torch.stack(tensors).to(self._device)
            with torch.no_grad():
                values = self._metric(batch).detach().cpu().reshape(-1).tolist()
            scores.extend(float(value) for value in values if math.isfinite(float(value)))
        return scores

    def _score_video(self, video_path: Path) -> Optional[float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            cap.release()
            return None
        crops = []
        previous_ms = -1
        frame_index = 0
        try:
            with self._extractor.create_landmarker() as landmarker:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    timestamp_ms = max(int(round(frame_index / fps * 1000.0)), previous_ms + 1)
                    previous_ms = timestamp_ms
                    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    image = self._extractor.mediapipe.Image(
                        image_format=self._extractor.mediapipe.ImageFormat.SRGB, data=rgb
                    )
                    result = landmarker.detect_for_video(image, timestamp_ms)
                    selected = select_face_index(result, self._extractor.face_index)
                    if selected is not None:
                        box = mouth_bbox(result.face_landmarks[selected], frame.shape[1], frame.shape[0], self.padding)
                        if box is not None:
                            x1, y1, x2, y2 = box
                            crops.append(rgb[y1:y2, x1:x2])
                    frame_index += 1
        finally:
            cap.release()
        scores = self._score_crops(crops) if crops else []
        return float(np.mean(scores)) if scores else None

"""MagFace -- Universal Representations for Face Recognition and Quality (CVPR 2021).

Meng et al. "MagFace: A Universal Representation for Face Recognition
and Quality Assessment" -- the magnitude (L2 norm) of an ArcFace
embedding is a reliable proxy for face image quality.  High-quality
faces produce embeddings with larger norms because the network learns
to push them further from the origin.

Implementation:
    1. Detect face with InsightFace (buffalo_l).
    2. Extract ArcFace embedding (512-d).
    3. quality = L2_norm(embedding), normalised to 0-1.

magface_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MagFaceModule(PipelineModule):
    name = "magface"
    description = "MagFace face magnitude quality (CVPR 2021)"
    default_config = {
        "subsample": 4,
        "face_model": "buffalo_l",
        "det_size": 640,
        # Normalization: typical ArcFace norms range ~15-30 for good faces
        "norm_min": 10.0,
        "norm_max": 30.0,
    }
    metric_groups = {
        "magface_score": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 4)
        self.face_model = self.config.get("face_model", "buffalo_l")
        self.det_size = self.config.get("det_size", 640)
        self.norm_min = self.config.get("norm_min", 10.0)
        self.norm_max = self.config.get("norm_max", 30.0)
        self._face_app = None
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            from insightface.app import FaceAnalysis

            self._face_app = FaceAnalysis(
                name=self.face_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
            self._ml_available = True
            self._backend = "insightface"
            logger.info("MagFace initialised with InsightFace (%s)", self.face_model)
        except ImportError:
            self._backend = "unavailable"
            logger.warning(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("MagFace setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            scores = []
            for frame in frames:
                score = self._compute_magface(frame)
                if score is not None:
                    scores.append(score)

            if scores:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.magface_score = float(np.clip(np.mean(scores), 0, 1))

        except Exception as e:
            logger.warning("MagFace failed for %s: %s", sample.path, e)

        return sample

    def _compute_magface(self, frame: np.ndarray) -> Optional[float]:
        """Compute MagFace quality for a single frame.

        Quality = normalised L2 norm of the ArcFace embedding.
        """
        faces = self._face_app.get(frame)
        if not faces:
            return None

        # Take largest face
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        embedding = face.embedding  # 512-d ArcFace embedding
        magnitude = float(np.linalg.norm(embedding))

        # Normalise to 0-1 using configured range
        quality = (magnitude - self.norm_min) / (self.norm_max - self.norm_min)
        return float(np.clip(quality, 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        """Extract frames (BGR) from a video or image via the shared cache."""
        from ayase.image import sample_frames

        return list(sample_frames(sample.path, max_frames=self.subsample, color="bgr"))

    def on_dispose(self) -> None:
        self._face_app = None
        import gc
        gc.collect()

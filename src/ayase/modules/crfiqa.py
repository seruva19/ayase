"""CR-FIQA -- Relative Classifiability Face Image Quality (CVPR 2023).

Ou et al. "CR-FIQA: Face Image Quality Assessment by Learning Sample
Relative Classifiability" -- quality is measured as the relative
classifiability of a face embedding.  High-quality faces produce
embeddings that are easy to classify (close to class centre, far from
decision boundary).

The paper trains a quality regression head alongside ArcFace that
predicts how easily classifiable each sample is.  Without the trained
regression head the best proxy is the embedding norm: in ArcFace with
the standard CosFace/ArcFace angular-margin loss the L2 norm of the
learned embedding correlates strongly with sample classifiability
(well-recognised faces are pushed to higher norms).

Implementation:
    1. Detect face with InsightFace (buffalo_l or buffalo_sc).
    2. Extract ArcFace embedding (512-d).
    3. quality = L2_norm(embedding), normalised to [0, 1].

crfiqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CRFIQAModule(PipelineModule):
    name = "crfiqa"
    description = "CR-FIQA face quality via classifiability (CVPR 2023)"
    default_config = {
        "subsample": 4,
        "face_model": "buffalo_l",
        "det_size": 640,
        # Normalization: typical ArcFace norms range ~15-30 for good faces
        "norm_min": 10.0,
        "norm_max": 30.0,
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
            logger.info("CR-FIQA initialised with InsightFace (%s)", self.face_model)
        except ImportError:
            logger.warning(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )
        except Exception as e:
            logger.warning("CR-FIQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            scores = []
            for frame in frames:
                score = self._compute_crfiqa(frame)
                if score is not None:
                    scores.append(score)

            if scores:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.crfiqa_score = float(np.clip(np.mean(scores), 0, 1))

        except Exception as e:
            logger.warning("CR-FIQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_crfiqa(self, frame: np.ndarray) -> Optional[float]:
        """Compute CR-FIQA quality for a single frame.

        Quality = normalised L2 norm of the ArcFace embedding.
        The CR-FIQA paper shows that relative classifiability correlates
        with embedding norm under angular-margin losses.  Without the
        paper's trained regression head, the norm is the best proxy.
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
        """Extract frames from video or load image."""
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

    def on_dispose(self) -> None:
        self._face_app = None
        import gc
        gc.collect()

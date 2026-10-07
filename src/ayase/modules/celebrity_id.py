"""Face identity similarity — DeepFace verification (own metric).

Computes face identity verification distance between video frames and
reference face images using DeepFace.  Lower distance = better identity match.
Inspired by the EvalCrafter celebrity_id_score dimension; its protocol is
not reproduced (different verifier wiring and aggregation).

Without ``reference_dir`` the module emits no score — identity drift between
frames of the same video is a different quantity (covered by
``face_identity_drift``) and is not written under this field.
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CelebrityIDModule(PipelineModule):
    name = "celebrity_id"
    provenance = "own"
    sources = {
        "face_id_similarity": "inspired by EvalCrafter celebrity_id_score (DeepFace) — https://github.com/evalcrafter/EvalCrafter",
    }
    description = "Face identity verification using DeepFace (own metric)"
    default_config = {
        "reference_dir": "",  # Directory of reference face images (optional)
        "num_frames": 8,
        "consistency_threshold": 0.4,  # cosine distance threshold for identity drift
        "model_name": "VGG-Face",
    }
    metric_groups = {
        "face_id_similarity": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.reference_dir = self.config.get("reference_dir", "")
        self.num_frames = self.config.get("num_frames", 8)
        self.consistency_threshold = self.config.get("consistency_threshold", 0.4)
        self.model_name = self.config.get("model_name", "VGG-Face")
        self._deepface = None
        self._ml_available = False
        self._backend = None

    def setup(self):
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._ml_available = True
            self._backend = "deepface"
            logger.info("DeepFace loaded for celebrity/identity verification.")
        except ImportError:
            self._backend = "unavailable"
            logger.warning("DeepFace not installed. Celebrity ID module disabled.")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning(f"Failed to setup DeepFace: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not self.reference_dir:
            return sample

        try:
            frames = self._load_frames(sample)
            if len(frames) < 2:
                return sample

            score = self._verify_against_references(frames)

            if score is None:
                return sample

            from ayase.models import QualityMetrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.face_id_similarity = float(score)

        except Exception as e:
            logger.warning(f"Celebrity ID check failed for {sample.path}: {e}")

        return sample

    def _verify_against_references(self, frames):
        """Compare frames against reference images (EvalCrafter mode)."""
        import glob
        import tempfile
        from pathlib import Path
        from PIL import Image

        ref_images = glob.glob(str(Path(self.reference_dir) / "*.jpg")) + \
                     glob.glob(str(Path(self.reference_dir) / "*.png"))
        if not ref_images:
            return None

        distances = []
        for frame in frames:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                Image.fromarray(frame).save(tmp_path)

            frame_dists = []
            for ref_path in ref_images:
                try:
                    result = self._deepface.verify(
                        img1_path=ref_path,
                        img2_path=tmp_path,
                        model_name=self.model_name,
                        enforce_detection=False,
                    )
                    frame_dists.append(result["distance"])
                except Exception:
                    continue

            import os
            os.unlink(tmp_path)

            if frame_dists:
                distances.append(min(frame_dists))

        if not distances:
            return None
        return float(np.mean(distances))

    def _load_frames(self, sample: Sample):
        frames = []
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                try:
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if total <= 0:
                        return frames
                    n = min(self.num_frames, total)
                    indices = np.linspace(0, total - 1, n, dtype=int)
                    for idx in indices:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if ret:
                            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                finally:
                    cap.release()
            else:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
        return frames

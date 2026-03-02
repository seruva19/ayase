"""Celebrity ID Score — EvalCrafter metric #17.

Computes face identity verification distance between video frames and
reference face images using DeepFace.  Lower distance = better identity match.

For dataset curation (no celebrity references), this module computes
frame-to-frame face identity consistency instead: it extracts face
embeddings from every sampled frame and measures how stable the identity is.
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
    description = "Face identity verification using DeepFace (EvalCrafter celebrity_id_score)"
    default_config = {
        "reference_dir": "",  # Directory of reference face images (optional)
        "num_frames": 8,
        "consistency_threshold": 0.4,  # cosine distance threshold for identity drift
        "model_name": "VGG-Face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.reference_dir = self.config.get("reference_dir", "")
        self.num_frames = self.config.get("num_frames", 8)
        self.consistency_threshold = self.config.get("consistency_threshold", 0.4)
        self.model_name = self.config.get("model_name", "VGG-Face")
        self._deepface = None
        self._ml_available = False

    def setup(self):
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._ml_available = True
            logger.info("DeepFace loaded for celebrity/identity verification.")
        except ImportError:
            logger.warning("DeepFace not installed. Celebrity ID module disabled.")
        except Exception as e:
            logger.warning(f"Failed to setup DeepFace: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._load_frames(sample)
            if len(frames) < 2:
                return sample

            if self.reference_dir:
                score = self._verify_against_references(frames)
            else:
                score = self._measure_identity_consistency(frames)

            if score is None:
                return sample

            from ayase.models import QualityMetrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.celebrity_id_score = float(score)

            if not self.reference_dir and score > self.consistency_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Face identity drift detected (distance={score:.3f})",
                        details={"face_identity_distance": float(score)},
                        recommendation="Face identity changes across frames; possible multi-person or scene change.",
                    )
                )

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

    def _measure_identity_consistency(self, frames):
        """No references: measure identity drift across frames using DeepFace embeddings."""
        import tempfile
        import os
        from PIL import Image

        embeddings = []
        for frame in frames:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                Image.fromarray(frame).save(tmp_path)
            try:
                result = self._deepface.represent(
                    img_path=tmp_path,
                    model_name=self.model_name,
                    enforce_detection=False,
                )
                if result and len(result) > 0:
                    embeddings.append(np.array(result[0]["embedding"]))
            except Exception:
                pass
            finally:
                os.unlink(tmp_path)

        if len(embeddings) < 2:
            return None

        # Cosine distance from first frame to all others
        ref = embeddings[0]
        ref_norm = ref / (np.linalg.norm(ref) + 1e-10)
        distances = []
        for emb in embeddings[1:]:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
            cos_sim = np.dot(ref_norm, emb_norm)
            distances.append(1.0 - cos_sim)  # cosine distance

        return float(np.mean(distances))

    def _load_frames(self, sample: Sample):
        frames = []
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    return frames
                n = min(self.num_frames, total)
                indices = np.linspace(0, total - 1, n, dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            else:
                img = cv2.imread(str(sample.path))
                if img is not None:
                    frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
        return frames

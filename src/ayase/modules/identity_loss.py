"""Identity Loss & Face Recognition Score — reference-based face identity metric.

Measures how well a generated image/video preserves the identity of a reference
face.  Standard metric used by IP-Adapter, DreamBooth, InstantID, and other
identity-preserving generation pipelines.

Outputs:
    identity_loss          — cosine distance 0-1 (lower = better preservation)
    face_recognition_score — cosine similarity 0-1 (higher = better match)

Requires ``sample.reference_path`` pointing to a reference face image.
Gracefully skips when no reference is provided.

Backends (real ArcFace face-recognition embeddings only):
    1. InsightFace (buffalo_l ArcFace)
    2. DeepFace (ArcFace) — fallback

The largest detected face is used. Pre-cropped face chips (an aligned face filling
the frame) are invisible to RetinaFace, so detection is retried once on a
replicate-padded copy (``pad_retry``).

Identity is defined by a face-recognition embedding; a geometric-landmark
similarity is NOT face recognition and is not used as a stand-in. When no real
face-recognition backend is available the metrics are left ``None``.
"""

import logging

import cv2
import numpy as np

from ayase.faces import detect_largest_face
from ayase.image import load_representative_frame, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class IdentityLossModule(PipelineModule):
    name = "identity_loss"
    description = "Face identity preservation metric (ArcFace cosine distance/similarity vs reference)"
    default_config = {
        "model_name": "buffalo_l",
        "subsample": 8,
        "warning_threshold": 0.5,
        # Retry detection on a replicate-padded copy when a tightly cropped face
        # leaves the detector no context (0 disables the retry).
        "pad_retry": 0.25,
    }
    metric_groups = {
        "face_recognition_score": "face",
        "identity_loss": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "buffalo_l")
        self.subsample = self.config.get("subsample", 8)
        self.warning_threshold = self.config.get("warning_threshold", 0.5)
        self.pad_retry = max(0.0, float(self.config.get("pad_retry", 0.25)))
        self._backend = None  # "insightface" | "deepface" | "unavailable"
        self._app = None  # InsightFace FaceAnalysis
        self._deepface = None

    def setup(self):
        # Tier 1: InsightFace (ArcFace)
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            self._backend = "insightface"
            logger.info("IdentityLoss: using InsightFace (ArcFace) backend.")
            return
        except Exception:
            pass

        # Tier 2: DeepFace (ArcFace)
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._backend = "deepface"
            logger.info("IdentityLoss: using DeepFace (ArcFace) backend.")
            return
        except Exception:
            pass

        self._backend = "unavailable"
        logger.warning(
            "IdentityLoss: no ArcFace face-recognition backend available "
            "(install insightface+onnxruntime or deepface); identity_loss left unset."
        )

    def process(self, sample: Sample) -> Sample:
        if self._backend not in ("insightface", "deepface"):
            return sample

        ref_path = getattr(sample, "reference_path", None)
        if ref_path is None:
            return sample

        try:
            ref_rgb = load_representative_frame(ref_path, color="rgb")
            if ref_rgb is None:
                return sample

            frames = self._load_frames(sample)
            if not frames:
                return sample

            if self._backend == "insightface":
                distance = self._compute_insightface(ref_rgb, frames)
            elif self._backend == "deepface":
                distance = self._compute_deepface(ref_path, sample, frames)
            else:
                distance = None

            if distance is None:
                return sample

            distance = float(np.clip(distance, 0.0, 1.0))
            similarity = 1.0 - distance

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.identity_loss = distance
            sample.quality_metrics.face_recognition_score = similarity

        except Exception as e:
            logger.warning(f"IdentityLoss failed for {sample.path}: {e}")

        return sample

    # -- Backend implementations ------------------------------------------------

    def _insightface_embedding(self, rgb_image):
        """Unit-norm ArcFace embedding of the largest face, or None."""
        # Inputs are RGB, but InsightFace .get() expects BGR (cv2 convention);
        # convert so detection + ArcFace embeddings are correct.
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb_image), cv2.COLOR_RGB2BGR)
        face, _ = detect_largest_face(self._app, bgr, self.pad_retry)
        if face is None:
            return None
        emb = face.embedding
        return emb / (np.linalg.norm(emb) + 1e-10)

    def _compute_insightface(self, ref_rgb, frames):
        ref_emb = self._insightface_embedding(ref_rgb)
        if ref_emb is None:
            return None

        distances = []
        for frame in frames:
            emb = self._insightface_embedding(frame)
            if emb is None:
                continue
            distances.append(1.0 - float(np.dot(ref_emb, emb)))

        return float(np.mean(distances)) if distances else None

    def _compute_deepface(self, ref_path, sample, frames):
        import tempfile
        import os
        from PIL import Image

        distances = []
        for frame in frames:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                Image.fromarray(np.ascontiguousarray(frame)).save(tmp_path)
            try:
                result = self._deepface.verify(
                    img1_path=str(ref_path),
                    img2_path=tmp_path,
                    model_name="ArcFace",
                    enforce_detection=False,
                )
                distances.append(result["distance"])
            except Exception:
                pass
            finally:
                os.unlink(tmp_path)

        return float(np.mean(distances)) if distances else None

    # -- Frame loading ----------------------------------------------------------

    def _load_frames(self, sample: Sample):
        try:
            return list(sample_frames(sample.path, max_frames=self.subsample, color="rgb"))
        except Exception as e:
            logger.debug(f"Frame loading failed: {e}")
            return []

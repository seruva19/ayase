"""Identity Loss & Face Recognition Score — reference-based face identity metric.

Measures how well a generated image/video preserves the identity of a reference
face.  Standard metric used by IP-Adapter, DreamBooth, InstantID, and other
identity-preserving generation pipelines.

Outputs:
    identity_loss          — cosine distance 0-1 (lower = better preservation)
    face_recognition_score — cosine similarity 0-1 (higher = better match)

Requires ``sample.reference_path`` pointing to a reference face image.
Gracefully skips when no reference is provided.

Tiered backends:
    1. InsightFace (buffalo_l ArcFace) — industry standard
    2. DeepFace (ArcFace) — fallback
    3. MediaPipe FaceMesh (geometric landmarks) — lightweight fallback
    4. Skip — no face models available
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class IdentityLossModule(PipelineModule):
    name = "identity_loss"
    description = "Face identity preservation metric (cosine distance/similarity vs reference)"
    default_config = {
        "model_name": "buffalo_l",
        "subsample": 8,
        "warning_threshold": 0.5,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "buffalo_l")
        self.subsample = self.config.get("subsample", 8)
        self.warning_threshold = self.config.get("warning_threshold", 0.5)
        self._backend = None  # "insightface" | "deepface" | "mediapipe"
        self._app = None  # InsightFace FaceAnalysis
        self._deepface = None
        self._mp_face_mesh = None

    def setup(self):
        # Tier 1: InsightFace
        try:
            from insightface.app import FaceAnalysis
            self._app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            self._backend = "insightface"
            logger.info("IdentityLoss: using InsightFace backend.")
            return
        except Exception:
            pass

        # Tier 2: DeepFace
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._backend = "deepface"
            logger.info("IdentityLoss: using DeepFace backend.")
            return
        except Exception:
            pass

        # Tier 3: MediaPipe FaceMesh
        try:
            import mediapipe as mp
            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
            )
            self._backend = "mediapipe"
            logger.info("IdentityLoss: using MediaPipe landmark fallback.")
            return
        except Exception:
            pass

        logger.warning("IdentityLoss: no face backend available — module disabled.")

    def process(self, sample: Sample) -> Sample:
        if self._backend is None:
            return sample

        ref_path = getattr(sample, "reference_path", None)
        if ref_path is None:
            return sample

        try:
            ref_img = cv2.imread(str(ref_path))
            if ref_img is None:
                return sample
            ref_rgb = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

            frames = self._load_frames(sample)
            if not frames:
                return sample

            if self._backend == "insightface":
                distance = self._compute_insightface(ref_rgb, frames)
            elif self._backend == "deepface":
                distance = self._compute_deepface(ref_path, sample, frames)
            else:
                distance = self._compute_mediapipe(ref_rgb, frames)

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

    def _compute_insightface(self, ref_rgb, frames):
        ref_faces = self._app.get(ref_rgb)
        if not ref_faces:
            return None
        ref_emb = ref_faces[0].embedding
        ref_emb = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)

        distances = []
        for frame in frames:
            faces = self._app.get(frame)
            if not faces:
                continue
            emb = faces[0].embedding
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            cos_sim = float(np.dot(ref_emb, emb))
            distances.append(1.0 - cos_sim)

        return float(np.mean(distances)) if distances else None

    def _compute_deepface(self, ref_path, sample, frames):
        import tempfile
        import os
        from PIL import Image

        distances = []
        for frame in frames:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                Image.fromarray(frame).save(tmp_path)
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

    def _compute_mediapipe(self, ref_rgb, frames):
        ref_lm = self._extract_landmarks(ref_rgb)
        if ref_lm is None:
            return None

        distances = []
        for frame in frames:
            lm = self._extract_landmarks(frame)
            if lm is None:
                continue
            dist = float(np.mean(np.linalg.norm(ref_lm - lm, axis=1)))
            distances.append(min(dist, 1.0))

        return float(np.mean(distances)) if distances else None

    def _extract_landmarks(self, rgb_image):
        """Extract normalized 468-point face landmarks from an RGB image."""
        results = self._mp_face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None
        lm = results.multi_face_landmarks[0]
        pts = np.array([[p.x, p.y, p.z] for p in lm.landmark])
        centroid = pts.mean(axis=0)
        pts = pts - centroid
        scale = np.linalg.norm(pts, axis=1).max() + 1e-10
        pts = pts / scale
        return pts

    # -- Frame loading ----------------------------------------------------------

    def _load_frames(self, sample: Sample):
        frames = []
        try:
            if sample.is_video:
                cap = cv2.VideoCapture(str(sample.path))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    cap.release()
                    return frames
                n = min(self.subsample, total)
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

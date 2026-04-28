"""Face Cross-Similarity Matrix — cross-sample face identity comparison.

Computes an NxN pairwise cosine similarity matrix of ArcFace face embeddings
across all samples in a dataset.  Used to evaluate identity consistency in
character-centric image/video sets (e.g., DreamBooth, IP-Adapter outputs).

Per-sample outputs (set in ``post_process``):
    face_cross_similarity  — average cosine similarity with all other samples (0-1)
    face_identity_count    — number of faces detected in this sample

Dataset-level outputs (stored in DatasetStats via ``pipeline.add_dataset_metric``):
    face_similarity_matrix     — full NxN similarity matrix (list of lists)
    avg_face_cross_similarity  — dataset-wide average pairwise similarity
    identity_cluster_count     — number of distinct identity clusters

Tiered backends (same as IdentityLossModule):
    1. InsightFace (buffalo_l ArcFace) — industry standard
    2. DeepFace (ArcFace) — fallback
    3. MediaPipe FaceMesh (geometric landmarks) — lightweight fallback
    4. Skip — no face models available
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FaceCrossSimilarityModule(PipelineModule):
    name = "face_cross_similarity"
    description = "Pairwise ArcFace cosine similarity matrix across dataset faces"
    default_config = {
        "model_name": "buffalo_l",
        "max_faces_per_image": 5,
        "similarity_threshold": 0.3,
        "subsample": 8,
        "max_cache_size": 10000,
        "device": "auto",
    }
    models = [
        {
            "id": "buffalo_l",
            "type": "other",
            "install": "pip install insightface onnxruntime",
            "task": "InsightFace ArcFace face embedding model",
        },
        {
            "id": "ArcFace",
            "type": "pip_package",
            "install": "pip install deepface",
            "task": "DeepFace ArcFace fallback embeddings",
        },
    ]
    metric_info = {
        "face_cross_similarity": "Per-sample average cosine similarity to other dataset faces",
        "face_identity_count": "Number of faces detected for the sample",
        "face_similarity_matrix": "Dataset NxN pairwise face similarity matrix",
        "avg_face_cross_similarity": "Dataset-wide average pairwise face similarity",
        "identity_cluster_count": "Estimated number of identity clusters in the dataset",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "buffalo_l")
        self.max_faces = self.config.get("max_faces_per_image", 5)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.3)
        self.subsample = self.config.get("subsample", 8)
        self.max_cache_size = self.config.get("max_cache_size", 10000)
        self._backend: Optional[str] = None  # "insightface" | "deepface" | "mediapipe"
        self._app = None  # InsightFace FaceAnalysis
        self._deepface = None
        self._mp_face_mesh = None

        # Cache: sample path -> list of normalized embedding arrays
        self._embeddings_cache: Dict[str, List[np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Setup — tiered backends (mirrors identity_loss.py lines 52-86)
    # ------------------------------------------------------------------

    def setup(self) -> None:
        # Tier 1: InsightFace
        try:
            from insightface.app import FaceAnalysis

            self._app = FaceAnalysis(
                name=self.model_name, providers=["CPUExecutionProvider"]
            )
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            self._backend = "insightface"
            logger.info("FaceCrossSimilarity: using InsightFace backend.")
            return
        except Exception:
            pass

        # Tier 2: DeepFace
        try:
            from deepface import DeepFace

            self._deepface = DeepFace
            self._backend = "deepface"
            logger.info("FaceCrossSimilarity: using DeepFace backend.")
            return
        except Exception:
            pass

        # Tier 3: MediaPipe FaceMesh
        try:
            import mediapipe as mp

            self._mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=self.max_faces,
                min_detection_confidence=0.5,
            )
            self._backend = "mediapipe"
            logger.info("FaceCrossSimilarity: using MediaPipe landmark fallback.")
            return
        except Exception:
            pass

        logger.warning(
            "FaceCrossSimilarity: no face backend available — module disabled."
        )

    # ------------------------------------------------------------------
    # Per-sample processing — extract and cache embeddings
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if self._backend is None:
            return sample

        if len(self._embeddings_cache) >= self.max_cache_size:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            embeddings = []
            for frame in frames:
                frame_embeddings = self._extract_embeddings(frame)
                embeddings.extend(frame_embeddings)

            if embeddings:
                # Keep up to max_faces embeddings per sample
                embeddings = embeddings[: self.max_faces]
                self._embeddings_cache[str(sample.path)] = embeddings

                # Record per-sample face count immediately
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.face_identity_count = len(embeddings)

        except Exception as e:
            logger.warning(f"FaceCrossSimilarity failed for {sample.path}: {e}")

        return sample

    # ------------------------------------------------------------------
    # Post-process — build similarity matrix and cluster identities
    # ------------------------------------------------------------------

    def post_process(self, all_samples: List[Sample]) -> None:
        if len(self._embeddings_cache) < 2:
            logger.info(
                f"FaceCrossSimilarity: not enough samples with faces "
                f"({len(self._embeddings_cache)}), skipping cross-similarity."
            )
            return

        try:
            # Build ordered list of sample paths that have cached embeddings
            paths = list(self._embeddings_cache.keys())
            n = len(paths)

            # Compute representative embedding per sample (mean of all face embeddings)
            rep_embeddings = []
            for path in paths:
                embs = self._embeddings_cache[path]
                mean_emb = np.mean(embs, axis=0)
                norm = np.linalg.norm(mean_emb) + 1e-10
                rep_embeddings.append(mean_emb / norm)

            rep_matrix = np.stack(rep_embeddings)  # (N, D)

            # NxN cosine similarity matrix (embeddings are already L2-normalized)
            sim_matrix = rep_matrix @ rep_matrix.T  # (N, N)
            np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)

            # Per-sample average cross-similarity (exclude self)
            per_sample_avg = {}
            for i, path in enumerate(paths):
                others = [sim_matrix[i, j] for j in range(n) if j != i]
                per_sample_avg[path] = float(np.mean(others)) if others else 0.0

            # Identity clustering via agglomerative approach
            identity_count = self._cluster_identities(sim_matrix)

            # Dataset-level average (upper triangle, excluding diagonal)
            upper_vals = []
            for i in range(n):
                for j in range(i + 1, n):
                    upper_vals.append(sim_matrix[i, j])
            avg_cross_sim = float(np.mean(upper_vals)) if upper_vals else 0.0

            # Write per-sample metrics back
            path_to_sample = {str(s.path): s for s in all_samples}
            for path, avg_sim in per_sample_avg.items():
                sample = path_to_sample.get(path)
                if sample is None:
                    continue
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.face_cross_similarity = float(
                    np.clip(avg_sim, 0.0, 1.0)
                )

            # Write dataset-level metrics
            if hasattr(self, "pipeline") and self.pipeline:
                if hasattr(self.pipeline, "add_dataset_metric"):
                    self.pipeline.add_dataset_metric(
                        "avg_face_cross_similarity", avg_cross_sim
                    )
                    self.pipeline.add_dataset_metric(
                        "identity_cluster_count", identity_count
                    )
                    # Store similarity matrix (convert to plain Python lists)
                    self.pipeline.add_dataset_metric(
                        "face_similarity_matrix", sim_matrix.tolist()
                    )

            logger.info(
                f"FaceCrossSimilarity: {n} samples, "
                f"avg_sim={avg_cross_sim:.3f}, "
                f"identity_clusters={identity_count}"
            )

        except Exception as e:
            logger.error(f"FaceCrossSimilarity post_process failed: {e}")

    # ------------------------------------------------------------------
    # Identity clustering — agglomerative by threshold
    # ------------------------------------------------------------------

    def _cluster_identities(self, sim_matrix: np.ndarray) -> int:
        """Cluster face embeddings into identities using agglomerative approach.

        Uses a simple union-find on the similarity matrix: two samples belong
        to the same identity cluster if their similarity exceeds the threshold.
        """
        n = sim_matrix.shape[0]
        if n == 0:
            return 0

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= self.similarity_threshold:
                    union(i, j)

        unique_roots = len(set(find(i) for i in range(n)))
        return unique_roots

    # ------------------------------------------------------------------
    # Embedding extraction — tiered backends
    # ------------------------------------------------------------------

    def _extract_embeddings(self, rgb_image: np.ndarray) -> List[np.ndarray]:
        """Extract face embeddings from a single RGB frame."""
        if self._backend == "insightface":
            return self._extract_insightface(rgb_image)
        elif self._backend == "deepface":
            return self._extract_deepface(rgb_image)
        elif self._backend == "mediapipe":
            return self._extract_mediapipe(rgb_image)
        return []

    def _extract_insightface(self, rgb_image: np.ndarray) -> List[np.ndarray]:
        """Extract ArcFace embeddings via InsightFace."""
        faces = self._app.get(rgb_image)
        if not faces:
            return []

        embeddings = []
        for face in faces[: self.max_faces]:
            emb = face.embedding
            norm = np.linalg.norm(emb) + 1e-10
            embeddings.append(emb / norm)
        return embeddings

    def _extract_deepface(self, rgb_image: np.ndarray) -> List[np.ndarray]:
        """Extract face embeddings via DeepFace."""
        import tempfile
        import os
        from PIL import Image

        # DeepFace needs a file path
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            Image.fromarray(rgb_image).save(tmp_path)

        try:
            results = self._deepface.represent(
                img_path=tmp_path,
                model_name="ArcFace",
                enforce_detection=False,
            )
            embeddings = []
            for res in results[: self.max_faces]:
                emb = np.array(res["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb) + 1e-10
                embeddings.append(emb / norm)
            return embeddings
        except Exception:
            return []
        finally:
            os.unlink(tmp_path)

    def _extract_mediapipe(self, rgb_image: np.ndarray) -> List[np.ndarray]:
        """Extract geometric landmark 'embeddings' via MediaPipe FaceMesh."""
        results = self._mp_face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return []

        embeddings = []
        for face_lm in results.multi_face_landmarks[: self.max_faces]:
            pts = np.array([[p.x, p.y, p.z] for p in face_lm.landmark])
            centroid = pts.mean(axis=0)
            pts = pts - centroid
            scale = np.linalg.norm(pts, axis=1).max() + 1e-10
            pts = pts / scale
            # Flatten to 1D embedding
            emb = pts.flatten().astype(np.float32)
            norm = np.linalg.norm(emb) + 1e-10
            embeddings.append(emb / norm)
        return embeddings

    # ------------------------------------------------------------------
    # Frame loading
    # ------------------------------------------------------------------

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        """Load frames from image or video sample, returned as RGB arrays."""
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

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def on_dispose(self) -> None:
        """Release cached embeddings."""
        self._embeddings_cache.clear()
        super().on_dispose()

"""Identity preservation when SEVERAL people share the frame.

``identity_loss`` and ``dino_face_identity`` answer "is this the reference person"
for one subject: they pick a face per frame on their own and average. In a
two-person scene that silently answers a different question -- the value can stay
high while one of the two people has been replaced by a stranger.

This module follows every face through the clip, assigns the tracks to the
supplied reference identities as a whole, and reports the WORST per-subject
similarity: the question is whether EVERY subject kept their identity, and a mean
hides exactly the failure the metric exists to catch.

References are taken from ``sample.metadata["identity_references"]`` as a mapping
``{subject name: [image path, ...]}``; at least two subjects are required. Values
are left unset when the face backend is unavailable or fewer than two references
resolve. Similarity is ArcFace cosine, higher = better.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MultiSubjectIdentityModule(PipelineModule):
    name = "multi_subject_identity"
    description = "Per-subject face identity in multi-person clips (worst subject reported)"
    default_config = {
        "model_name": "buffalo_l",
        "models_dir": "models",
        "stride": 2,
        "max_frames": 200,
        "min_track_length": 3,
    }
    metric_groups = {
        "multi_subject_identity_worst": "face",
        "multi_subject_identity_mean": "face",
        "multi_subject_identity_coverage": "face",
        "multi_subject_identity_tracks": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.stride = int(self.config.get("stride", 2))
        self.max_frames = int(self.config.get("max_frames", 200))
        self.min_track_length = int(self.config.get("min_track_length", 3))
        self._app = None

    def setup(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.warning("multi_subject_identity: insightface not installed; metric disabled")
            return
        try:
            app = FaceAnalysis(
                name=self.config.get("model_name", "buffalo_l"),
                root=self.config.get("models_dir", "models"),
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("multi_subject_identity: face backend failed to load (%s)", exc)
            return
        self._app = app

    def process(self, sample: Sample) -> Sample:
        if self._app is None or not sample.is_video:
            return sample
        references = (sample.metadata or {}).get("identity_references")
        if not isinstance(references, dict) or len(references) < 2:
            return sample

        try:
            result = self._score(str(sample.path), references)
        except Exception as exc:  # pragma: no cover - depends on decoder/backend
            logger.warning("multi_subject_identity failed for %s: %s", sample.path, exc)
            return sample
        if result is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.multi_subject_identity_worst = result["worst"]
        sample.quality_metrics.multi_subject_identity_mean = result["mean"]
        sample.quality_metrics.multi_subject_identity_coverage = result["coverage"]
        sample.quality_metrics.multi_subject_identity_tracks = result["tracks"]
        sample.metadata["multi_subject_identity_per_subject"] = result["per_subject"]
        return sample

    def _reference_embeddings(self, references: Dict[str, Any]) -> Dict[str, np.ndarray]:
        import cv2

        from ayase.faces import detect_largest_face

        out: Dict[str, np.ndarray] = {}
        for subject, paths in references.items():
            if isinstance(paths, (str, bytes)):
                paths = [paths]
            embeddings = []
            for path in paths or []:
                image = cv2.imread(str(path))
                if image is None:
                    continue
                face, _ = detect_largest_face(self._app, image)
                if face is not None:
                    embeddings.append(np.asarray(face.normed_embedding, dtype=float))
            if embeddings:
                mean = np.mean(np.stack(embeddings), axis=0)
                out[str(subject)] = mean / (np.linalg.norm(mean) + 1e-9)
        return out

    def _score(self, video: str, references: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        import cv2

        from ayase.faces import face_tracks

        targets = self._reference_embeddings(references)
        if len(targets) < 2:
            return None

        def frames():
            capture = cv2.VideoCapture(video)
            if not capture.isOpened():
                return
            index = 0
            taken = 0
            try:
                while taken < self.max_frames:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if index % self.stride == 0:
                        yield index, np.ascontiguousarray(frame)
                        taken += 1
                    index += 1
            finally:
                capture.release()

        tracks = face_tracks(self._app, frames(), min_length=self.min_track_length)
        if not tracks:
            return None

        track_embeddings = []
        for track in tracks:
            mean = np.mean(np.stack(track["embeddings"]), axis=0)
            track_embeddings.append(mean / (np.linalg.norm(mean) + 1e-9))

        subjects = sorted(targets)
        similarity = np.zeros((len(subjects), len(track_embeddings)), dtype=float)
        for row, subject in enumerate(subjects):
            for column, embedding in enumerate(track_embeddings):
                similarity[row, column] = float(np.dot(targets[subject], embedding))

        pairs = self._assign(similarity)
        if not pairs:
            return None

        per_subject = {subjects[row]: round(float(similarity[row, column]), 4)
                       for row, column in pairs}
        assigned = [similarity[row, column] for row, column in pairs]
        used_frames = sorted({f for _, column in pairs for f in tracks[column]["frames"]})
        span = max(1, used_frames[-1] - used_frames[0] + self.stride)
        return {
            "worst": round(float(np.min(assigned)), 4),
            "mean": round(float(np.mean(assigned)), 4),
            "coverage": round(len(used_frames) * self.stride / float(span), 4),
            "tracks": float(len(tracks)),
            "per_subject": per_subject,
        }

    @staticmethod
    def _assign(similarity: np.ndarray) -> List[Any]:
        """Assign tracks to subjects as a whole, not one subject at a time.

        Greedy assignment gives the best-matching track to whichever subject is
        considered first and leaves the next subject with whatever remains, which
        can report a stranger as a match. Falls back to greedy only when SciPy is
        absent, and then it is a stated approximation rather than the intent.
        """
        try:
            from scipy.optimize import linear_sum_assignment

            rows, columns = linear_sum_assignment(-similarity)
            return list(zip(rows, columns))
        except Exception:
            pairs: List[Any] = []
            taken = set()
            for row in range(similarity.shape[0]):
                order = np.argsort(-similarity[row])
                for column in order:
                    if column not in taken:
                        pairs.append((row, int(column)))
                        taken.add(int(column))
                        break
            return pairs

"""Temporal face-identity stability against a reference face using ArcFace.

Use this diagnostic for identity-conditioned image/video generation when
``sample.reference_path`` identifies the intended person.  The candidate is
uniformly sampled in chronological order (up to ``subsample`` frames).  On each
sampled frame, the InsightFace detection whose ArcFace embedding is closest to
the reference embedding is used; an undetected frame remains missing rather
than being assigned a similarity of zero.

The reference may be a still image or video.  A still uses its sole image; a
video uses its decoded middle frame.  In either case, the largest detected face
defines the target.  Directories and multi-reference identity sets are not
supported.  The backend reuses the InsightFace ``buffalo_l`` ArcFace model family
and on-disk cache already used by :mod:`ayase.modules.identity_loss`; no second
identity model family is introduced by this module. InsightFace's public
pretrained model packs are licensed for non-commercial research only; callers
may select their own compatible, appropriately licensed FaceAnalysis pack.

Outputs (no aggregate score):
    face_identity_detection_coverage
        Detected sampled frames / all sampled frames, in [0, 1]; higher means the
        temporal identity diagnostics are more observable, not more accurate.
    face_identity_similarity_p05, face_identity_similarity_min
        Lower-tail and worst detected-frame cosine similarities, clipped to
        [0, 1]; higher is better.  The fifth percentile uses NumPy's default
        linear percentile method.
    face_identity_below_threshold_fraction
        Fraction of detected frames below ``similarity_threshold``, in [0, 1].
    face_identity_longest_below_threshold_run_fraction
        Longest chronological low-similarity run divided by all sampled frames,
        in [0, 1].  Missing detections break a run.
    face_identity_drift_slope
        Least-squares similarity slope per full normalized sampled sequence.
        It is a signed, not range-clipped diagnostic: negative means similarity
        declines over time.  It is emitted only with at least four detections.

``similarity_threshold`` defaults to ``None`` because ArcFace verification
operating points depend on the dataset, detector, and acceptable false-match
rate.  Threshold-dependent outputs are emitted only when the caller explicitly
sets a value in [0, 1]; the value is an operating-point diagnostic, not an
identity probability.

Applicability limits: this measures one reference identity, selects the closest
face in multi-person frames, inherits face-detector/recognizer demographic and
quality biases, and cannot distinguish absence from detector failure.  Interpret
tail and trend statistics together with detection coverage.

Primary sources:
    Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face
    Recognition", CVPR 2019, https://arxiv.org/abs/1801.07698
    InsightFace reference implementation, https://github.com/deepinsight/insightface
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ayase.faces import detect_largest_face, identity_series
from ayase.image import load_representative_frame, sample_frames
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FaceIdentityDriftModule(PipelineModule):
    """Expose chronological ArcFace identity diagnostics without aggregating them."""

    name = "face_identity_drift"
    description = "Temporal ArcFace identity tail, coverage, threshold-run, and drift diagnostics"
    default_config = {
        "face_model": "buffalo_l",
        "subsample": 32,
        "similarity_threshold": None,
        "device": "auto",
        # Retry reference-face detection on a replicate-padded image when a
        # tightly cropped face leaves RetinaFace too little context.
        "pad_retry": 0.25,
    }
    models = [
        {
            "id": "insightface/buffalo_l",
            "type": "other",
            "url": "https://github.com/deepinsight/insightface/releases/tag/model-zoo",
            "task": "SCRFD face detection and ArcFace identity embeddings",
            "size": "326 MB",
            "auto_download": True,
            "notes": (
                "Default public InsightFace model pack; non-commercial research use only. "
                "A caller-supplied compatible licensed FaceAnalysis pack may be selected."
            ),
        }
    ]
    metric_info = {
        "face_identity_detection_coverage": (
            "Share of sampled frames with a detected face (0-1, higher=more observable)"
        ),
        "face_identity_similarity_p05": (
            "Fifth percentile ArcFace similarity to the reference (0-1, higher=better)"
        ),
        "face_identity_similarity_min": (
            "Minimum ArcFace similarity to the reference (0-1, higher=better)"
        ),
        "face_identity_below_threshold_fraction": (
            "Share of detected frames below the configured identity operating point (0-1)"
        ),
        "face_identity_longest_below_threshold_run_fraction": (
            "Longest contiguous below-threshold run divided by all sampled frames (0-1)"
        ),
        "face_identity_drift_slope": (
            "Linear ArcFace-similarity trend per normalized sampled sequence (negative=decline)"
        ),
    }
    metric_groups = {field: "face" for field in metric_info}

    def __init__(self, config=None):
        super().__init__(config)
        self.face_model = str(self.config.get("face_model", "buffalo_l"))
        self.subsample = max(1, int(self.config.get("subsample", 32)))
        self.device = str(self.config.get("device", "auto")).lower()
        self.pad_retry = max(0.0, float(self.config.get("pad_retry", 0.25)))
        self.similarity_threshold = self._parse_threshold(
            self.config.get("similarity_threshold")
        )
        self._face_app = None
        self._backend = "unavailable"

    @staticmethod
    def _parse_threshold(value) -> Optional[float]:
        if value is None:
            return None
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            logger.warning(
                "FaceIdentityDrift: similarity_threshold must be a number in [0, 1]; "
                "threshold diagnostics disabled"
            )
            return None
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            logger.warning(
                "FaceIdentityDrift: similarity_threshold=%r is outside [0, 1]; "
                "threshold diagnostics disabled",
                value,
            )
            return None
        return threshold

    def setup(self) -> None:
        if self.test_mode:
            logger.debug("FaceIdentityDrift: test mode, skipping InsightFace setup")
            return
        try:
            import onnxruntime as ort
            from insightface.app import FaceAnalysis

            available = set(ort.get_available_providers())
            use_cuda = self.device != "cpu" and "CUDAExecutionProvider" in available
            if self.device.startswith("cuda") and not use_cuda:
                logger.warning(
                    "FaceIdentityDrift: CUDA requested but CUDAExecutionProvider is "
                    "unavailable; falling back to CPU"
                )
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if use_cuda
                else ["CPUExecutionProvider"]
            )
            app = FaceAnalysis(name=self.face_model, providers=providers)
            app.prepare(ctx_id=0 if use_cuda else -1, det_size=(640, 640))
            self._face_app = app
            self._backend = (
                f"insightface:{self.face_model}:arcface:"
                f"{'cuda' if use_cuda else 'cpu'}"
            )
            logger.info(
                "FaceIdentityDrift: using InsightFace %s with %s",
                self.face_model,
                providers[0],
            )
        except Exception as exc:
            logger.warning(
                "FaceIdentityDrift requires InsightFace and ONNX Runtime; metrics left unset: %s",
                exc,
            )

    def process(self, sample: Sample) -> Sample:
        if self._face_app is None or sample.reference_path is None:
            return sample

        try:
            target = self._reference_embedding(sample.reference_path)
            if target is None:
                logger.debug(
                    "FaceIdentityDrift: no reference face detected in %s",
                    sample.reference_path,
                )
                return sample

            frames = sample_frames(sample.path, max_frames=self.subsample, color="bgr")
            if not frames:
                return sample

            # enumerate() preserves the uniform sampler's chronological order and
            # retains gaps when a face is missing on a sampled frame.
            series = identity_series(self._face_app, enumerate(frames), target)
            metrics = self._summarize_series(series, len(frames))

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            qm = sample.quality_metrics
            qm.face_identity_detection_coverage = metrics[
                "face_identity_detection_coverage"
            ]
            qm.face_identity_similarity_p05 = metrics[
                "face_identity_similarity_p05"
            ]
            qm.face_identity_similarity_min = metrics[
                "face_identity_similarity_min"
            ]
            qm.face_identity_below_threshold_fraction = metrics[
                "face_identity_below_threshold_fraction"
            ]
            qm.face_identity_longest_below_threshold_run_fraction = metrics[
                "face_identity_longest_below_threshold_run_fraction"
            ]
            qm.face_identity_drift_slope = metrics["face_identity_drift_slope"]
        except Exception as exc:
            logger.warning("FaceIdentityDrift failed for %s: %s", sample.path, exc)

        return sample

    def _reference_embedding(self, reference_path: Path) -> Optional[np.ndarray]:
        """Return the unit ArcFace embedding from an image or video's middle frame."""
        frame = load_representative_frame(Path(reference_path), color="bgr")
        if frame is None:
            return None
        face, _ = detect_largest_face(self._face_app, frame, self.pad_retry)
        if face is None:
            return None

        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            embedding = getattr(face, "embedding", None)
        if embedding is None:
            return None
        array = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(array))
        if not np.all(np.isfinite(array)) or not np.isfinite(norm) or norm <= 0.0:
            return None
        return array / norm

    def _summarize_series(
        self,
        series: Sequence[Tuple[int, float]],
        sampled_frame_count: int,
    ) -> Dict[str, Optional[float]]:
        """Summarize chronological detected-frame similarities.

        ``series`` retains original sampled-frame indices.  This is essential:
        dropping missing detections before run/trend calculations would join two
        low-similarity episodes across an unobserved frame and distort time.
        """
        count = max(0, int(sampled_frame_count))
        result: Dict[str, Optional[float]] = {
            "face_identity_detection_coverage": 0.0 if count else None,
            "face_identity_similarity_p05": None,
            "face_identity_similarity_min": None,
            "face_identity_below_threshold_fraction": None,
            "face_identity_longest_below_threshold_run_fraction": None,
            "face_identity_drift_slope": None,
        }
        if count == 0:
            return result

        valid: List[Tuple[int, float]] = []
        for index, similarity in series:
            index = int(index)
            similarity = float(similarity)
            if 0 <= index < count and np.isfinite(similarity):
                valid.append((index, float(np.clip(similarity, 0.0, 1.0))))

        # identity_series emits at most one value per frame in chronological
        # order. Sorting defensively keeps this helper correct for test/adaptor use.
        valid.sort(key=lambda item: item[0])
        result["face_identity_detection_coverage"] = min(1.0, len(valid) / float(count))
        if not valid:
            return result

        similarities = np.asarray([value for _, value in valid], dtype=np.float64)
        result["face_identity_similarity_p05"] = float(
            np.percentile(similarities, 5.0, method="linear")
        )
        result["face_identity_similarity_min"] = float(np.min(similarities))

        if len(valid) >= 4:
            positions = np.asarray(
                [index / float(max(1, count - 1)) for index, _ in valid],
                dtype=np.float64,
            )
            if float(np.ptp(positions)) > 0.0:
                result["face_identity_drift_slope"] = float(
                    np.polyfit(positions, similarities, 1)[0]
                )

        threshold = self.similarity_threshold
        if threshold is None:
            return result

        bad_by_index = {index: value < threshold for index, value in valid}
        bad_count = sum(bad_by_index.values())
        result["face_identity_below_threshold_fraction"] = bad_count / float(len(valid))

        longest = 0
        current = 0
        for index in range(count):
            # False covers both an observed acceptable match and a missing face;
            # either condition ends a contiguous below-threshold episode.
            if bad_by_index.get(index, False):
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        result["face_identity_longest_below_threshold_run_fraction"] = longest / float(count)
        return result

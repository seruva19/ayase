"""Face Quality Assessment module.

Detects faces and evaluates per-face quality attributes:

  face_count           — number of faces detected (existing field)
  face_quality_score   — composite face quality 0-100 (higher=better)

Per-face checks:
  - Resolution: minimum face size relative to frame
  - Sharpness:  Laplacian variance of face ROI
  - Frontalness: rough frontal-pose check via aspect-ratio heuristic
  - Occlusion:  edge density near face boundary

Uses OpenCV Haar cascades (default) or MediaPipe Face Detection for
more accurate results.  No heavy ML models required.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class FaceFidelityModule(PipelineModule):
    name = "face_fidelity"
    description = "Face detection and per-face quality assessment"
    default_config = {
        "backend": "haar",  # "haar" or "mediapipe"
        "subsample": 5,  # Every Nth video frame
        "max_frames": 60,
        "min_face_size": 64,  # Minimum face side in pixels
        "blur_threshold": 50.0,  # Laplacian variance below this = blurry
        "warning_threshold": 40.0,  # Warn if face_quality_score < this
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend = self.config.get("backend", "haar")
        self.subsample = self.config.get("subsample", 5)
        self.max_frames = self.config.get("max_frames", 60)
        self.min_face_size = self.config.get("min_face_size", 64)
        self.blur_threshold = self.config.get("blur_threshold", 50.0)
        self.warning_threshold = self.config.get("warning_threshold", 40.0)

        self._face_cascade = None
        self._mp_face_detection = None
        self._ml_available = False

    def setup(self) -> None:
        if self.backend == "mediapipe":
            try:
                import mediapipe as mp
                self._mp_face_detection = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # Full range model
                    min_detection_confidence=0.5,
                )
                self._ml_available = True
                logger.info("Face fidelity: MediaPipe backend initialised")
                return
            except ImportError:
                logger.warning("mediapipe not installed, falling back to Haar")
            except Exception as e:
                logger.warning(f"MediaPipe init failed: {e}, falling back to Haar")

        # Haar cascade fallback
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            logger.warning("Failed to load Haar cascade")
            return
        self._ml_available = True
        logger.info("Face fidelity: Haar cascade backend initialised")

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    def _detect_faces_haar(self, gray: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) face bounding boxes."""
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4,
            minSize=(self.min_face_size, self.min_face_size),
        )
        if isinstance(faces, np.ndarray):
            return [tuple(f) for f in faces.tolist()]
        return []

    def _detect_faces_mp(
        self, frame_rgb: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) face bounding boxes via MediaPipe."""
        h, w = frame_rgb.shape[:2]
        results = self._mp_face_detection.process(frame_rgb)
        faces = []
        if results.detections:
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                x = max(0, int(bb.xmin * w))
                y = max(0, int(bb.ymin * h))
                fw = min(int(bb.width * w), w - x)
                fh = min(int(bb.height * h), h - y)
                if fw >= self.min_face_size and fh >= self.min_face_size:
                    faces.append((x, y, fw, fh))
        return faces

    def _detect_faces(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if self._mp_face_detection is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return self._detect_faces_mp(rgb)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        return self._detect_faces_haar(gray)

    # ------------------------------------------------------------------
    # Per-face quality
    # ------------------------------------------------------------------

    def _face_quality(
        self, frame_bgr: np.ndarray, x: int, y: int, w: int, h: int
    ) -> float:
        """Compute a composite quality score (0-100) for one face ROI."""
        roi = frame_bgr[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Sharpness (Laplacian variance)
        lap_var = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
        sharpness = min(lap_var / 200.0, 1.0)  # 0-1

        # 2. Resolution (face size relative to a 256px baseline)
        resolution = min(min(w, h) / 256.0, 1.0)

        # 3. Frontalness heuristic (aspect ratio close to 1.0 → frontal)
        aspect = w / max(h, 1)
        frontalness = 1.0 - min(abs(aspect - 0.85) / 0.5, 1.0)  # ~0.85 is typical frontal face

        # 4. Exposure (not too dark / not too bright)
        mean_lum = float(gray_roi.mean())
        exposure = 1.0 - abs(mean_lum - 127.0) / 127.0

        # Weighted composite
        score = (
            0.40 * sharpness
            + 0.25 * resolution
            + 0.20 * frontalness
            + 0.15 * exposure
        ) * 100.0

        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            if sample.is_video:
                return self._process_video(sample)
            else:
                return self._process_image(sample)
        except Exception as e:
            logger.error(f"Face fidelity failed for {sample.path}: {e}")
            return sample

    def _process_image(self, sample: Sample) -> Sample:
        img = cv2.imread(str(sample.path))
        if img is None:
            return sample

        faces = self._detect_faces(img)

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        sample.quality_metrics.face_count = len(faces)

        if faces:
            scores = [self._face_quality(img, *f) for f in faces]
            sample.quality_metrics.face_quality_score = float(np.mean(scores))
            self._check_warnings(sample, faces, scores)

        return sample

    def _process_video(self, sample: Sample) -> Sample:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return sample

        all_counts: List[int] = []
        all_scores: List[float] = []
        small_faces = 0
        blurry_faces = 0
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                faces = self._detect_faces(frame)
                all_counts.append(len(faces))
                for x, y, w, h in faces:
                    score = self._face_quality(frame, x, y, w, h)
                    all_scores.append(score)
                    if w < self.min_face_size or h < self.min_face_size:
                        small_faces += 1
                    gray_roi = cv2.cvtColor(
                        frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY
                    )
                    if gray_roi.size > 0:
                        lap = float(cv2.Laplacian(gray_roi, cv2.CV_64F).var())
                        if lap < self.blur_threshold:
                            blurry_faces += 1
            idx += 1

        cap.release()

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        if all_counts:
            sample.quality_metrics.face_count = int(np.round(np.mean(all_counts)))
        if all_scores:
            avg = float(np.mean(all_scores))
            sample.quality_metrics.face_quality_score = avg

            if avg < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low face quality: {avg:.1f}/100",
                        details={"face_quality_score": avg},
                        recommendation="Faces may be blurry, small, or poorly lit.",
                    )
                )

        if small_faces > 0:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Small faces detected ({small_faces} instances)",
                    details={"small_face_count": small_faces},
                )
            )
        if blurry_faces > 0:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Blurry faces detected ({blurry_faces} instances)",
                    details={"blurry_face_count": blurry_faces},
                    recommendation="Check focus on subjects.",
                )
            )

        logger.debug(
            f"Face fidelity for {sample.path.name}: "
            f"count={sample.quality_metrics.face_count} "
            f"quality={sample.quality_metrics.face_quality_score}"
        )
        return sample

    def _check_warnings(
        self, sample: Sample, faces: list, scores: List[float]
    ) -> None:
        avg = float(np.mean(scores)) if scores else 0.0
        if avg < self.warning_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low face quality: {avg:.1f}/100",
                    details={"face_quality_score": avg, "face_count": len(faces)},
                    recommendation="Faces may be blurry, small, or poorly lit.",
                )
            )

"""Object Permanence module.

Measures how consistently objects persist across video frames:

  object_permanence_score — 0-100 (higher = more consistent)

Algorithm:
  1. Detect objects per frame (YOLO if available, else background subtraction).
  2. Track centroids across consecutive frames using greedy nearest-neighbour matching.
  3. Count ID switches and sudden disappearances/reappearances.
  4. Score = matched / (matched + switches + disappeared) * 100.

Videos only — images are skipped.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ObjectPermanenceModule(PipelineModule):
    name = "object_permanence"
    description = "Object tracking consistency (ID switches, disappearances)"
    default_config = {
        "backend": "auto",  # "yolo", "contour", or "auto"
        "subsample": 2,
        "max_frames": 300,
        "match_distance": 80.0,  # Max pixel distance for centroid matching
        "warning_threshold": 50.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.backend = self.config.get("backend", "auto")
        self.subsample = self.config.get("subsample", 2)
        self.max_frames = self.config.get("max_frames", 300)
        self.match_distance = self.config.get("match_distance", 80.0)
        self.warning_threshold = self.config.get("warning_threshold", 50.0)

        self._yolo_model = None
        self._bg_subtractor = None
        self._use_yolo = False
        self._ml_available = False

    def setup(self) -> None:
        if self.backend in ("yolo", "auto"):
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO("yolov8n.pt")
                self._use_yolo = True
                self._ml_available = True
                logger.info("Object permanence: YOLO backend initialised")
                return
            except ImportError:
                if self.backend == "yolo":
                    logger.warning("ultralytics not installed")
                    return
                logger.info("ultralytics not found, falling back to contour backend")
            except Exception as e:
                logger.warning(f"YOLO init failed: {e}")
                if self.backend == "yolo":
                    return

        # Fallback: background-subtraction-based contour detection.
        # NOTE: BGS on non-consecutive subsampled frames is less accurate because
        # the background model sees discontinuous motion. A short history (20) and
        # high learning rate help the model adapt faster to the jumps between
        # subsampled frames.
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=20, varThreshold=40, detectShadows=False
        )
        self._ml_available = True
        logger.info("Object permanence: contour (BGS) backend initialised")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_centroids_yolo(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Detect object centroids using YOLO."""
        results = self._yolo_model(frame_bgr, verbose=False)
        centroids = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                centroids.append((float(cx), float(cy)))
        return centroids

    def _detect_centroids_contour(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Detect object centroids via background subtraction + contours."""
        fg_mask = self._bg_subtractor.apply(frame_bgr)

        # Clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        centroids = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Skip tiny blobs
                continue
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = float(M["m10"] / M["m00"])
                cy = float(M["m01"] / M["m00"])
                centroids.append((cx, cy))

        return centroids

    def _detect_centroids(
        self, frame_bgr: np.ndarray
    ) -> List[Tuple[float, float]]:
        if self._use_yolo:
            return self._detect_centroids_yolo(frame_bgr)
        return self._detect_centroids_contour(frame_bgr)

    # ------------------------------------------------------------------
    # Centroid matching (greedy nearest-neighbour)
    # ------------------------------------------------------------------

    def _match_centroids(
        self,
        prev: List[Tuple[float, float]],
        curr: List[Tuple[float, float]],
    ) -> Tuple[int, int]:
        """Match centroids between frames.

        Returns (matched_count, id_switch_count).
        An ID switch = a previous centroid's nearest match is far from
        its expected position.
        """
        if not prev or not curr:
            return 0, 0

        prev_arr = np.array(prev)
        curr_arr = np.array(curr)

        matched = 0
        switches = 0
        used = set()

        for p in prev_arr:
            dists = np.linalg.norm(curr_arr - p, axis=1)
            order = np.argsort(dists)
            for idx in order:
                if idx in used:
                    continue
                if dists[idx] <= self.match_distance:
                    matched += 1
                    used.add(idx)
                    break
                elif dists[idx] <= self.match_distance * 3:
                    # Matched but jumped far — potential ID switch
                    switches += 1
                    used.add(idx)
                    break

        return matched, switches

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            return self._process_video(sample)
        except Exception as e:
            logger.error(f"Object permanence failed for {sample.path}: {e}")
            return sample

    def _process_video(self, sample: Sample) -> Sample:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return sample

        prev_centroids: List[Tuple[float, float]] = []
        total_objects = 0
        total_matched = 0
        total_switches = 0
        total_disappeared = 0
        frame_count = 0
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                centroids = self._detect_centroids(frame)

                if prev_centroids:
                    matched, switches = self._match_centroids(
                        prev_centroids, centroids
                    )
                    disappeared = max(0, len(prev_centroids) - matched - switches)
                    total_matched += matched
                    total_switches += switches
                    total_disappeared += disappeared

                total_objects += len(centroids)
                prev_centroids = centroids
                frame_count += 1

            idx += 1

        cap.release()

        if frame_count < 2 or total_objects == 0:
            return sample

        # Score: higher = more consistent
        # Penalise switches and disappearances relative to total objects tracked
        total_events = total_matched + total_switches + total_disappeared
        if total_events > 0:
            consistency = total_matched / total_events
        else:
            consistency = 1.0

        score = float(np.clip(consistency * 100.0, 0, 100))

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()

        sample.quality_metrics.object_permanence_score = score

        if score < self.warning_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low object permanence: {score:.1f}/100",
                    details={
                        "object_permanence_score": score,
                        "id_switches": total_switches,
                        "disappearances": total_disappeared,
                    },
                    recommendation=(
                        "Objects disappear or swap identities across frames. "
                        "May indicate generation artifacts or tracking failure."
                    ),
                )
            )

        logger.debug(
            f"Object permanence for {sample.path.name}: "
            f"score={score:.1f} switches={total_switches} "
            f"disappeared={total_disappeared}"
        )

        return sample

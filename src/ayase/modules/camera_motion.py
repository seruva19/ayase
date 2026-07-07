"""Camera motion stability analysis via ORB feature matching and homography estimation.

Separates global camera motion from local scene motion to detect jitter/shake.
Returns camera_motion_score (0-1, higher = more stable). Inspired by VMBench."""

import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class CameraMotionModule(PipelineModule):
    name = "camera_motion"
    description = "Analyzes camera motion stability (VMBench) using Homography"
    default_config = {}
    metric_groups = {
        "camera_motion_score": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "algorithmic"

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            self._analyze_stability(sample)
        except Exception as e:
            logger.warning(f"Camera motion check failed: {e}")

        return sample

    def _analyze_stability(self, sample: Sample) -> None:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return

        motion_errors = []
        try:
            # Read first frame
            ret, prev_frame = cap.read()
            if not ret:
                return

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

            # Feature detector (ORB is fast)
            orb = cv2.ORB_create(nfeatures=500)
            kp1, des1 = orb.detectAndCompute(prev_gray, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            frame_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check every 5th frame
                if frame_idx % 5 != 0:
                    frame_idx += 1
                    continue

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                kp2, des2 = orb.detectAndCompute(gray, None)

                if des1 is not None and des2 is not None and len(des1) > 10 and len(des2) > 10:
                    matches = bf.match(des1, des2)
                    matches = sorted(matches, key=lambda x: x.distance)

                    # Take top matches
                    good_matches = matches[:50]

                    if len(good_matches) > 10:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        # Find Homography
                        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                        if M is not None:
                            # Translation magnitude between frames
                            trans = np.sqrt(M[0, 2] ** 2 + M[1, 2] ** 2)

                            # Global motion magnitude (translation) vs local residual
                            global_motion = trans

                            h, w = gray.shape
                            aligned_prev = cv2.warpPerspective(prev_gray, M, (w, h))
                            diff = cv2.absdiff(gray, aligned_prev)
                            local_motion = np.mean(diff)

                            if global_motion > 0.1:
                                ratio = local_motion / (global_motion + 1e-7)
                                if ratio < 0.2 and global_motion > 5.0:
                                    # Dominant camera motion (pan over static scene)
                                    sample.validation_issues.append(
                                        ValidationIssue(
                                            severity=ValidationSeverity.INFO,
                                            message=f"Dominant camera motion detected (ratio: {ratio:.2f})",
                                            details={"global_motion": float(global_motion), "local_motion": float(local_motion)},
                                        )
                                    )

                            # Large inter-frame translation indicates shake
                            if trans > 50.0:
                                motion_errors.append(trans)

                prev_gray = gray
                kp1, des1 = kp2, des2
                frame_idx += 1
        finally:
            cap.release()

        # Store camera motion score as a stability metric (0-1, higher = more stable)
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if motion_errors:
            avg_shake = float(np.mean(motion_errors))
            # Map shake magnitude to 0-1 stability score
            stability = 1.0 / (1.0 + avg_shake / 20.0)
            sample.quality_metrics.camera_motion_score = round(stability, 4)
        else:
            sample.quality_metrics.camera_motion_score = 1.0  # No shake detected

        if len(motion_errors) > 3:
            avg_shake = np.mean(motion_errors)
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"High Camera Shake Detected (Avg Shift: {avg_shake:.1f}px)",
                    details={"camera_shake": float(avg_shake)}
                )
            )

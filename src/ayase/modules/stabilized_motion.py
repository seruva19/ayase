import cv2
import numpy as np
import logging

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class StabilizedMotionModule(PipelineModule):
    name = "stabilized_motion"
    description = "Calculates motion scores with camera stabilization (ORB+Homography)"
    default_config = {
        "step": 2,
        "threshold_px": 0.5,
        "stabilize": True,
        "high_camera_motion_threshold": 5.0,
        "static_threshold": 0.1,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.step = self.config.get("step", 2)
        self.threshold_px = self.config.get("threshold_px", 0.5)
        self.stabilize = self.config.get("stabilize", True)
        self.high_camera_motion_threshold = self.config.get("high_camera_motion_threshold", 5.0)
        self.static_threshold = self.config.get("static_threshold", 0.1)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            motion_stats = self._calculate_motion_score(str(sample.path))

            if motion_stats:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()

                sample.quality_metrics.stabilized_motion_score = motion_stats["scene_motion_mean"]
                sample.quality_metrics.stabilized_camera_score = motion_stats["camera_motion_mean"]

                # You might want to flag static videos or excessively shaky ones
                if (
                    motion_stats["label"] == "static"
                    and motion_stats["scene_motion_mean"] < self.static_threshold
                ):
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message="Video appears static",
                            details=motion_stats,
                        )
                    )
                elif motion_stats["camera_motion_mean"] > self.high_camera_motion_threshold:
                    sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.WARNING,
                            message=f"High Camera Motion: {motion_stats['camera_motion_mean']:.1f}",
                            details=motion_stats,
                        )
                    )

        except Exception as e:
            logger.warning(f"Stabilized motion check failed for {sample.path}: {e}")

        return sample

    def _calculate_motion_score(self, video_path):
        cap = cv2.VideoCapture(video_path)
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return None

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 1)

        orb = cv2.ORB_create(1000) if self.stabilize else None
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) if self.stabilize else None

        flow_mags = []
        active_ratios = []
        camera_motions = []

        while True:
            for _ in range(self.step):
                ret, frame = cap.read()
                if not ret:
                    break
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 1)

            camera_motion = 0.0
            stabilized_curr = curr_gray

            if self.stabilize:
                try:
                    kp1, des1 = orb.detectAndCompute(prev_gray, None)
                    kp2, des2 = orb.detectAndCompute(curr_gray, None)

                    if des1 is not None and des2 is not None and len(kp1) > 8 and len(kp2) > 8:
                        matches = bf.match(des1, des2)
                        if len(matches) > 10:
                            matches = sorted(matches, key=lambda x: x.distance)[:200]
                            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(
                                -1, 1, 2
                            )
                            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(
                                -1, 1, 2
                            )

                            H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
                            if H is not None:
                                h, w = prev_gray.shape
                                stabilized_curr = cv2.warpPerspective(curr_gray, H, (w, h))

                                corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(
                                    -1, 1, 2
                                )
                                transformed = cv2.perspectiveTransform(corners, H)
                                camera_motion = np.mean(
                                    np.linalg.norm(transformed - corners, axis=2)
                                )
                except Exception:
                    logger.debug("Failed to compute homography for stabilized motion.")

            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    stabilized_curr,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                flow_mags.append(np.mean(mag))
                active_ratios.append(np.mean(mag > self.threshold_px))
                camera_motions.append(camera_motion)
            except Exception:
                logger.debug("Failed to compute camera motion for stabilized motion.")

            prev_gray = stabilized_curr

        cap.release()

        if not flow_mags:
            return None

        scene_motion = float(np.median(flow_mags))
        active = float(np.median(active_ratios))
        camera = float(np.median(camera_motions)) if camera_motions else 0.0

        if scene_motion > 0.4 or active > 0.08:
            label = "dynamic"
        elif scene_motion > 0.15 or active > 0.02:
            label = "mixed"
        else:
            label = "static"

        return {
            "scene_motion_mean": scene_motion,
            "camera_motion_mean": camera,
            "active_ratio": active,
            "label": label,
        }

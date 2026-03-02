import logging

import numpy as np
import cv2
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class PhysicsModule(PipelineModule):
    name = "physics"
    description = "CoTracker / FVMD (Trajectory analysis) - Keypoint Proxy"
    default_config = {}

    def __init__(self, config=None):
        super().__init__(config)
    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            # CoTracker is a specific Meta model.
            # FVMD requires calculating Fréchet distance of motion vectors against a reference dataset.
            # CoTracker requires the CoTracker model (Meta Research).

            # CURRENT IMPLEMENTATION: Lightweight Proxy
            # We track keypoints (ORB/Shi-Tomasi) and check for "impossible" accelerations
            # or chaotic trajectories that violate physics (teleportation/glitches).
            # This serves as a "Sanity Check" version of FVMD.

            self._analyze_trajectories(sample)

        except Exception as e:
            logger.warning(f"Physics/Trajectory check failed: {e}")

        return sample

    def _analyze_trajectories(self, sample: Sample) -> None:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 10:
            cap.release()
            return

        # Parameters for ShiTomasi corner detection
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

        # Parameters for Lucas Kanade optical flow
        lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        
        # Analyze 3 windows: Start, Middle, End
        # Window size = 15 frames
        window_size = 15
        windows = [
            0,
            max(0, total_frames // 2 - window_size // 2),
            max(0, total_frames - window_size - 1)
        ]
        windows = sorted(list(set(windows))) # Unique and sorted
        
        max_acceleration = 0.0

        for start_frame in windows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, old_frame = cap.read()
            if not ret:
                continue

            old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
            p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

            if p0 is None:
                continue

            prev_velocities = np.zeros((len(p0), 2))
            
            # Sub-loop for window
            for i in range(window_size):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    velocities = good_new - good_old
                    
                    # Match velocities
                    curr_prev_vel = prev_velocities[st.flatten() == 1]
                    
                    if len(curr_prev_vel) == len(velocities):
                        acc = velocities - curr_prev_vel
                        acc_mag = np.linalg.norm(acc, axis=1)
                        if len(acc_mag) > 0:
                            frame_max_acc = np.max(acc_mag)
                            max_acceleration = max(max_acceleration, frame_max_acc)
                            
                    old_gray = frame_gray.copy()
                    p0 = good_new.reshape(-1, 1, 2)
                    prev_velocities = velocities
                else:
                    break

        cap.release()

        # Heuristic: If a point accelerates > 50px/frame^2, it's likely a glitch or teleportation
        if max_acceleration > 50.0:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Physically Implausible Motion (Max Accel: {max_acceleration:.1f}px/fr^2)",
                    details={"max_acceleration": float(max_acceleration)},
                )
            )

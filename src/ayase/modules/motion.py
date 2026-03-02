import logging
import cv2
import numpy as np
from typing import Optional, List

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class MotionModule(PipelineModule):
    name = "motion"
    description = "Analyzes motion dynamics (optical flow, flickering)"
    default_config = {
        "sample_rate": 5,
        "low_motion_threshold": 0.5,
        "high_motion_threshold": 20.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.sample_rate = self.config.get("sample_rate", 5) # Process every Nth frame
        self.low_motion_threshold = self.config.get("low_motion_threshold", 0.5)
        self.high_motion_threshold = self.config.get("high_motion_threshold", 20.0)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            self._analyze_motion(sample)
        except Exception as e:
            logger.error(f"Motion analysis failed: {e}")

        return sample

    def _analyze_motion(self, sample: Sample) -> None:
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return

        prev_gray = None
        flows = []
        diffs = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % self.sample_rate != 0:
                frame_idx += 1
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if prev_gray is not None:
                # 1. Optical Flow (Farneback) - Dense
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                avg_flow = np.mean(mag)
                flows.append(avg_flow)
                
                # 2. Pixel Difference (Flickering/Static check)
                diff = cv2.absdiff(prev_gray, gray)
                avg_diff = np.mean(diff)
                diffs.append(avg_diff)
                
            prev_gray = gray
            frame_idx += 1
            
        cap.release()

        if not flows:
            return

        avg_motion = float(np.mean(flows))
        avg_diff = float(np.mean(diffs))

        # Store motion score in quality metrics
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.motion_score = avg_motion

        # Thresholds (tunable)
        if avg_motion < self.low_motion_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low motion (static/slideshow): {avg_motion:.2f}",
                    details={"avg_flow": float(avg_motion)},
                    recommendation="Remove static clips or slideshows from the training set as they provide little temporal information."
                )
            )
            
        if avg_motion > self.high_motion_threshold:
             sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"High motion: {avg_motion:.2f}",
                    details={"avg_flow": float(avg_motion)},
                    recommendation="Check for camera shake or fast movement. Consider stabilizing or discarding if motion blur is excessive."
                )
            )

        # Flickering detection (high pixel diff but low flow might indicate flicker, 
        # but simplistic flicker is just high variance in luminance without structural change.
        # Here we just use high pixel diff as a proxy for "something changing")
        
        # 3. Effective FPS Check
        self._check_effective_fps(sample)

    def _check_effective_fps(self, sample: Sample):
        """
        Checks a short segment of video for duplicate frames to estimate 'Effective' FPS.
        Useful for detecting upsampled anime/cartoons (e.g. 12fps in 24fps container).
        """
        try:
            cap = cv2.VideoCapture(str(sample.path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps <= 0 or total_frames < 2:
                cap.release()
                return

            # Check 1 second or max 30 frames from middle
            frames_to_check = int(min(fps, 30))
            if frames_to_check < 5: 
                cap.release()
                return
            
            start_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            prev_gray = None
            unique_changes = 0
            frames_read = 0
            
            for _ in range(frames_to_check):
                ret, frame = cap.read()
                if not ret:
                    break
                
                frames_read += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is None:
                    # First frame is always unique
                    unique_changes += 1
                else:
                    # Check difference
                    diff = cv2.absdiff(prev_gray, gray)
                    mean_diff = np.mean(diff)
                    
                    # Threshold for determining if frame is "new" content
                    # 1.0 is very sensitive, 5.0 is robust. 2.0 is a good middle ground.
                    if mean_diff > 2.0:
                        unique_changes += 1
                        
                prev_gray = gray
                
            cap.release()
            
            if frames_read > 0:
                effective_ratio = unique_changes / frames_read
                effective_fps = fps * effective_ratio
                
                # If Effective FPS is significantly lower than Container FPS (e.g. < 70%)
                if effective_ratio < 0.7:
                     sample.validation_issues.append(
                        ValidationIssue(
                            severity=ValidationSeverity.INFO,
                            message=f"Low Effective FPS: ~{effective_fps:.1f} (Container: {fps:.1f})",
                            details={
                                "effective_fps": float(effective_fps), 
                                "container_fps": float(fps),
                                "fps_ratio": float(effective_ratio)
                            },
                            recommendation="Video contains Duplicate Frames (e.g., upsampled 12fps -> 24fps). Consider downsampling to save compute."
                        )
                    )
        except Exception as e:
            logger.warning(f"Effective FPS check failed: {e}")



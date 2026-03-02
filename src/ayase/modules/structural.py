import logging
import cv2
import numpy as np
from typing import List, Tuple

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class StructuralModule(PipelineModule):
    name = "structural"
    description = "Checks structural integrity (scene cuts, black bars)"
    default_config = {"detect_cuts": True, "detect_black_bars": True}

    def __init__(self, config=None):
        super().__init__(config)
        self.detect_cuts = self.config.get("detect_cuts", True)
        self.detect_black_bars = self.config.get("detect_black_bars", True)
        self._scenedetect_available = False

    def setup(self) -> None:
        try:
            import scenedetect  # noqa: F401
            from scenedetect import VideoManager, SceneManager  # noqa: F401
            from scenedetect.detectors import ContentDetector  # noqa: F401

            self._scenedetect_available = True
        except ImportError:
            logger.warning("PySceneDetect not installed. Scene detection disabled.")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        if self.detect_cuts and self._scenedetect_available:
            self._check_scene_cuts(sample)

        if self.detect_black_bars:
            self._check_black_bars(sample)

        return sample

    def _check_scene_cuts(self, sample: Sample) -> None:
        try:
            from scenedetect import VideoManager, SceneManager
            from scenedetect.detectors import ContentDetector

            video_manager = VideoManager([str(sample.path)])
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector())

            video_manager.start()
            scene_manager.detect_scenes(frame_source=video_manager)
            scene_list = scene_manager.get_scene_list()
            video_manager.release()

            if len(scene_list) > 1:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Multiple scenes detected ({len(scene_list)}). Video contains cuts.",
                        details={
                            "scenes": [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
                        },
                        recommendation="Split video into individual scenes using `scenedetect` or `ffmpeg` to avoid continuity errors in training.",
                    )
                )
        except Exception as e:
            logger.error(f"Scene detection failed: {e}")

    def _check_black_bars(self, sample: Sample) -> None:
        # Check a few frames for consistent black bars
        cap = cv2.VideoCapture(str(sample.path))
        if not cap.isOpened():
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        timestamps = [0, 0.5, 0.9]  # Check beginning, middle, end

        total_frames_checked = 0
        letterbox_detected = 0

        for t in timestamps:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_count * t))
            ret, frame = cap.read()
            if not ret:
                continue

            if self._has_black_bars(frame):
                letterbox_detected += 1
            total_frames_checked += 1

        cap.release()

        if total_frames_checked > 0 and letterbox_detected == total_frames_checked:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message="Letterboxing (black bars) detected. Consider cropping.",
                    recommendation="Crop the video to remove black bars using `ffmpeg -vf crop=...` or automated cropping tools.",
                )
            )

    def _has_black_bars(self, frame: np.ndarray, threshold: int = 10) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Check top/bottom
        # If the average intensity of the first 5% or last 5% of rows is very low
        margin_h = int(h * 0.05)
        top_mean = np.mean(gray[:margin_h, :])
        bottom_mean = np.mean(gray[-margin_h:, :])

        if top_mean < threshold and bottom_mean < threshold:
            return True

        return False

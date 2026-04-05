"""Scene cut and shot detection using PySceneDetect ContentDetector.

Counts scene boundaries and flags videos with excessive shot counts.
Useful for identifying multi-scene clips that should be split for training."""

import logging
from typing import List, Tuple

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class SceneModule(PipelineModule):
    name = "scene"
    description = "Detects scene cuts and shots using PySceneDetect"
    default_config = {
        "threshold": 27.0,  # Detection sensitivity
        "min_scene_len": 15,  # Frames
        "warn_on_high_shot_count": True,
        "shot_count_threshold": 3,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.threshold = self.config.get("threshold", 27.0)
        self.min_scene_len = self.config.get("min_scene_len", 15)
        self.warn_on_high_shot_count = self.config.get("warn_on_high_shot_count", True)
        self.shot_count_threshold = self.config.get("shot_count_threshold", 3)
        self._available = False

    def setup(self) -> None:
        try:
            import scenedetect  # noqa: F401
            self._available = True
        except ImportError as e:
            logger.warning(f"PySceneDetect not available: {e}")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._available:
            return sample

        try:
            from scenedetect import ContentDetector, detect

            # Using simple detect function which wraps VideoManager, SceneManager etc.
            scene_list = detect(str(sample.path), ContentDetector(threshold=self.threshold, min_scene_len=self.min_scene_len))
            
            shot_count = len(scene_list)
            if shot_count == 0:
                # If no cuts found, it's one continuous shot (scene_list usually includes 0-end if using scene_manager)
                # but 'detect' return scenes. If scene_list is empty, something might be wrong or it's one shot.
                # Usually scenedetect returns at least one scene.
                shot_count = 1
            
            # Add to issues
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Scene Analysis: {shot_count} shots detected.",
                    details={
                        "shot_count": shot_count,
                        "scenes": [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
                    }
                )
            )

            if self.warn_on_high_shot_count and shot_count > self.shot_count_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High shot count ({shot_count}). Video may contain many cuts.",
                        details={"shot_count": shot_count, "threshold": self.shot_count_threshold},
                        recommendation="Consider splitting the video into individual shots if training for continuous motion."
                    )
                )

        except Exception as e:
            logger.warning(f"Scene detection failed for {sample.path}: {e}")

        return sample

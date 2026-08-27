"""Technical stability of event and shot boundaries in long-form video.

Uses LongAV-Compass-style boundary windows and penalizes black frames, flashes,
duplicate frames, and freezes while allowing ordinary scene cuts. The score is
0-1 and higher is better. No model or optional dependency is required.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class LongFormTransitionStabilityModule(PipelineModule):
    name = "long_form_transition_stability"
    description = "Boundary-local black-frame, flash, duplicate, and freeze stability"
    default_config = {
        "analysis_fps": 8.0,
        "boundary_margin_sec": 2.0,
        "boundaries_sec": [],
        "cut_threshold": 0.25,
        "minimum_boundary_gap_sec": 1.0,
        "max_frames": 2400,
    }
    metric_info = {
        "long_form_transition_stability": (
            "Mean technical stability around event/shot boundaries (0-1, higher=better)"
        )
    }
    metric_groups = {"long_form_transition_stability": "temporal"}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self.analysis_fps = float(self.config.get("analysis_fps", 8.0))
        self.boundary_margin_sec = float(self.config.get("boundary_margin_sec", 2.0))
        self.boundaries_sec = [float(value) for value in self.config.get("boundaries_sec", [])]
        self.cut_threshold = float(self.config.get("cut_threshold", 0.25))
        self.minimum_boundary_gap_sec = float(
            self.config.get("minimum_boundary_gap_sec", 1.0)
        )
        self.max_frames = int(self.config.get("max_frames", 2400))
        self._backend = "algorithmic"

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        try:
            frames = self._sample_video(sample.path)
            if len(frames) < 2:
                return sample
            boundaries = self._boundary_indices(frames)
            if boundaries:
                radius = max(1, round(self.boundary_margin_sec * self.analysis_fps))
                scores = [
                    self._technical_score(frames[max(0, index - radius) : index + radius + 1])
                    for index in boundaries
                ]
                score = float(np.mean(scores))
            else:
                score = 1.0

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.long_form_transition_stability = score
        except Exception as exc:
            logger.warning("Long-form transition stability failed for %s: %s", sample.path, exc)
        return sample

    def _sample_video(self, path: Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or self.analysis_fps)
        step = max(1, round(source_fps / max(self.analysis_fps, 0.1)))
        frames: list[np.ndarray] = []
        index = 0
        try:
            while len(frames) < self.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    height, width = frame.shape[:2]
                    target_width = min(160, width)
                    target_height = max(1, round(height * target_width / max(width, 1)))
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_AREA)
                    frames.append(gray.astype(np.float32) / 255.0)
                index += 1
        finally:
            cap.release()
        return frames

    def _boundary_indices(self, frames: list[np.ndarray]) -> list[int]:
        if self.boundaries_sec:
            return sorted(
                {
                    min(len(frames) - 1, max(1, round(second * self.analysis_fps)))
                    for second in self.boundaries_sec
                }
            )
        diffs = [float(np.mean(np.abs(frames[i] - frames[i - 1]))) for i in range(1, len(frames))]
        candidates = [index + 1 for index, value in enumerate(diffs) if value >= self.cut_threshold]
        minimum_gap = max(1, round(self.minimum_boundary_gap_sec * self.analysis_fps))
        selected: list[int] = []
        for index in candidates:
            if not selected or index - selected[-1] >= minimum_gap:
                selected.append(index)
            elif diffs[index - 1] > diffs[selected[-1] - 1]:
                selected[-1] = index
        return selected

    def _technical_score(self, frames: list[np.ndarray]) -> float:
        if not frames:
            return 0.0
        brightness = [float(frame.mean()) for frame in frames]
        black_ratio = sum(value < 0.05 for value in brightness) / len(brightness)
        diffs = [float(np.mean(np.abs(frames[i] - frames[i - 1]))) for i in range(1, len(frames))]
        duplicate_flags = [value < 0.01 for value in diffs]
        duplicate_ratio = sum(duplicate_flags) / len(duplicate_flags) if duplicate_flags else 0.0
        freeze_seconds = self._max_true_run(duplicate_flags) / max(self.analysis_fps, 0.1)
        flashes = sum(
            abs(brightness[index] - brightness[index - 1]) > 0.45
            for index in range(1, len(brightness))
        )
        penalty = 70.0 * black_ratio + 45.0 * duplicate_ratio + 12.0 * freeze_seconds + 8.0 * flashes
        return float(np.clip(1.0 - penalty / 100.0, 0.0, 1.0))

    @staticmethod
    def _max_true_run(flags: list[bool]) -> int:
        best = current = 0
        for flag in flags:
            current = current + 1 if flag else 0
            best = max(best, current)
        return best

"""Codec-Specific Quality module.

Analyses codec-level attributes of video files:

  codec_efficiency — quality-per-bit efficiency score 0-100
  gop_quality      — GOP structure appropriateness 0-100
  codec_artifacts  — severity of codec-specific artifacts 0-100 (lower=better)

Extracts metadata via ``ffprobe`` (must be on PATH).  Optionally
inspects frame-level statistics for artifact detection.
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Expected bits per pixel per second for reasonable quality per codec
# (rough heuristics for ~1080p content)
CODEC_BPP_TARGETS = {
    "h264": 0.10,
    "h265": 0.06,
    "hevc": 0.06,
    "vp9": 0.06,
    "av1": 0.05,
    "vp8": 0.14,
    "mpeg4": 0.16,
}


class CodecSpecificQualityModule(PipelineModule):
    name = "codec_specific_quality"
    description = "Codec-level efficiency, GOP quality, and artifact detection"
    default_config = {
        "max_frames": 100,  # For artifact detection
        "subsample": 10,
        "warning_efficiency": 30.0,
        "warning_artifacts": 40.0,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.max_frames = self.config.get("max_frames", 100)
        self.subsample = self.config.get("subsample", 10)
        self.warning_efficiency = self.config.get("warning_efficiency", 30.0)
        self.warning_artifacts = self.config.get("warning_artifacts", 40.0)
        self._ffprobe_available = False

    def setup(self) -> None:
        # Check ffprobe availability
        try:
            subprocess.run(
                ["ffprobe", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            self._ffprobe_available = True
            logger.info("Codec quality: ffprobe available")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("ffprobe not found on PATH — codec metrics limited")

    # ------------------------------------------------------------------
    # Metadata extraction
    # ------------------------------------------------------------------

    def _ffprobe_json(self, path: Path, *args: str) -> Optional[dict]:
        """Run ffprobe and return parsed JSON."""
        if not self._ffprobe_available:
            return None
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                *args, str(path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            logger.debug(f"ffprobe failed: {e}")
        return None

    def _get_video_stream(self, path: Path) -> Optional[dict]:
        """Get the first video stream info."""
        info = self._ffprobe_json(path, "-show_streams", "-select_streams", "v:0")
        if info and "streams" in info and info["streams"]:
            return info["streams"][0]
        return None

    def _get_frame_info(self, path: Path) -> Optional[List[dict]]:
        """Get per-frame type info (I/P/B)."""
        info = self._ffprobe_json(
            path,
            "-show_frames", "-select_streams", "v:0",
            "-show_entries", "frame=pict_type,pkt_size,key_frame",
        )
        if info and "frames" in info:
            return info["frames"]
        return None

    # ------------------------------------------------------------------
    # Codec efficiency
    # ------------------------------------------------------------------

    def _compute_efficiency(self, stream: dict) -> float:
        """Compute quality-per-bit efficiency score (0-100).

        Compares actual bits-per-pixel to expected for the codec.
        """
        codec = stream.get("codec_name", "").lower()
        bitrate = stream.get("bit_rate")
        width = stream.get("width")
        height = stream.get("height")
        fps_str = stream.get("r_frame_rate", "30/1")

        if not all([bitrate, width, height]):
            return 50.0  # Unknown — neutral

        try:
            bitrate = int(bitrate)
            width = int(width)
            height = int(height)
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return 50.0

        pixels_per_sec = width * height * fps
        if pixels_per_sec == 0:
            return 50.0

        actual_bpp = bitrate / pixels_per_sec
        target_bpp = CODEC_BPP_TARGETS.get(codec, 0.10)

        # Ratio: < 1 means more efficient than target
        ratio = actual_bpp / target_bpp
        # Score: ratio=1 → 50, ratio=0.5 → 75, ratio=2 → 25
        score = 100.0 - (ratio * 50.0)
        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # GOP quality
    # ------------------------------------------------------------------

    def _compute_gop_quality(self, frames: List[dict]) -> float:
        """Assess GOP structure quality (0-100).

        Checks:
        - Reasonable I-frame spacing (not too far apart)
        - Mix of frame types
        - No excessive consecutive B-frames
        """
        if not frames:
            return 50.0

        types = [f.get("pict_type", "?") for f in frames]
        total = len(types)
        i_count = sum(1 for t in types if t == "I")
        b_count = sum(1 for t in types if t == "B")
        p_count = sum(1 for t in types if t == "P")

        if total == 0:
            return 50.0

        # I-frame ratio: ideally ~1-5% of frames
        i_ratio = i_count / total
        # Too many I-frames (>15%) → inefficient, too few (<0.5%) → fragile
        if i_ratio > 0.15:
            i_score = max(0, 1.0 - (i_ratio - 0.15) * 5)
        elif i_ratio < 0.005:
            i_score = 0.3
        else:
            i_score = 1.0

        # I-frame spacing consistency
        i_positions = [i for i, t in enumerate(types) if t == "I"]
        if len(i_positions) >= 2:
            gaps = np.diff(i_positions)
            gap_cv = float(np.std(gaps) / (np.mean(gaps) + 1e-6))
            spacing_score = max(0, 1.0 - gap_cv)
        else:
            spacing_score = 0.5

        # B-frame streaks: long runs of B-frames can cause quality dips
        max_b_run = 0
        current_b = 0
        for t in types:
            if t == "B":
                current_b += 1
                max_b_run = max(max_b_run, current_b)
            else:
                current_b = 0

        b_run_score = 1.0 if max_b_run <= 4 else max(0, 1.0 - (max_b_run - 4) * 0.1)

        score = (0.4 * i_score + 0.3 * spacing_score + 0.3 * b_run_score) * 100
        return float(np.clip(score, 0, 100))

    # ------------------------------------------------------------------
    # Codec artifacts (block boundary detection)
    # ------------------------------------------------------------------

    def _detect_block_artifacts(self, path: Path) -> float:
        """Detect block-boundary artifacts in video frames.

        Measures periodic discontinuities at 8x8 or 16x16 block
        boundaries (typical of H.264/H.265/MPEG codecs).
        Score 0-100 (lower = fewer artifacts).
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0.0

        block_scores = []
        idx = 0

        while idx < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.subsample == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                score = self._block_boundary_strength(gray)
                block_scores.append(score)
            idx += 1

        cap.release()

        if not block_scores:
            return 0.0

        return float(np.mean(block_scores))

    @staticmethod
    def _block_boundary_strength(gray: np.ndarray, block_size: int = 8) -> float:
        """Measure average discontinuity at block boundaries.

        At every block_size boundary, compute the absolute difference
        between adjacent pixel rows/columns.  In heavily compressed
        video, these boundaries show visible steps.
        """
        h, w = gray.shape

        # Horizontal block boundaries
        h_boundaries = list(range(block_size, h - 1, block_size))
        h_diffs = []
        for y in h_boundaries:
            diff = np.abs(gray[y, :] - gray[y - 1, :])
            h_diffs.append(float(diff.mean()))

        # Vertical block boundaries
        v_boundaries = list(range(block_size, w - 1, block_size))
        v_diffs = []
        for x in v_boundaries:
            diff = np.abs(gray[:, x] - gray[:, x - 1])
            v_diffs.append(float(diff.mean()))

        # Compare boundary diffs to non-boundary diffs (background level)
        all_h = np.abs(np.diff(gray, axis=0)).mean()
        all_v = np.abs(np.diff(gray, axis=1)).mean()

        boundary_avg = np.mean(h_diffs + v_diffs) if (h_diffs or v_diffs) else 0
        background_avg = (all_h + all_v) / 2

        if background_avg < 1e-6:
            return 0.0

        # Ratio of boundary strength to background
        ratio = boundary_avg / background_avg
        # ratio ~1.0 = no extra artifacts, ratio >1.5 = visible blocking
        artifact_score = max(0, (ratio - 1.0) * 200.0)
        return float(np.clip(artifact_score, 0, 100))

    # ------------------------------------------------------------------
    # Process
    # ------------------------------------------------------------------

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            # 1. Codec efficiency (via ffprobe)
            stream = self._get_video_stream(sample.path)
            if stream:
                sample.quality_metrics.codec_efficiency = self._compute_efficiency(stream)

            # 2. GOP quality (via ffprobe frame analysis)
            frame_info = self._get_frame_info(sample.path)
            if frame_info:
                sample.quality_metrics.gop_quality = self._compute_gop_quality(frame_info)

            # 3. Block artifact detection (via OpenCV)
            artifacts = self._detect_block_artifacts(sample.path)
            sample.quality_metrics.codec_artifacts = artifacts

            # Warnings
            eff = sample.quality_metrics.codec_efficiency
            if eff is not None and eff < self.warning_efficiency:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low codec efficiency: {eff:.1f}/100",
                        details={"codec_efficiency": eff},
                        recommendation="Video bitrate may be too high for the codec.",
                    )
                )

            if artifacts > self.warning_artifacts:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Codec block artifacts detected: {artifacts:.1f}/100",
                        details={"codec_artifacts": artifacts},
                        recommendation="Visible block-boundary artifacts from compression.",
                    )
                )

            logger.debug(
                f"Codec quality for {sample.path.name}: "
                f"eff={eff} gop={sample.quality_metrics.gop_quality} "
                f"artifacts={artifacts:.1f}"
            )

        except Exception as e:
            logger.error(f"Codec quality failed for {sample.path}: {e}")

        return sample

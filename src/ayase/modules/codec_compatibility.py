"""Codec / container compatibility checker.

Validates that video files use codecs, pixel formats, and container formats
compatible with standard ML dataloaders (decord, torchvision, OpenCV).

Based on filtering criteria from Stable Video Diffusion, VideoCrafter2, and
Omni-Video 2 data curation pipelines.
"""

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Codecs that standard ML dataloaders (decord, torchvision, opencv) can decode
SAFE_VIDEO_CODECS = {
    "h264", "h265", "hevc", "vp8", "vp9", "av1",
    "mpeg4", "mpeg2video", "mjpeg", "png", "rawvideo",
}

# Pixel formats that are widely supported
SAFE_PIXEL_FORMATS = {
    "yuv420p", "yuv422p", "yuv444p", "yuvj420p", "yuvj422p",
    "rgb24", "bgr24", "nv12", "nv21",
}

# Containers that work reliably
SAFE_CONTAINERS = {
    "mp4", "mkv", "avi", "webm", "mov", "flv", "ts",
}


class CodecCompatibilityModule(PipelineModule):
    name = "codec_compatibility"
    description = "Validates codec, pixel format, and container for ML dataloader compatibility"
    default_config = {
        "min_bitrate_kbps": 500,     # Minimum video bitrate (kbps)
        "min_bpp": 0.02,             # Minimum bits-per-pixel (bitrate quality ratio)
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.min_bitrate_kbps = self.config.get("min_bitrate_kbps", 500)
        self.min_bpp = self.config.get("min_bpp", 0.02)
        self._ffprobe = shutil.which("ffprobe")

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        if not self._ffprobe:
            logger.debug("ffprobe not found; skipping codec compatibility check")
            return sample

        try:
            info = self._probe(sample.path)
            if info is None:
                return sample

            self._check_codec(sample, info)
            self._check_pixel_format(sample, info)
            self._check_container(sample, info)
            self._check_bitrate_quality(sample, info)

        except Exception as e:
            logger.warning(f"Codec compatibility check failed for {sample.path}: {e}")

        return sample

    # ------------------------------------------------------------------
    def _probe(self, path: Path) -> Optional[Dict]:
        cmd = [
            self._ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)

    def _video_stream(self, info: Dict) -> Optional[Dict]:
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                return s
        return None

    # ------------------------------------------------------------------
    def _check_codec(self, sample: Sample, info: Dict) -> None:
        vs = self._video_stream(info)
        if not vs:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message="No video stream found in container",
                )
            )
            return

        codec = vs.get("codec_name", "unknown").lower()
        if codec not in SAFE_VIDEO_CODECS:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Uncommon video codec '{codec}' may fail in ML dataloaders",
                    details={"codec": codec, "safe_codecs": sorted(SAFE_VIDEO_CODECS)},
                    recommendation=f"Re-encode to H.264: ffmpeg -i input -c:v libx264 output.mp4",
                )
            )

    def _check_pixel_format(self, sample: Sample, info: Dict) -> None:
        vs = self._video_stream(info)
        if not vs:
            return
        pix_fmt = vs.get("pix_fmt", "unknown").lower()
        if pix_fmt not in SAFE_PIXEL_FORMATS:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Pixel format '{pix_fmt}' may not be decoded correctly",
                    details={"pix_fmt": pix_fmt},
                    recommendation=f"Re-encode with yuv420p: ffmpeg -i input -pix_fmt yuv420p output.mp4",
                )
            )

    def _check_container(self, sample: Sample, info: Dict) -> None:
        fmt = info.get("format", {}).get("format_name", "")
        # format_name can be comma-separated (e.g. "mov,mp4,m4a,3gp,3g2,mj2")
        known = any(f.strip() in SAFE_CONTAINERS for f in fmt.split(","))
        if not known:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Uncommon container format '{fmt}'",
                    details={"format_name": fmt},
                )
            )

    def _check_bitrate_quality(self, sample: Sample, info: Dict) -> None:
        vs = self._video_stream(info)
        if not vs:
            return

        # Try stream-level bitrate, fall back to container-level
        bitrate_str = vs.get("bit_rate") or info.get("format", {}).get("bit_rate")
        if not bitrate_str:
            return

        try:
            bitrate = int(bitrate_str)
        except (ValueError, TypeError):
            return

        bitrate_kbps = bitrate / 1000

        if bitrate_kbps < self.min_bitrate_kbps:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Very low video bitrate ({bitrate_kbps:.0f} kbps)",
                    details={"bitrate_kbps": bitrate_kbps},
                    recommendation="Over-compressed video degrades training quality. Source a higher-bitrate copy.",
                )
            )

        # Bits-per-pixel: bitrate / (width * height * fps)
        width = int(vs.get("width", 0))
        height = int(vs.get("height", 0))
        fps_parts = vs.get("r_frame_rate", "0/1").split("/")
        try:
            fps = float(fps_parts[0]) / float(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
        except (ValueError, ZeroDivisionError):
            fps = 0

        if width > 0 and height > 0 and fps > 0:
            bpp = bitrate / (width * height * fps)
            if bpp < self.min_bpp:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low bits-per-pixel ({bpp:.4f}). Heavy compression for this resolution.",
                        details={"bpp": float(bpp), "resolution": f"{width}x{height}", "fps": fps},
                        recommendation="Video is over-compressed relative to its resolution. "
                                       "Re-encode at a higher CRF or discard.",
                    )
                )

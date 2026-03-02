"""Video processing utilities for Ayase."""

import logging
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def split_video_by_scenes(
    video_path: Path, scene_changes: List[float], output_dir: Path
) -> List[Path]:
    """Split a video into multiple segments based on scene changes.

    Args:
        video_path: Path to the source video
        scene_changes: List of timestamps (seconds) where cuts occur
        output_dir: Directory to save the segments

    Returns:
        List of paths to the created video segments
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not scene_changes:
        logger.info(f"No scene changes provided for {video_path}, skipping split.")
        return [video_path]

    output_dir.mkdir(parents=True, exist_ok=True)

    # Add start and end timestamps for the split intervals
    # We assume the video starts at 0.0
    # To get the end, we'd ideally need the duration, but ffmpeg handles "to end"
    intervals = [0.0] + sorted(scene_changes)

    segment_paths = []

    for i in range(len(intervals)):
        start = intervals[i]
        end = intervals[i + 1] if i + 1 < len(intervals) else None

        segment_name = f"{video_path.stem}_scene_{i+1}{video_path.suffix}"
        segment_path = output_dir / segment_name

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(video_path),
        ]

        if end:
            duration = end - start
            cmd.extend(["-t", str(duration)])

        # Use -c copy for fast stream copying without re-encoding
        cmd.extend(["-c", "copy", str(segment_path)])

        try:
            subprocess.run(cmd, capture_output=True, check=True)
            segment_paths.append(segment_path)
            logger.debug(f"Created segment: {segment_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error splitting video at {start}: {e.stderr.decode()}")

    return segment_paths

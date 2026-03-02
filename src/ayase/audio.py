"""Audio utility functions for Ayase."""

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from .models import AudioMetadata

logger = logging.getLogger(__name__)


def get_audio_metadata(path: Path) -> Optional[AudioMetadata]:
    """Get metadata for the audio stream in a file.

    Args:
        path: Path to the media file

    Returns:
        AudioMetadata object or None if no audio or error
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "a",
            str(path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        if not data.get("streams"):
            return None

        stream = data["streams"][0]

        return AudioMetadata(
            sample_rate=int(stream.get("sample_rate", 0)),
            channels=int(stream.get("channels", 0)),
            bitrate=int(stream.get("bit_rate")) if stream.get("bit_rate") else None,
            codec=stream.get("codec_name", "unknown"),
            duration=float(stream.get("duration", 0.0)),
        )
    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
    ) as e:
        logger.debug(f"Could not get audio metadata for {path}: {e}")
        return None


def has_audio(path: Path) -> bool:
    """Check if a media file has an audio stream.

    Args:
        path: Path to the media file

    Returns:
        True if the file has audio, False otherwise
    """
    return get_audio_metadata(path) is not None

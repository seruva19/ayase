"""ffmpeg invocation with argument lists and checked return codes."""

import logging
import subprocess
from typing import List

logger = logging.getLogger(__name__)

FFMPEG_BINARY = "ffmpeg"
# Only errors reach stderr, so an exception carries the cause rather than the build banner.
_QUIET_ARGS = ("-hide_banner", "-loglevel", "error")
FFMPEG_TIMEOUT_SECONDS = 3600
_STDERR_TAIL_CHARS = 500


def run_ffmpeg(args: List[str]) -> None:
    """Run ffmpeg with ``args`` (an argument list, no shell).

    Raises ``RuntimeError`` when the binary is missing, exits with a non-zero code or
    exceeds ``FFMPEG_TIMEOUT_SECONDS``.
    """
    command = [FFMPEG_BINARY, *_QUIET_ARGS, *args]
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=FFMPEG_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found in PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffmpeg did not finish within {FFMPEG_TIMEOUT_SECONDS} s") from exc
    if done.returncode != 0:
        stderr_tail = done.stderr.strip()[-_STDERR_TAIL_CHARS:]
        raise RuntimeError(f"ffmpeg exited with code {done.returncode}: {stderr_tail}")

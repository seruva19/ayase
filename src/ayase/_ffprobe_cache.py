"""Bounded cache for successful, equivalent ffprobe invocations."""

import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union


def _file_identity(path: Path) -> Tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


@lru_cache(maxsize=128)
def _cached_stdout(
    command: Tuple[str, ...],
    timeout: Optional[float],
    text: bool,
    working_dir: str,
    tool_identity: Tuple[int, ...],
    input_identity: Tuple[int, ...],
) -> Union[str, bytes]:
    return subprocess.run(
        command, capture_output=True, text=text, timeout=timeout, check=True
    ).stdout


def ffprobe_stdout(
    command: Sequence[str], *, timeout: Optional[float] = None, text: bool = False
) -> Union[str, bytes]:
    """Reuse stdout only when executable, file, and entire argument list match."""
    executable = Path(command[0])
    try:
        if not executable.is_absolute():
            raise OSError("ffprobe executable is resolved through PATH")
        working_dir = os.getcwd()
        tool_identity = _file_identity(executable)
        input_identity = _file_identity(Path(command[-1]))
    except OSError:
        return subprocess.run(
            command, capture_output=True, text=text, timeout=timeout, check=True
        ).stdout
    return _cached_stdout(tuple(command), timeout, text, working_dir, tool_identity, input_identity)

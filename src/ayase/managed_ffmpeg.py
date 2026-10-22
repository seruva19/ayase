"""Resolve a private FFmpeg executable with the features Ayase requires."""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

Pathish = Union[str, os.PathLike[str]]


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


@lru_cache(maxsize=16)
def ffmpeg_has_filter(executable: str, filter_name: str = "libvmaf") -> bool:
    """Return whether an FFmpeg executable advertises ``filter_name``."""

    try:
        result = subprocess.run(
            [executable, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            creationflags=_creation_flags(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    output = result.stdout + "\n" + result.stderr
    return result.returncode == 0 and any(
        line.split()[1:2] == [filter_name] for line in output.splitlines() if line.split()
    )


def _candidate(value: Optional[Pathish]) -> Optional[str]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_file():
        return str(path.resolve())
    return shutil.which(os.fspath(value))


def _test_mode_enabled() -> bool:
    """Return whether provisioning is disabled for lightweight inspection."""

    if os.environ.get("AYASE_TEST_MODE", "") == "1":
        return True
    try:
        from .pipeline import PipelineModule

        return bool(PipelineModule._global_test_mode)
    except ImportError:
        return False


def resolve_ffmpeg(
    configured: Optional[Pathish] = None,
    *,
    required_filter: str = "libvmaf",
) -> Path:
    """Resolve FFmpeg without changing PATH, preferring user and system installs.

    ``imageio-ffmpeg>=0.6.0`` supplies maintained Windows and Linux x86-64
    binaries built with libvmaf. It is used only when an existing FFmpeg does
    not expose the requested filter.
    """

    return resolve_media_tool("ffmpeg", required_filter=required_filter, configured=configured)


def _native_tool(name: str, models_dir: Pathish) -> Path:
    from .managed_native import ensure_managed_native_environment

    interpreter = ensure_managed_native_environment(
        "media-tools",
        ("ffmpeg=9.0.2",),
        models_dir,
        python_version="3.11",
    )
    prefix = interpreter.parent if os.name == "nt" else interpreter.parent.parent
    directory = prefix / "Library" / "bin" if os.name == "nt" else prefix / "bin"
    suffix = ".exe" if os.name == "nt" else ""
    executable = directory / f"{name}{suffix}"
    if not executable.is_file():
        raise RuntimeError(f"Managed FFmpeg package does not provide {name}: {executable}")
    return executable.resolve()


def resolve_media_tool(
    name: str,
    required_filter: Optional[str] = None,
    *,
    configured: Optional[Pathish] = None,
    models_dir: Pathish = "models",
) -> Path:
    """Resolve FFmpeg-family tools privately without modifying the host PATH."""

    if name not in {"ffmpeg", "ffprobe"}:
        raise ValueError(f"Unsupported media tool: {name}")

    candidates = [_candidate(configured), shutil.which(name)]
    for executable in candidates:
        if executable and (
            required_filter is None or ffmpeg_has_filter(executable, required_filter)
        ):
            return Path(executable).resolve()

    if name == "ffmpeg":
        try:
            import imageio_ffmpeg

            packaged = Path(imageio_ffmpeg.get_ffmpeg_exe()).expanduser().resolve()
        except (ImportError, OSError, RuntimeError):
            packaged = None
        if (
            packaged is not None
            and packaged.is_file()
            and (required_filter is None or ffmpeg_has_filter(str(packaged), required_filter))
        ):
            return packaged

    if _test_mode_enabled():
        raise RuntimeError(
            f"{name} is unavailable in Ayase test mode; native media-tool "
            "provisioning is disabled"
        )

    managed = _native_tool(name, models_dir)
    if required_filter is not None and not ffmpeg_has_filter(str(managed), required_filter):
        raise RuntimeError(
            f"Managed FFmpeg does not provide the {required_filter} filter: {managed}"
        )
    return managed


__all__ = ["ffmpeg_has_filter", "resolve_ffmpeg", "resolve_media_tool"]

"""Resolve canonical metric executables from private conda environments."""

from __future__ import annotations

import hashlib
import json
import lzma
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Optional, Sequence, Union

Pathish = Union[str, os.PathLike[str]]


def resolve_native_tool(
    name: str,
    packages: Sequence[str],
    models_dir: Pathish = "models",
    *,
    configured: Optional[Pathish] = None,
) -> Path:
    """Prefer an explicit/system executable, then provision a private copy."""

    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        found = shutil.which(os.fspath(configured))
        if found:
            return Path(found).resolve()
    found = shutil.which(name)
    if found:
        return Path(found).resolve()

    from .managed_native import ensure_managed_native_environment

    environment_key = hashlib.sha256(json.dumps(list(packages)).encode("utf-8")).hexdigest()[:12]
    python = ensure_managed_native_environment(
        "tools-" + environment_key, packages, models_dir, python_version="3.11"
    )
    prefix = python.parent if os.name == "nt" else python.parent.parent
    tool_dir = prefix / "Library" / "bin" if os.name == "nt" else prefix / "bin"
    executable = tool_dir / (name + (".exe" if os.name == "nt" else ""))
    if not executable.is_file():
        raise RuntimeError(f"Managed packages {tuple(packages)!r} do not provide {name}")
    return executable.resolve()


_LIBJXL_WINDOWS_URL = (
    "https://github.com/libjxl/libjxl/releases/download/v0.12.0/" "jxl-x64-windows-static.zip"
)
_LIBJXL_WINDOWS_SHA256 = "3025d7e308390796d20492322e606bc92decaee7b6bc99d3f7547870ae5db7de"
_LIBJXL_LINUX_URL = (
    "https://github.com/libjxl/libjxl/releases/download/v0.12.0/" "jxl-linux-x86_64-static.tar.lz"
)
_LIBJXL_LINUX_SHA256 = "5318a1ea40adad76d023e0c17a03d4627f8282f83cb2b575e69be8e74f1ff456"


def _decode_lzip(member: bytes) -> bytes:
    """Decode and validate all concatenated members in a lzip archive."""

    output = bytearray()
    offset = 0
    while offset < len(member):
        if len(member) - offset < 26 or member[offset : offset + 5] != b"LZIP\x01":
            raise RuntimeError("invalid lzip header")
        code = member[offset + 5]
        if code > 40:
            raise RuntimeError("unsupported lzip dictionary code")
        dictionary_size = 1 << (code & 0x1F)
        dictionary_size -= (dictionary_size // 16) * ((code >> 5) & 7)
        decoder = lzma.LZMADecompressor(
            format=lzma.FORMAT_RAW,
            filters=[
                {
                    "id": lzma.FILTER_LZMA1,
                    "dict_size": dictionary_size,
                    "lc": 3,
                    "lp": 0,
                    "pb": 2,
                }
            ],
        )
        payload = member[offset + 6 :]
        try:
            decoded = decoder.decompress(payload)
        except lzma.LZMAError as exc:
            raise RuntimeError("invalid lzip payload") from exc
        compressed_size = len(payload) - len(decoder.unused_data)
        trailer_offset = offset + 6 + compressed_size
        if trailer_offset + 20 > len(member):
            raise RuntimeError("truncated lzip trailer")
        expected_crc, expected_size, expected_member_size = struct.unpack(
            "<IQQ", member[trailer_offset : trailer_offset + 20]
        )
        actual_member_size = trailer_offset + 20 - offset
        if zlib.crc32(decoded) != expected_crc:
            raise RuntimeError("lzip data checksum mismatch")
        if len(decoded) != expected_size or actual_member_size != expected_member_size:
            raise RuntimeError("lzip trailer size mismatch")
        output.extend(decoded)
        offset += actual_member_size
    return bytes(output)


def resolve_butteraugli(
    models_dir: Pathish = "models", configured: Optional[Pathish] = None
) -> Path:
    """Resolve official ``butteraugli_main`` with a verified Windows release fallback."""

    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file():
            return candidate.resolve()
    found = shutil.which(os.fspath(configured) if configured else "butteraugli_main")
    if found:
        return Path(found).resolve()
    root = Path(models_dir).expanduser().resolve() / ".ayase" / "tools" / "libjxl-0.12.0"
    windows = os.name == "nt"
    relative_executable = Path(
        "x64-windows-static/bin/butteraugli_main.exe" if windows else "tools/butteraugli_main"
    )
    executable = root / relative_executable
    if executable.is_file():
        return executable
    root.parent.mkdir(parents=True, exist_ok=True)
    from .managed_runtime import _SetupLock

    with _SetupLock(root.parent / "libjxl-0.12.0.lock"):
        if executable.is_file():
            return executable
        with tempfile.TemporaryDirectory(dir=root.parent) as temporary_name:
            temporary = Path(temporary_name)
            archive = temporary / ("libjxl.zip" if windows else "libjxl.tar.lz")
            url = _LIBJXL_WINDOWS_URL if windows else _LIBJXL_LINUX_URL
            expected_sha256 = _LIBJXL_WINDOWS_SHA256 if windows else _LIBJXL_LINUX_SHA256
            with urllib.request.urlopen(url, timeout=120) as response:
                with archive.open("wb") as output:
                    shutil.copyfileobj(response, output)
            actual = hashlib.sha256(archive.read_bytes()).hexdigest()
            if actual != expected_sha256:
                raise RuntimeError(f"Official libjxl archive checksum mismatch: {actual}")
            extracted = temporary / "extracted"
            extracted.mkdir()
            resolved_root = extracted.resolve()
            if windows:
                with zipfile.ZipFile(archive) as bundle:
                    names = bundle.namelist()
                    for name in names:
                        destination = (extracted / name).resolve()
                        if (
                            destination != resolved_root
                            and resolved_root not in destination.parents
                        ):
                            raise RuntimeError("unsafe path in official libjxl archive")
                extractor = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import sys,zipfile_deflate64 as z; "
                        "z.ZipFile(sys.argv[1]).extractall(sys.argv[2])",
                        str(archive),
                        str(extracted),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=180,
                    check=False,
                    creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                )
                if extractor.returncode != 0:
                    raise RuntimeError(
                        "official libjxl ZIP extraction failed: "
                        + (extractor.stderr.strip() or extractor.stdout.strip())
                    )
            else:
                tar_path = temporary / "libjxl.tar"
                tar_path.write_bytes(_decode_lzip(archive.read_bytes()))
                with tarfile.open(tar_path) as bundle:
                    for member in bundle.getmembers():
                        destination = (extracted / member.name).resolve()
                        if (
                            destination != resolved_root
                            and resolved_root not in destination.parents
                        ):
                            raise RuntimeError("unsafe path in official libjxl archive")
                        if member.issym() or member.islnk():
                            raise RuntimeError("links are not allowed in official libjxl archive")
                    bundle.extractall(extracted)
            extracted_executable = extracted / relative_executable
            if not extracted_executable.is_file():
                raise RuntimeError("Official libjxl archive lacks butteraugli_main")
            if not windows:
                extracted_executable.chmod(0o755)
            if root.exists():
                shutil.rmtree(root)
            extracted.replace(root)
    return executable


__all__ = ["resolve_butteraugli", "resolve_native_tool"]

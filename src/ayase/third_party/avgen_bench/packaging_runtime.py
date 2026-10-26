"""Build old pure-Python AVGen dependencies with compatible private build tools.

Source archives are the original digest-pinned PyPI distributions. A private
setuptools version retains MoviePy 1.0.3's legacy test command without changing
its runtime code or requiring a compiler or user-prepared environment.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Dict, Mapping, Union

_BUILD_REQUIREMENTS = ("setuptools==70.3.0", "wheel==0.45.1")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(artifact: Mapping[str, Any], destination: Path) -> None:
    size = 0
    with urllib.request.urlopen(artifact["url"], timeout=120) as response:
        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > artifact["bytes"]:
                    raise ValueError("AVGen dependency archive exceeds its pinned size")
                output.write(chunk)
    if size != artifact["bytes"] or _sha256(destination) != artifact["sha256"]:
        raise ValueError("AVGen dependency archive failed SHA-256 verification")


def ensure_package_wheel(
    name: str,
    artifact: Mapping[str, Any],
    models_dir: Union[str, os.PathLike],
    *,
    test_mode: bool = False,
) -> Path:
    """Acquire or build one pinned package, without modifying its source code."""
    if test_mode:
        raise RuntimeError("AVGen dependency preparation is disabled in test mode")
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for character in name):
        raise ValueError("invalid AVGen dependency name")
    filename = artifact["filename"]
    if Path(filename).name != filename or "/" in filename or "\\" in filename:
        raise ValueError("invalid AVGen dependency archive filename")
    from ayase.managed_runtime import _SetupLock, ensure_managed_environment

    root = Path(models_dir).expanduser().resolve() / ".ayase" / "avgen-bench" / "wheels"
    identity = hashlib.sha256(
        json.dumps(
            {"artifact": dict(artifact), "build_tools": _BUILD_REQUIREMENTS}, sort_keys=True
        ).encode()
    ).hexdigest()[:20]
    cache = root / (name + "-" + identity)
    with _SetupLock(root / (name + "-" + identity + ".lock")):
        marker = cache / "manifest.json"
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if (
                not isinstance(saved, dict)
                or saved.get("source_sha256") != artifact["sha256"]
                or saved.get("source_url") != artifact["url"]
                or saved.get("build_tools") != list(_BUILD_REQUIREMENTS)
            ):
                raise ValueError("AVGen cached dependency source identity is invalid")
            wheel_name = saved.get("filename")
            if (
                not isinstance(wheel_name, str)
                or Path(wheel_name).name != wheel_name
                or any(character in wheel_name for character in "/\\:")
                or not wheel_name.endswith(".whl")
                or cache.is_symlink()
                or cache.resolve().parent != root.resolve()
            ):
                raise ValueError("AVGen cached dependency wheel path is invalid")
            wheel = cache / wheel_name
            if wheel.is_symlink() or not wheel.is_file():
                raise ValueError("AVGen cached dependency wheel is missing or linked")
            actual_sha = _sha256(wheel)
            if actual_sha != saved.get("sha256"):
                raise ValueError("AVGen cached dependency wheel is damaged")
            if artifact["kind"] == "bdist_wheel" and (
                wheel_name != artifact["filename"]
                or wheel.stat().st_size != artifact["bytes"]
                or actual_sha != artifact["sha256"]
            ):
                raise ValueError("AVGen cached dependency wheel differs from its publisher artifact")
            return wheel
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=name + "-", dir=root) as temporary:
            work = Path(temporary)
            archive = work / filename
            _download(artifact, archive)
            if artifact["kind"] == "bdist_wheel":
                wheel = archive
            else:
                python = ensure_managed_environment(
                    "avgen-build-tools", _BUILD_REQUIREMENTS, models_dir, python_version="3.10.19"
                )
                output = work / "output"
                output.mkdir()
                log_path = work / "build.log"
                environment = os.environ.copy()
                environment["TEMP"] = str(work)
                environment["TMP"] = str(work)
                environment["PYTHONUTF8"] = "1"
                environment["PYTHONDONTWRITEBYTECODE"] = "1"
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(
                        [
                            str(python),
                            "-m",
                            "pip",
                            "wheel",
                            "--no-build-isolation",
                            "--no-deps",
                            "--wheel-dir",
                            str(output),
                            str(archive),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        timeout=600,
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                    )
                if result.returncode != 0:
                    logs = root / "logs"
                    logs.mkdir(exist_ok=True)
                    retained = logs / (name + "-" + identity + ".log")
                    retained.write_bytes(log_path.read_bytes())
                    raise RuntimeError(f"AVGen dependency wheel build failed; see {retained}")
                wheels = list(output.glob("*.whl"))
                if len(wheels) != 1:
                    raise ValueError("AVGen dependency build returned no unique wheel")
                wheel = wheels[0]
            staged = work / "ready"
            staged.mkdir()
            target = staged / wheel.name
            target.write_bytes(wheel.read_bytes())
            saved: Dict[str, Any] = {
                "filename": wheel.name,
                "sha256": _sha256(wheel),
                "source_sha256": artifact["sha256"],
                "source_url": artifact["url"],
                "build_tools": list(_BUILD_REQUIREMENTS),
            }
            (staged / "manifest.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
            staged.replace(cache)
            return cache / wheel.name

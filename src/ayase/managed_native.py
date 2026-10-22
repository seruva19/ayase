"""Private micromamba environments for metrics with published native binaries."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

from ayase.managed_runtime import _SetupLock

_MICROMAMBA_VERSION = "2.9.0"
_MICROMAMBA_BUILDS: Dict[str, Tuple[str, str]] = {
    "win-64": (
        "https://api.anaconda.org/download/conda-forge/micromamba/2.9.0/win-64/"
        "micromamba-2.9.0-0.tar.bz2",
        "97a336f4ab794bd96a6a4da5e6ed63e75a1d31830414a182419b23d3b36f3fe0",
    ),
    "linux-64": (
        "https://api.anaconda.org/download/conda-forge/micromamba/2.9.0/linux-64/"
        "micromamba-2.9.0-0.tar.bz2",
        "8761c382127e6363bd9e0a2451aa3ef90d071a79133f736e2f759a3bf13040dd",
    ),
}


def _platform_subdir() -> str:
    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64"}:
        raise RuntimeError(f"managed native environments do not support {machine!r}")
    if os.name == "nt":
        return "win-64"
    if platform.system() == "Linux":
        return "linux-64"
    raise RuntimeError(f"managed native environments do not support {platform.system()!r}")


def _python(environment: Path) -> Path:
    return environment / ("python.exe" if os.name == "nt" else "bin/python")


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _log_tail(path: Path, lines: int = 40) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return "installation log could not be read"


def _micromamba(models_dir: Path) -> Path:
    subdir = _platform_subdir()
    filename = "micromamba.exe" if subdir == "win-64" else "micromamba"
    target = models_dir / ".ayase" / "tools" / f"micromamba-{_MICROMAMBA_VERSION}" / filename
    if target.is_file():
        return target
    url, expected = _MICROMAMBA_BUILDS[subdir]
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ayase-micromamba-") as directory:
        archive = Path(directory) / "micromamba.tar.bz2"
        _download(url, archive)
        actual = hashlib.sha256(archive.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError("downloaded micromamba archive failed SHA-256 verification")
        member_name = f"Library/bin/{filename}" if subdir == "win-64" else f"bin/{filename}"
        with tarfile.open(archive, "r:bz2") as bundle:
            try:
                member = bundle.getmember(member_name)
            except KeyError as exc:
                raise RuntimeError("micromamba archive does not contain its executable") from exc
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError("micromamba executable is not a regular archive member")
            temporary = target.with_suffix(target.suffix + f".tmp-{uuid.uuid4().hex}")
            with source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output)
            if os.name != "nt":
                temporary.chmod(0o755)
            temporary.replace(target)
    return target


def ensure_managed_native_environment(
    name: str,
    conda_packages: Sequence[str],
    models_dir: Union[os.PathLike[str], str] = "models",
    *,
    python_version: str = "3.11",
    pip_requirements: Sequence[str] = (),
    pip_options: Sequence[str] = (),
) -> Path:
    """Return a Python executable from a cached conda-forge binary environment."""

    if not name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-" for character in name.lower()
    ):
        raise ValueError("managed native environment name contains unsupported characters")
    if not conda_packages:
        raise ValueError("managed native environment requires at least one package")
    root = Path(models_dir).expanduser().resolve() / ".ayase"
    mamba_root = root / "mamba"
    package_cache = root / "conda-packages"
    logs = root / "logs"
    payload = {
        "schema": 1,
        "platform": _platform_subdir(),
        "python": python_version,
        "packages": list(conda_packages),
        "pip_requirements": list(pip_requirements),
        "pip_options": list(pip_options),
        "micromamba": _MICROMAMBA_VERSION,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    environment = root / "environments" / f"native-{name}-{fingerprint[:16]}"
    manifest = environment / "ayase-native.json"
    lock = root / "locks" / f"native-{name}-{fingerprint}.lock"
    with _SetupLock(lock):
        interpreter = _python(environment)
        if interpreter.is_file() and manifest.is_file():
            try:
                if json.loads(manifest.read_text(encoding="utf-8")) == payload:
                    return interpreter
            except (OSError, json.JSONDecodeError):
                pass
        executable = _micromamba(Path(models_dir).expanduser().resolve())
        if environment.exists():
            shutil.rmtree(environment)
        environment.parent.mkdir(parents=True, exist_ok=True)
        mamba_root.mkdir(parents=True, exist_ok=True)
        package_cache.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        install_log = logs / f"native-{name}-{fingerprint[:16]}.log"
        process_environment = os.environ.copy()
        process_environment.update(
            {
                "MAMBA_ROOT_PREFIX": str(mamba_root),
                "CONDA_PKGS_DIRS": str(package_cache),
            }
        )
        command = [
            str(executable),
            "create",
            "--no-rc",
            "--root-prefix",
            str(mamba_root),
            "--yes",
            "--strict-channel-priority",
            "--channel",
            "conda-forge",
            "--prefix",
            str(environment),
            f"python={python_version}",
            *conda_packages,
        ]
        with install_log.open("w", encoding="utf-8", errors="replace") as output:
            result = subprocess.run(
                command,
                check=False,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=process_environment,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        if result.returncode != 0:
            shutil.rmtree(environment, ignore_errors=True)
            tail = _log_tail(install_log)
            raise RuntimeError(f"managed native environment installation failed:\n{tail}")
        interpreter = _python(environment)
        if pip_requirements:
            with install_log.open("a", encoding="utf-8", errors="replace") as output:
                pip_result = subprocess.run(
                    [str(interpreter), "-m", "pip", "install", *pip_options, *pip_requirements],
                    check=False,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=process_environment,
                    creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                )
            if pip_result.returncode != 0:
                shutil.rmtree(environment, ignore_errors=True)
                tail = _log_tail(install_log)
                raise RuntimeError(f"managed native pip installation failed:\n{tail}")
        temporary_manifest = manifest.with_suffix(f".tmp-{uuid.uuid4().hex}")
        temporary_manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary_manifest.replace(manifest)
        return interpreter


def managed_native_executable(interpreter: Union[os.PathLike[str], str], name: str) -> Path:
    """Resolve an executable installed beside a managed native interpreter."""

    python = Path(interpreter).resolve()
    root = python.parent if os.name == "nt" else python.parent.parent
    candidates = (
        (root / "Library" / "bin" / f"{name}.exe", root / "Scripts" / f"{name}.exe")
        if os.name == "nt"
        else (root / "bin" / name,)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"managed native executable {name!r} is missing under {root}")


__all__ = ["ensure_managed_native_environment", "managed_native_executable"]

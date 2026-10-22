"""Provision isolated, cached Python environments for isolated metric runtimes.

Callers pass pinned requirement strings or local package directories. ``source_roots``
are fingerprint inputs for source that is imported or installed indirectly: Git roots
use HEAD plus a source-file digest, while other roots use that digest alone. Model
weights, datasets, caches, logs, and generated outputs are deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

Pathish = Union[str, os.PathLike[str]]
logger = logging.getLogger(__name__)

_LOCK_TIMEOUT_SECONDS = 1800.0
_LOCK_POLL_SECONDS = 0.1
_STALE_LOCK_SECONDS = 21600.0
_LOG_TAIL_LINES = 40
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _run(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creation_flags(),
        env=env,
    )


def _run_logged(command: Sequence[str], log_path: Path) -> int:
    """Run a potentially verbose command while streaming all output to disk."""

    with log_path.open("a", encoding="utf-8") as stream:
        process = subprocess.Popen(
            list(command),
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=_creation_flags(),
        )
        return process.wait()


def _python_identity(python: Path) -> Dict[str, str]:
    probe = _run(
        [
            str(python),
            "-c",
            "import json,platform,sys; print(json.dumps({"
            "'version': platform.python_version(), 'implementation': platform.python_implementation(),"
            "'executable': sys.executable, 'platform': platform.platform()}))",
        ]
    )
    if probe.returncode != 0:
        raise RuntimeError(f"Unable to inspect Python interpreter {python}: {probe.stdout.strip()}")
    try:
        value = json.loads(probe.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid response from Python interpreter {python}") from exc
    return {str(key): str(item) for key, item in value.items()}


def _uv_environment(cache_root: Path) -> Dict[str, str]:
    environment = os.environ.copy()
    environment["UV_CACHE_DIR"] = str(cache_root / "uv-cache")
    environment["UV_PYTHON_INSTALL_DIR"] = str(cache_root / "python")
    return environment


def _resolve_managed_python(version: str, cache_root: Path) -> Path:
    """Acquire and resolve a uv-managed Python without modifying ``PATH``."""

    # Requested versions are downloaded on demand and --managed-python excludes
    # unrelated system installs: https://docs.astral.sh/uv/concepts/python-versions/
    logger.info("Resolving Ayase-managed Python %s under %s", version, cache_root)
    uv = [str(Path(sys.executable).resolve()), "-m", "uv"]
    environment = _uv_environment(cache_root)
    installed = _run(
        [
            *uv,
            "python",
            "install",
            version,
            "--managed-python",
            "--no-config",
            "--no-progress",
            "--no-bin",
            "--no-registry",
        ],
        env=environment,
    )
    if installed.returncode != 0:
        raise RuntimeError(
            f"Unable to acquire managed Python {version!r} with uv:\n{_tail(installed.stdout)}"
        )
    found = _run(
        [*uv, "python", "find", version, "--managed-python", "--no-config"],
        env=environment,
    )
    if found.returncode != 0 or not found.stdout.strip():
        raise RuntimeError(
            f"Unable to resolve managed Python {version!r} with uv:\n{_tail(found.stdout)}"
        )
    return Path(found.stdout.strip().splitlines()[-1]).resolve()


def _iter_source_files(root: Path) -> Iterable[Path]:
    ignored = {
        ".git",
        ".hg",
        ".svn",
        ".venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "cache",
        "checkpoints",
        "logs",
        "outputs",
        "runs",
        "weights",
    }
    source_suffixes = {
        ".c",
        ".cc",
        ".cfg",
        ".cpp",
        ".cu",
        ".cuh",
        ".h",
        ".hpp",
        ".ini",
        ".json",
        ".lock",
        ".py",
        ".pyi",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
    source_names = {"dockerfile", "makefile"}
    if root.is_file():
        if root.suffix.lower() in source_suffixes or root.name.lower() in source_names:
            yield root
        return
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative_parts = path.relative_to(root).parts
        if (
            path.is_file()
            and not any(part.lower() in ignored for part in relative_parts)
            and (path.suffix.lower() in source_suffixes or path.name.lower() in source_names)
        ):
            yield path


def _content_identity(root: Path) -> str:
    digest = hashlib.sha256()
    for path in _iter_source_files(root):
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8", "surrogatepass"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _source_identity(source: Path) -> Dict[str, str]:
    root = source.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Managed runtime source root does not exist: {root}")
    try:
        git_root = _run(["git", "-C", str(root), "rev-parse", "--show-toplevel"])
    except OSError:
        return {"path": str(root), "kind": "content", "content": _content_identity(root)}
    if git_root.returncode == 0:
        repository = Path(git_root.stdout.strip()).resolve()
        try:
            root.relative_to(repository)
        except ValueError:
            return {"path": str(root), "kind": "content", "content": _content_identity(root)}
        head = _run(["git", "-C", str(repository), "rev-parse", "HEAD"])
        if head.returncode == 0:
            return {
                "path": str(root),
                "kind": "git",
                "revision": head.stdout.strip(),
                "content": _content_identity(root),
            }
    return {"path": str(root), "kind": "content", "content": _content_identity(root)}


def _normalize_requirement(requirement: Pathish) -> str:
    value = os.fspath(requirement)
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return value


def _interpreter(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _tail(text: str, lines: int = _LOG_TAIL_LINES) -> str:
    return "\n".join(text.splitlines()[-lines:])


def _log_tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        return _tail(stream.read().decode("utf-8", errors="replace"))


def _process_is_alive(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == still_active
    finally:
        kernel32.CloseHandle(handle)


class _SetupLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._owned = False

    def __enter__(self) -> "_SetupLock":
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                self._remove_stale_lock()
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for managed environment lock: {self.path}"
                    )
                time.sleep(_LOCK_POLL_SECONDS)
                continue
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(json.dumps({"pid": os.getpid(), "created": time.time()}))
            self._owned = True
            return self

    def _remove_stale_lock(self) -> None:
        """Recover a lock abandoned by a terminated provisioning process."""

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            pid = int(data["pid"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            try:
                if time.time() - self.path.stat().st_mtime >= _STALE_LOCK_SECONDS:
                    self.path.unlink(missing_ok=True)
            except OSError:
                pass
            return
        if not _process_is_alive(pid):
            self.path.unlink(missing_ok=True)

    def __exit__(self, *_args: object) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)


def ensure_managed_environment(
    name: str,
    requirements: Sequence[Pathish],
    models_dir: Pathish = "models",
    *,
    python: Optional[Pathish] = None,
    python_version: Optional[str] = None,
    pip_options: Sequence[str] = (),
    no_deps_requirements: Sequence[Pathish] = (),
    source_roots: Sequence[Pathish] = (),
) -> Path:
    """Return the interpreter in an isolated environment containing requirements.

    Environments live below ``models_dir/.ayase/environments`` and are reused only
    when Python, platform, requirements, pip options, and source identities match.
    ``python_version`` automatically acquires a uv-managed interpreter, while
    ``python`` is an expert override pointing to an existing interpreter. They are
    mutually exclusive.
    Installation output is persisted beside the environments. Failed attempts never
    publish a completion manifest and are retried by the next call.
    """

    if not name or name in {".", ".."}:
        raise ValueError("Managed environment name must not be empty")
    safe_name = _SAFE_NAME.sub("-", name).strip(".-")
    if not safe_name:
        raise ValueError("Managed environment name must contain a letter or number")
    if python is not None and python_version is not None:
        raise ValueError("python and python_version are mutually exclusive")

    cache_root = Path(models_dir).expanduser().resolve() / ".ayase"
    base_python = (
        _resolve_managed_python(python_version, cache_root)
        if python_version is not None
        else Path(python or sys.executable).expanduser().resolve()
    )
    normalized_requirements = [_normalize_requirement(item) for item in requirements]
    normalized_no_deps = [_normalize_requirement(item) for item in no_deps_requirements]
    local_requirements = [
        _source_identity(Path(item))
        for item in (*normalized_requirements, *normalized_no_deps)
        if Path(item).exists()
    ]
    payload: Dict[str, Any] = {
        "schema": 1,
        "python": _python_identity(base_python),
        "python_request": python_version,
        "host_platform": platform.platform(),
        "requirements": normalized_requirements,
        "no_deps_requirements": normalized_no_deps,
        "local_requirements": local_requirements,
        "pip_options": list(pip_options),
        "sources": [_source_identity(Path(item)) for item in source_roots],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(encoded).hexdigest()
    environment = cache_root / "environments" / f"{safe_name}-{fingerprint[:16]}"
    manifest = environment / "ayase-managed.json"
    lock = cache_root / "locks" / f"{safe_name}-{fingerprint}.lock"
    log_path = cache_root / "logs" / f"{safe_name}-{fingerprint[:16]}.log"

    with _SetupLock(lock):
        interpreter = _interpreter(environment)
        if manifest.is_file() and interpreter.is_file():
            try:
                if (
                    json.loads(manifest.read_text(encoding="utf-8")).get("fingerprint")
                    == fingerprint
                ):
                    return interpreter
            except (OSError, json.JSONDecodeError):
                pass

        if environment.exists():
            shutil.rmtree(environment)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        logger.info("Preparing managed backend %s; installation log: %s", name, log_path)
        try:
            created = _run([str(base_python), "-m", "venv", str(environment), "--clear"])
            if created.returncode != 0:
                raise RuntimeError(
                    f"Unable to create managed environment with {base_python}:\n"
                    f"{_tail(created.stdout)}"
                )
            environment_python = _interpreter(environment)
            if normalized_requirements:
                command = [
                    str(environment_python),
                    "-m",
                    "pip",
                    "install",
                    *pip_options,
                    *normalized_requirements,
                ]
                returncode = _run_logged(command, log_path)
                if returncode != 0:
                    tail = _log_tail(log_path)
                    raise RuntimeError(
                        f"Managed environment installation failed for {name!r}; "
                        f"full log: {log_path}\n{tail}"
                    )
            else:
                with log_path.open("a", encoding="utf-8") as stream:
                    stream.write("No dependency-resolving requirements requested.\n")

            if normalized_no_deps:
                command = [
                    str(environment_python),
                    "-m",
                    "pip",
                    "install",
                    *pip_options,
                    "--no-deps",
                    *normalized_no_deps,
                ]
                returncode = _run_logged(command, log_path)
                if returncode != 0:
                    tail = _log_tail(log_path)
                    raise RuntimeError(
                        f"Managed no-deps installation failed for {name!r}; "
                        f"full log: {log_path}\n{tail}"
                    )

            completed = dict(payload)
            completed.update(
                {"fingerprint": fingerprint, "created": time.time(), "log": str(log_path)}
            )
            manifest_tmp = environment / "ayase-managed.json.tmp"
            manifest_tmp.write_text(
                json.dumps(completed, indent=2, sort_keys=True), encoding="utf-8"
            )
            manifest_tmp.replace(environment / "ayase-managed.json")
            return _interpreter(environment)
        except Exception:
            if environment.exists():
                shutil.rmtree(environment, ignore_errors=True)
            raise

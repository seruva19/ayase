"""Host-side source acquisition and portable staging for AVGen-Bench workers.

The immutable source cache is separate from writable per-run inputs and results.
Only the source and metadata files identified by the bundled manifest are used;
upstream demonstration media and training-log symlinks are never extracted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from .protocol import INTERNAL_DIMENSION_KEYS, SOURCE_REVISION, safe_content_token

Pathish = Union[str, os.PathLike]
_SOURCE_MARKER = ".ayase-source-revision"
_RUNNER_PATH = Path(__file__).with_name("_native_runner.py")


def _source_manifest() -> Dict[str, Any]:
    path = Path(__file__).with_name("source_manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("revision") != SOURCE_REVISION or not manifest.get("members"):
        raise ValueError("invalid AVGen-Bench source manifest")
    return dict(manifest)


def _runtime_manifest() -> Dict[str, Any]:
    manifest = json.loads(
        Path(__file__).with_name("runtime_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("schema") != 1 or manifest.get("source_revision") != SOURCE_REVISION:
        raise ValueError("invalid AVGen-Bench runtime manifest")
    return dict(manifest)


def _runtime_platform() -> str:
    if platform.machine().lower() not in {"amd64", "x86_64"}:
        raise RuntimeError("AVGen-Bench private runtimes require x86-64")
    if os.name == "nt":
        return "win-64"
    if platform.system() == "Linux":
        return "linux-64"
    raise RuntimeError("AVGen-Bench private runtimes currently support Windows and Linux")


def ensure_runtime(dimension: str, models_dir: Pathish, *, test_mode: bool = False) -> Path:
    """Automatically prepare one specialist without injecting host ML dependencies."""
    if test_mode:
        raise RuntimeError("AVGen-Bench runtime preparation is disabled in test mode")
    manifest = _runtime_manifest()
    if dimension in manifest["lightweight_dimensions"]:
        if sys.version_info >= (3, 10):
            return Path(sys.executable).resolve()
        from ayase.managed_runtime import ensure_managed_environment

        return ensure_managed_environment("avgen-api", (), models_dir, python_version="3.10.19")
    if dimension not in manifest["runtimes"]:
        raise ValueError("unknown AVGen-Bench dimension")
    spec = manifest["runtimes"][dimension]["platforms"][_runtime_platform()]
    name = "avgen-" + dimension.lower().replace("-", "_")
    if spec["provisioner"] == "ensure_managed_native_environment":
        from ayase.managed_native import ensure_managed_native_environment

        return ensure_managed_native_environment(
            name,
            spec["conda_packages"],
            models_dir,
            python_version=spec["python_version"],
            pip_requirements=spec["pip_requirements"],
            pip_options=spec["pip_options"],
        )
    from ayase.managed_runtime import ensure_managed_environment
    from .packaging_runtime import ensure_package_wheel

    wheels = [
        str(ensure_package_wheel(package, manifest["source_packages"][package], models_dir))
        for package in spec["source_package_names"]
    ]
    return ensure_managed_environment(
        name,
        spec["pip_requirements"],
        models_dir,
        python_version=spec["python_version"],
        pip_options=spec["pip_options"],
        no_deps_requirements=wheels,
    )


def runtime_ffmpeg(interpreter: Pathish, models_dir: Pathish) -> Path:
    """Prefer the specialist's shared FFmpeg, otherwise acquire Ayase's media tool."""
    from ayase.managed_native import managed_native_executable

    try:
        return managed_native_executable(interpreter, "ffmpeg")
    except (RuntimeError, FileNotFoundError):
        from ayase.managed_ffmpeg import resolve_media_tool

        return resolve_media_tool("ffmpeg", models_dir=models_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_parts(name: str) -> tuple:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or "\\" in name
        or any(part in {"", ".", ".."} or ":" in part for part in path.parts)
    ):
        raise ValueError("unsafe AVGen-Bench source path")
    return path.parts


def verify_source(root: Pathish, manifest: Optional[Mapping[str, Any]] = None) -> Path:
    """Check every selected source file against its immutable Git-blob digest."""
    root = Path(root).expanduser().resolve()
    expected = manifest if manifest is not None else _source_manifest()
    for name, entry in expected["members"].items():
        target = root.joinpath(*_member_parts(name))
        resolved = target.resolve()
        if root not in resolved.parents or target.is_symlink():
            raise ValueError("source entry escapes the AVGen-Bench source root")
        if (
            not target.is_file()
            or target.stat().st_size != entry["bytes"]
            or _sha256(target) != entry["sha256"]
        ):
            raise ValueError(f"AVGen-Bench source verification failed: {name}")
    return root


def _download_source_archive(manifest: Mapping[str, Any], destination: Path) -> None:
    size = 0
    with urllib.request.urlopen(manifest["archive_url"], timeout=180) as response:
        with destination.open("wb") as target:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > manifest["archive_bytes"]:
                    raise ValueError("AVGen-Bench source archive exceeds its pinned size")
                target.write(chunk)
    if size != manifest["archive_bytes"] or _sha256(destination) != manifest["archive_sha256"]:
        raise ValueError("AVGen-Bench source archive failed SHA-256 verification")


def _extract_source(archive: Path, target: Path, manifest: Mapping[str, Any]) -> None:
    """Extract only pinned regular files, without trusting ZIP path attributes."""
    root_name = manifest["archive_root"]
    _member_parts(root_name)
    with zipfile.ZipFile(archive) as bundle:
        entries = {entry.filename: entry for entry in bundle.infolist()}
        if len(entries) != len(bundle.infolist()):
            raise ValueError("duplicate paths in AVGen-Bench source archive")
        for name, expected in manifest["members"].items():
            parts = _member_parts(name)
            member = entries.get(root_name + "/" + name)
            if member is None or member.is_dir():
                raise ValueError(f"AVGen-Bench archive omits source file: {name}")
            mode = member.external_attr >> 16
            if stat.S_IFMT(mode) not in {0, stat.S_IFREG}:
                raise ValueError("AVGen-Bench source entry is not a regular file")
            if member.file_size != expected["bytes"]:
                raise ValueError(f"AVGen-Bench source file has an unexpected size: {name}")
            output = target.joinpath(*parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, output.open("xb") as destination:
                shutil.copyfileobj(source, destination)
    verify_source(target, manifest)


def ensure_source(
    models_dir: Pathish,
    *,
    source_override: Optional[Pathish] = None,
    test_mode: bool = False,
) -> Path:
    """Automatically acquire a checked source cache; expert overrides stay optional."""
    if test_mode:
        raise RuntimeError("AVGen-Bench source acquisition is disabled in test mode")
    manifest = _source_manifest()
    if source_override is not None:
        return verify_source(source_override, manifest)
    from ayase.managed_runtime import _SetupLock

    parent = Path(models_dir).expanduser().resolve() / ".ayase" / "sources"
    destination = parent / ("avgen-bench-" + SOURCE_REVISION)
    with _SetupLock(parent / ("avgen-bench-" + SOURCE_REVISION + ".lock")):
        if destination.is_dir():
            marker = destination / _SOURCE_MARKER
            if not marker.is_file() or marker.read_text(encoding="ascii") != SOURCE_REVISION:
                raise ValueError("AVGen-Bench cache has no matching source revision marker")
            return verify_source(destination, manifest)
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="avgen-source-", dir=parent) as temporary:
            work = Path(temporary)
            archive = work / "source.zip"
            _download_source_archive(manifest, archive)
            extracted = work / "source"
            extracted.mkdir()
            _extract_source(archive, extracted, manifest)
            (extracted / _SOURCE_MARKER).write_text(SOURCE_REVISION, encoding="ascii")
            extracted.replace(destination)
    return destination


def _stage_video(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copyfile(source, target)
        return "copy"


def stage_records(records: Sequence[Mapping[str, Any]], work_dir: Pathish) -> Dict[str, Any]:
    """Build source-compatible inputs without changing prompt text or source files.

    ``content`` titles become safe physical tokens only in temporary input copies.
    Original identities and the reversible token mapping remain in the ledger.
    ``expected_faces`` is the full source annotation object, when supplied.
    """
    work = Path(work_dir).expanduser().resolve()
    videos = work / "videos"
    prompts = work / "prompts"
    faces = work / "face_prompts"
    if any(path.exists() for path in (videos, prompts, faces)):
        raise ValueError("AVGen-Bench staging requires a fresh work directory")
    prepared = []
    identities = set()
    ids = set()
    for row in records:
        record = dict(row)
        category, content, prompt = record["category"], record["content"], record["prompt_text"]
        if not isinstance(category, str) or not re.fullmatch(r"[a-z0-9_-]+", category):
            raise ValueError("AVGen-Bench category must be a lowercase directory name")
        if category.upper() in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }:
            raise ValueError("AVGen-Bench category is a reserved Windows directory name")
        if not isinstance(prompt, str):
            raise ValueError(
                "AVGen-Bench prompt text must be a string; captionless inputs use an empty string"
            )
        token = safe_content_token(category, content)
        identity = (category, content)
        identifier = record["id"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in ids
            or identity in identities
        ):
            raise ValueError("AVGen-Bench input identities must be unique")
        ids.add(identifier)
        identities.add(identity)
        source = Path(record["video_path"]).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"AVGen-Bench video is missing: {source}")
        if source.suffix.lower() != ".mp4":
            raise ValueError(
                "the released AVGen-Bench suite requires MP4 input; renaming or transcoding would change the evaluated media"
            )
        record.update(
            video_path=str(source),
            safe_content=token,
            staged_video_path=str(videos / category / (token + ".mp4")),
        )
        prepared.append(record)
    grouped_prompts: Dict[str, list] = {}
    grouped_faces: Dict[str, list] = {}
    for record in prepared:
        record["staging_method"] = _stage_video(
            Path(record["video_path"]), Path(record["staged_video_path"])
        )
        category = record["category"]
        grouped_prompts.setdefault(category, []).append(
            {"content": record["safe_content"], "prompt": record["prompt_text"]}
        )
        annotation = record.get("expected_faces")
        if annotation is not None:
            if not isinstance(annotation, dict):
                raise ValueError("expected_faces must be a full annotation object")
            annotation = dict(annotation)
            annotation.update(content=record["safe_content"], prompt=record["prompt_text"])
            grouped_faces.setdefault(category, []).append(annotation)
    for directory, groups, suffix in (
        (prompts, grouped_prompts, ".json"),
        (faces, grouped_faces, ".expected_faces.json"),
    ):
        directory.mkdir(parents=True, exist_ok=True)
        for category, rows in groups.items():
            (directory / (category + suffix)).write_text(
                json.dumps(rows, ensure_ascii=False, allow_nan=False), encoding="utf-8"
            )
    videos.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema": 1,
        "records": prepared,
        "videos_dir": str(videos),
        "prompts_dir": str(prompts),
        "face_prompts_dir": str(faces),
    }
    work.mkdir(parents=True, exist_ok=True)
    (work / "staging.json").write_text(
        json.dumps(ledger, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
    )
    return ledger


def _terminate_worker(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                timeout=15,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)


def _private_environment_prefix(interpreter: Path, *, windows: bool) -> Path:
    """Resolve either a conda-prefix or venv-prefix from its private interpreter."""
    if windows and interpreter.parent.name.lower() == "scripts":
        return interpreter.parent.parent
    return interpreter.parent if windows else interpreter.parent.parent


def _private_gpu_library_dirs(prefix: Path, *, windows: bool) -> list[Path]:
    """Find existing Torch/NVIDIA shared-library directories inside one private prefix."""
    if windows:
        site_packages = [prefix / "Lib" / "site-packages"]
    else:
        site_packages = sorted((prefix / "lib").glob("python*/site-packages"))
    directories: list[Path] = []
    for site in site_packages:
        candidates = [site / "torch" / "lib"]
        nvidia = site / "nvidia"
        if nvidia.is_dir():
            candidates.extend(sorted(nvidia.glob("*/bin")))
            candidates.extend(sorted(nvidia.glob("*/lib")))
        for candidate in candidates:
            if candidate.is_dir() and candidate not in directories:
                directories.append(candidate)
    return directories


def _prepend_environment_paths(
    environment: Dict[str, str], name: str, directories: Sequence[Path]
) -> None:
    values = [str(path) for path in directories]
    existing = environment.get(name)
    if existing:
        values.append(existing)
    if values:
        environment[name] = os.pathsep.join(values)


def execute_worker(
    request: Mapping[str, Any],
    interpreter: Pathish,
    *,
    api_key: Optional[str] = None,
    timeout_s: float = 14400,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """Run one specialist, with credentials confined to its private environment.

    The caller supplies an automatically provisioned interpreter. No host package
    import is required in that environment; result files use strict JSON IPC.
    """
    if test_mode:
        raise RuntimeError("AVGen-Bench workers are disabled in test mode")
    dimension = request.get("dimension")
    if dimension not in INTERNAL_DIMENSION_KEYS:
        raise ValueError("unknown AVGen-Bench dimension")
    config = request.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("AVGen-Bench worker config must be a mapping")
    if any(
        config.get(name) is not None for name in ("api_key", "gemini_api_key", "google_api_key")
    ):
        raise ValueError("pass AVGen-Bench credentials privately, never in the worker request")
    if timeout_s <= 0 or not math.isfinite(timeout_s):
        raise ValueError("AVGen-Bench worker timeout must be finite and positive")
    work = Path(request["work_dir"]).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    stem = str(dimension).replace("-", "_")
    input_file = work / (stem + "-request.json")
    result_file = work / (stem + "-result.json")
    log_file = work / (stem + "-worker.log")
    if result_file.exists() or input_file.exists():
        raise ValueError("AVGen-Bench worker requires a fresh operation directory")
    input_file.write_text(
        json.dumps(dict(request), ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["HF_HOME"] = str(
        Path(request["models_dir"]).expanduser().resolve() / ".ayase" / "cache" / "huggingface"
    )
    environment["TORCH_HOME"] = str(
        Path(request["models_dir"]).expanduser().resolve() / ".ayase" / "cache" / "torch"
    )
    if api_key is not None:
        environment["GEMINI_API_KEY"] = api_key
        environment["GOOGLE_API_KEY"] = api_key
    # POSIX venv executables are usually symlinks to a base interpreter.
    # Keep the venv spelling so Python loads its pyvenv.cfg and installed packages.
    python = Path(interpreter).expanduser().absolute()
    windows = os.name == "nt"
    prefix = _private_environment_prefix(python, windows=windows)
    gpu_libraries = _private_gpu_library_dirs(prefix, windows=windows)
    if (prefix / "conda-meta").is_dir():
        environment["CONDA_PREFIX"] = str(prefix)
        bins = (
            [prefix / "Library" / "bin", prefix / "Scripts", prefix]
            if windows
            else [prefix / "bin"]
        )
        _prepend_environment_paths(environment, "PATH", bins)
    if windows:
        _prepend_environment_paths(environment, "PATH", gpu_libraries)
    else:
        native_libraries = [prefix / "lib"] if (prefix / "conda-meta").is_dir() else []
        _prepend_environment_paths(
            environment, "LD_LIBRARY_PATH", [*native_libraries, *gpu_libraries]
        )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(python),
                str(_RUNNER_PATH),
                "--request",
                str(input_file),
                "--result",
                str(result_file),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=work,
            creationflags=flags,
            start_new_session=os.name != "nt",
        )
        try:
            exit_code = process.wait(timeout=timeout_s)
        except BaseException:
            _terminate_worker(process)
            raise
    if exit_code != 0 or not result_file.is_file():
        raise RuntimeError(
            f"AVGen-Bench {dimension} worker failed (exit={exit_code}); see {log_file}"
        )

    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite value in AVGen-Bench worker result: {value}")

    result = json.loads(result_file.read_text(encoding="utf-8"), parse_constant=reject_constant)
    if (
        not isinstance(result, dict)
        or result.get("schema") != 1
        or result.get("dimension") != dimension
        or result.get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("AVGen-Bench worker result does not match its request")
    expected_ids = [row["id"] for row in request.get("records", [])]
    rows = result.get("records")
    if not isinstance(rows, list) or [row.get("id") for row in rows] != expected_ids:
        raise ValueError("AVGen-Bench worker result omits or reorders input records")
    for row in rows:
        score = row.get("raw_score")
        if row.get("status") == "scored" and (
            type(score) not in (int, float) or not math.isfinite(score)
        ):
            raise ValueError("AVGen-Bench worker returned an invalid numeric measurement")
    return dict(result)

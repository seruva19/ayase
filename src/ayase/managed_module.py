"""Bridge pipeline modules into dependency-isolated persistent workers."""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import subprocess
import sys
import time
import types
import uuid
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .managed_runtime import ensure_managed_environment
from .models import Sample

logger = logging.getLogger(__name__)

_CORE_REQUIREMENTS = (
    "pydantic>=2.5,<3",
    "pydantic-settings>=2.1,<3",
    "numpy>=1.24,<2",
    "opencv-python>=4.9,<4.12",
    "pillow>=10.2,<13",
    "rich>=13.7,<15",
    "tqdm>=4.66,<5",
    "tomli>=2,<3;python_version<'3.11'",
    "tomli-w>=1,<2",
)
_SETUP_TIMEOUT = 7200.0
_DEFAULT_OPERATION_TIMEOUT = 1800.0
_METHODS = ("process", "process_batch", "post_process", "on_execute", "on_dispose")


def _read_json_after_publish(path: Path, deadline: float) -> Dict[str, Any]:
    """Read a protocol file while tolerating transient Windows share locks."""
    while True:
        try:
            return dict(json.loads(path.read_text(encoding="utf-8")))
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def _best_effort_unlink(path: Path) -> None:
    """Remove protocol data without masking an already completed operation."""
    deadline = time.monotonic() + 0.25
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                return
            time.sleep(0.02)


def _dependency_for(name: str) -> Any:
    try:
        from .backend_dependencies import get_backend_dependency
    except ImportError:
        return None
    return get_backend_dependency(name)


def _managed_requirements(dependency: Any) -> Sequence[str]:
    from .backend_dependencies import managed_requirements

    return _requirements_with_core(managed_requirements(dependency))


def _managed_pip_options(dependency: Any) -> Sequence[str]:
    from .backend_dependencies import managed_pip_options

    return tuple(str(item) for item in managed_pip_options(dependency))


def _creation_flags() -> int:
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) if os.name == "nt" else 0


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9_.-]+)", requirement)
    return match.group(1).lower().replace("_", "-") if match else requirement


def _requirements_with_core(requirements: Sequence[str]) -> Sequence[str]:
    declared = {_requirement_name(item) for item in requirements}
    core = [item for item in _CORE_REQUIREMENTS if _requirement_name(item) not in declared]
    return (*core, *requirements)


def _copy_sample(
    target: Sample, document: Mapping[str, Any], written_fields: Sequence[str] = ()
) -> Sample:
    updated = Sample.model_validate(document)
    for field in Sample.model_fields:
        setattr(target, field, deepcopy(getattr(updated, field)))
    if target.quality_metrics is not None:
        for field in written_fields:
            if field in type(target.quality_metrics).model_fields:
                setattr(target.quality_metrics, field, getattr(target.quality_metrics, field))
    return target


def _json_value(value: Any, location: str = "config") -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return _json_value(value.value, location)
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{location}[]") for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"Managed module {location} contains a non-string mapping key")
        return {key: _json_value(item, f"{location}.{key}") for key, item in value.items()}
    raise TypeError(
        f"Managed module {location} contains unsupported {type(value).__name__}; "
        "use JSON values, pathlib.Path, or Enum"
    )


def _worker_environment(module: Any, interpreter: Path) -> Dict[str, str]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(_package_root()), *(item for item in [existing_pythonpath] if item)]
    )
    cache_root = Path(module.config.get("models_dir", "models")).resolve() / ".ayase" / "cache"
    environment["HF_HOME"] = str(cache_root / "huggingface")
    environment["TORCH_HOME"] = str(cache_root / "torch")
    Path(environment["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(environment["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)

    conda_prefix = interpreter.parent if os.name == "nt" else interpreter.parent.parent
    if (conda_prefix / "conda-meta").is_dir():
        environment["CONDA_PREFIX"] = str(conda_prefix)
        native_paths = [conda_prefix / "Library" / "bin", conda_prefix / "Scripts", conda_prefix]
        if os.name != "nt":
            native_paths = [conda_prefix / "bin"]
        environment["PATH"] = os.pathsep.join(
            [*(str(path) for path in native_paths), environment.get("PATH", "")]
        )
    return environment


class ManagedModuleBridge:
    """Own one persistent isolated worker and proxy a module's lifecycle to it."""

    def __init__(self, module: Any, interpreter: Path, runtime_root: Path) -> None:
        self.module = module
        self.interpreter = interpreter
        self.runtime_root = runtime_root
        self.timeout = float(
            module.config.get("managed_operation_timeout", _DEFAULT_OPERATION_TIMEOUT)
        )
        self._originals: Dict[str, Callable[..., Any]] = {
            name: getattr(module, name) for name in _METHODS
        }
        self._process: Optional[subprocess.Popen[str]] = None
        self._log_stream: Any = None
        self.backend_identity: Optional[str] = None
        self._request_dir = runtime_root / "requests"
        self._request_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        config_path = self.runtime_root / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = dict(self.module.config)
        config["_ayase_managed_worker"] = True
        config_path.write_text(json.dumps(_json_value(config)), encoding="utf-8")
        log_path = self.runtime_root / "worker.log"
        self._log_stream = log_path.open("a", encoding="utf-8")
        environment = _worker_environment(self.module, self.interpreter)
        command = [
            str(self.interpreter),
            "-m",
            "ayase._managed_module_worker",
            "--module",
            type(self.module).__module__,
            "--qualname",
            type(self.module).__qualname__,
            "--config",
            str(config_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=self._log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            env=environment,
            creationflags=_creation_flags(),
        )
        try:
            response = self.request("mount", {}, timeout=_SETUP_TIMEOUT)
            if not response.get("mounted"):
                raise RuntimeError(
                    f"Managed worker did not make module {self.module.name} available "
                    f"(backend={response.get('backend')!r}, "
                    f"ml_available={response.get('ml_available')!r})"
                )
        except Exception:
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=10)
            if self._log_stream is not None:
                self._log_stream.close()
                self._log_stream = None
            raise
        backend = response.get("backend")
        self.backend_identity = str(backend) if backend else None
        self._forward_dataset_metrics(response)
        self._bind_proxies()

    def _bind_proxies(self) -> None:
        self.module.process = types.MethodType(
            lambda _module, sample: self.process(sample), self.module
        )
        self.module.process_batch = types.MethodType(
            lambda _module, samples: self.process_batch(samples), self.module
        )
        self.module.post_process = types.MethodType(
            lambda _module, samples: self.post_process(samples), self.module
        )
        self.module.on_execute = types.MethodType(lambda _module: self.on_execute(), self.module)
        self.module.on_dispose = types.MethodType(lambda _module: self.dispose(), self.module)

    def request(
        self, operation: str, payload: Mapping[str, Any], *, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Managed module worker is not running")
        token = uuid.uuid4().hex
        request_path = self._request_dir / f"{token}.request.json"
        reply_path = self._request_dir / f"{token}.reply.json"
        temporary = request_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"operation": operation, "payload": payload, "reply": str(reply_path)},
            ),
            encoding="utf-8",
        )
        os.replace(temporary, request_path)
        process.stdin.write(str(request_path) + "\n")
        process.stdin.flush()
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        try:
            while not reply_path.is_file():
                returncode = process.poll()
                if returncode is not None:
                    raise RuntimeError(
                        f"Managed worker for {self.module.name} exited with code {returncode}; "
                        f"log: {self.runtime_root / 'worker.log'}"
                    )
                if time.monotonic() >= deadline:
                    process.kill()
                    raise TimeoutError(
                        f"Managed worker operation {operation!r} timed out for {self.module.name}"
                    )
                time.sleep(0.02)
            response = _read_json_after_publish(reply_path, deadline)
        finally:
            _best_effort_unlink(request_path)
            _best_effort_unlink(reply_path)
        if not response.get("ok"):
            raise RuntimeError(
                f"Managed worker {operation!r} failed for {self.module.name}: "
                f"{response.get('error')}\n{response.get('traceback', '')}"
            )
        return dict(response.get("result", {}))

    def _record_sample_error(self, sample: Sample, operation: str, exc: Exception) -> Sample:
        from .pipeline import Pipeline

        detail = f"managed backend {operation} failed: {type(exc).__name__}: {exc}"
        logger.error("Module %s %s", self.module.name, detail)
        Pipeline._register_module_failure(sample, self.module.name, detail)
        pipeline = self.module.pipeline
        if pipeline is not None:
            pipeline.module_failures[self.module.name] = detail
        return sample

    @staticmethod
    def _record(sample: Sample) -> Dict[str, Any]:
        return {"token": str(id(sample)), "sample": sample.model_dump(mode="json")}

    def _forward_dataset_metrics(self, result: Mapping[str, Any]) -> None:
        pipeline = self.module.pipeline
        if pipeline is not None:
            previous = pipeline._active_dataset_module
            pipeline._active_dataset_module = self.module
            try:
                for name, value in result.get("dataset_metrics", {}).items():
                    pipeline.add_dataset_metric(name, value)
            finally:
                pipeline._active_dataset_module = previous

    def process(self, sample: Sample) -> Sample:
        try:
            result = self.request("process", {"record": self._record(sample)})
            self._forward_dataset_metrics(result)
            return _copy_sample(sample, result["sample"], result.get("written_fields", ()))
        except Exception as exc:
            return self._record_sample_error(sample, "process", exc)

    def process_batch(self, samples: List[Sample]) -> List[Sample]:
        try:
            result = self.request(
                "process_batch", {"records": [self._record(sample) for sample in samples]}
            )
            self._forward_dataset_metrics(result)
            documents = result["samples"]
            if len(documents) != len(samples):
                raise RuntimeError("managed worker returned a different batch size")
            written = result.get("written_fields", [[] for _ in samples])
            return [
                _copy_sample(sample, document, fields)
                for sample, document, fields in zip(samples, documents, written)
            ]
        except Exception as exc:
            return [self._record_sample_error(sample, "process_batch", exc) for sample in samples]

    def on_execute(self) -> None:
        result = self.request("on_execute", {})
        self._forward_dataset_metrics(result)

    def post_process(self, samples: List[Sample]) -> None:
        result = self.request(
            "post_process", {"records": [self._record(sample) for sample in samples]}
        )
        documents = result["samples"]
        if len(documents) != len(samples):
            raise RuntimeError("managed worker returned a different post-process batch size")
        written = result.get("written_fields", [[] for _ in samples])
        for sample, document, fields in zip(samples, documents, written):
            _copy_sample(sample, document, fields)
        self._forward_dataset_metrics(result)

    def dispose(self) -> None:
        try:
            if self._process is not None and self._process.poll() is None:
                self.request("on_dispose", {})
                self._process.wait(timeout=10)
        finally:
            if self._process is not None and self._process.stdin is not None:
                self._process.stdin.close()
            if self._process is not None and self._process.poll() is None:
                self._process.kill()
                self._process.wait(timeout=10)
            if self._log_stream is not None:
                self._log_stream.close()
            for name, method in self._originals.items():
                setattr(self.module, name, method)
            self.module._managed_bridge = None
            self.module._mounted = False


def attach_managed_backend(module: Any) -> bool:
    """Attach an isolated worker when the module has a backend dependency spec.

    Returns ``False`` when the module is unregistered, already inside a worker, or
    running in test mode. Provisioning and worker startup failures raise visibly.
    """

    if (
        module.test_mode
        or module.config.get("_ayase_managed_worker")
        or module.config.get("managed_runtime") is False
    ):
        return False
    dependency = _dependency_for(module.name)
    if dependency is None:
        return False
    conditions = tuple(getattr(dependency, "config_conditions", ()))
    if conditions and any(str(module.config.get(key)) != expected for key, expected in conditions):
        return False
    unsupported_reason = getattr(dependency, "unsupported_reason", None)
    if unsupported_reason:
        logger.warning(
            "Automatic isolated backend is unavailable for %s: %s; trying the configured host backend",
            module.name,
            unsupported_reason,
        )
        return False
    platform_reasons = dict(getattr(dependency, "unsupported_platforms", ()))
    platform_keys = {sys.platform}
    if os.name == "nt":
        platform_keys.add("win-64")
    elif sys.platform.startswith("linux"):
        platform_keys.add("linux-64")
    unsupported_platform = next((key for key in platform_keys if key in platform_reasons), None)
    if unsupported_platform is not None:
        logger.warning(
            "Automatic isolated backend is unavailable for %s on %s: %s; "
            "trying the configured host backend",
            module.name,
            unsupported_platform,
            platform_reasons[unsupported_platform],
        )
        return False
    existing = getattr(module, "_managed_bridge", None)
    if existing is not None:
        return True

    models_dir = Path(module.config.get("models_dir", "models")).expanduser().resolve()
    requirements = _managed_requirements(dependency)
    pip_options = tuple(_managed_pip_options(dependency))
    conda_packages = tuple(getattr(dependency, "conda_packages", ()))
    no_deps_requirements = tuple(getattr(dependency, "no_deps_requirements", ()))
    environment_key = hashlib.sha256(
        json.dumps(
            {
                "requirements": list(requirements),
                "python": dependency.python_version,
                "pip_options": list(pip_options),
                "conda_packages": list(conda_packages),
                "no_deps_requirements": list(no_deps_requirements),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    core_sources = [
        str(path)
        for path in (
            __file__,
            Path(__file__).with_name("_managed_module_worker.py"),
            Path(__file__).with_name("models.py"),
            Path(__file__).with_name("pipeline.py"),
            Path(__file__).with_name("modules"),
        )
    ]
    if conda_packages:
        if no_deps_requirements:
            raise ValueError("no-deps requirements are not supported by native environments")
        from .managed_native import ensure_managed_native_environment

        interpreter = ensure_managed_native_environment(
            f"backend-{environment_key}",
            conda_packages,
            models_dir,
            python_version=dependency.python_version,
            pip_requirements=requirements,
            pip_options=pip_options,
        )
    else:
        interpreter = ensure_managed_environment(
            f"backend-{environment_key}",
            requirements,
            models_dir,
            python_version=dependency.python_version,
            pip_options=pip_options,
            no_deps_requirements=no_deps_requirements,
            source_roots=(*core_sources, *tuple(getattr(dependency, "source_roots", ()))),
        )
    safe_module_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", module.name).strip(".-")
    runtime_root = (
        models_dir / ".ayase" / "workers" / safe_module_name / f"{os.getpid()}-{uuid.uuid4().hex}"
    )
    bridge = ManagedModuleBridge(module, interpreter, runtime_root)
    module._managed_bridge = bridge
    try:
        bridge.start()
    except Exception:
        module._managed_bridge = None
        bridge.dispose()
        raise
    module._mounted = True
    backend_identity = getattr(bridge, "backend_identity", None)
    module._backend = f"managed:{backend_identity}" if backend_identity else "managed:unknown"
    if hasattr(module, "_ml_available"):
        module._ml_available = True
    return True

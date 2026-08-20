"""Install and manage the private Ayase runtime environment."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import venv
from pathlib import Path
from typing import Optional

from ayase_protocol import SHUTDOWN_PATH, validate_health

from .transport import Transport


def _cache_root() -> Path:
    configured = os.environ.get("AYASE_CLIENT_HOME")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "ayase-client"


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


class RuntimeManager:
    def __init__(self, version: str) -> None:
        self.version = version
        self.root = _cache_root() / "runtimes" / version
        self.process: Optional[subprocess.Popen[bytes]] = None
        self.transport: Optional[Transport] = None
        self._log_handle = None

    @property
    def python(self) -> Path:
        override = os.environ.get("AYASE_RUNTIME_PYTHON")
        return Path(override) if override else _venv_python(self.root)

    def install(self) -> Path:
        override = os.environ.get("AYASE_RUNTIME_PYTHON")
        if override:
            return Path(override)
        python = self.python
        marker = self.root / ".installed"
        installed = marker.exists() and marker.read_text(encoding="utf-8").strip() == self.version
        if python.exists() and installed:
            return python
        self.root.mkdir(parents=True, exist_ok=True)
        if not python.exists():
            venv.EnvBuilder(with_pip=True).create(self.root)
        spec = os.environ.get("AYASE_RUNTIME_SPEC", f"ayase=={self.version}")
        subprocess.run(
            [str(python), "-m", "pip", "install", spec],
            check=True,
        )
        marker.write_text(self.version, encoding="utf-8")
        return python

    @staticmethod
    def _free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

    def start(self) -> Transport:
        if self.transport is not None:
            return self.transport
        python = self.install()
        port = self._free_port()
        token = os.urandom(32).hex()
        self.root.mkdir(parents=True, exist_ok=True)
        log_path = self.root / "worker.log"
        self._log_handle = log_path.open("ab")
        self.process = subprocess.Popen(
            [
                str(python),
                "-m",
                "ayase.isolated_worker",
                "--port",
                str(port),
            ],
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "AYASE_WORKER_TOKEN": token},
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
        transport = Transport(f"http://127.0.0.1:{port}", token)
        deadline = time.monotonic() + 120
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"Ayase worker exited; see {log_path}")
            try:
                validate_health(transport.health(), self.version)
                self.transport = transport
                return transport
            except Exception as exc:
                last_error = exc
                time.sleep(0.2)
        raise RuntimeError(f"Ayase worker did not start; see {log_path}: {last_error}")

    def stop(self) -> None:
        if self.transport is not None:
            try:
                self.transport.request("POST", SHUTDOWN_PATH, {})
            except Exception:
                pass
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        if self._log_handle is not None:
            self._log_handle.close()
        self.transport = None
        self.process = None

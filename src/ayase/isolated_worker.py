"""Local JSON worker used by Ayase's lightweight isolated-client facade.

The worker deliberately uses only the Python standard library for transport.
It is bound to loopback and protected by a per-process bearer token; media is
passed by path, so local runs do not copy large files through the protocol.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from ayase_protocol import (
    EXPORT_PATH,
    HEALTH_PATH,
    PROTOCOL_VERSION,
    RUN_PATH,
    SHUTDOWN_PATH,
)

from . import __version__
from .config import AyaseConfig
from .models import Sample
from .pipeline import AyasePipeline

class _WorkerState:
    def __init__(self, token: str) -> None:
        self.token = token
        self.pipelines: Dict[str, AyasePipeline] = {}
        self.lock = threading.Lock()


def _build_pipeline(payload: Dict[str, Any]) -> AyasePipeline:
    config = payload.get("config")
    if isinstance(config, dict):
        config = AyaseConfig.model_validate(config)
    return AyasePipeline(
        config=config,
        profile=payload.get("profile"),
        modules=payload.get("modules"),
    )


class _Handler(BaseHTTPRequestHandler):
    server: "_WorkerServer"

    def log_message(self, format: str, *args: Any) -> None:
        # Library/module logs remain available on stderr; suppress access noise.
        return

    def _send(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        return secrets.compare_digest(supplied, f"Bearer {self.server.state.token}")

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 64 * 1024 * 1024:
            raise ValueError("request is too large")
        data = self.rfile.read(length)
        return json.loads(data) if data else {}

    def do_GET(self) -> None:
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        if self.path == HEALTH_PATH:
            self._send(
                200,
                {
                    "status": "ok",
                    "protocol_version": PROTOCOL_VERSION,
                    "ayase_version": __version__,
                },
            )
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._authorized():
            self._send(401, {"error": "unauthorized"})
            return
        try:
            payload = self._read_json()
            if self.path == RUN_PATH:
                pipeline = _build_pipeline(payload)
                samples_data = payload.get("samples")
                samples = None
                if samples_data is not None:
                    samples = [Sample.model_validate(item) for item in samples_data]
                results = pipeline.run(
                    payload["dataset_path"],
                    samples=samples,
                    recursive=payload.get("recursive", True),
                )
                session_id = secrets.token_urlsafe(18)
                with self.server.state.lock:
                    self.server.state.pipelines[session_id] = pipeline
                self._send(
                    200,
                    {
                        "session_id": session_id,
                        "results": {
                            key: value.model_dump(mode="json") for key, value in results.items()
                        },
                        "stats": pipeline.stats.model_dump(mode="json"),
                    },
                )
                return
            if self.path == EXPORT_PATH:
                with self.server.state.lock:
                    pipeline = self.server.state.pipelines[payload["session_id"]]
                pipeline.export(payload["path"], format=payload.get("format", "json"))
                self._send(200, {"status": "ok"})
                return
            if self.path == SHUTDOWN_PATH:
                self._send(200, {"status": "stopping"})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            self._send(404, {"error": "not found"})
        except KeyError as exc:
            self._send(400, {"error": f"missing or unknown value: {exc}"})
        except Exception as exc:
            self._send(500, {"error": str(exc), "error_type": type(exc).__name__})


class _WorkerServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], token: str) -> None:
        self.state = _WorkerState(token)
        super().__init__(address, _Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Ayase isolated worker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--token", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "::1", "localhost"}:
        parser.error("the isolated worker may only bind to loopback")
    token = args.token or os.environ.get("AYASE_WORKER_TOKEN")
    if not token:
        parser.error("AYASE_WORKER_TOKEN is required")
    server = _WorkerServer((args.host, args.port), token)
    server.serve_forever()
    server.server_close()


if __name__ == "__main__":
    main()

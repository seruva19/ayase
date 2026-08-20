"""Isolated facade matching the common ``ayase.AyasePipeline`` workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from ayase_protocol import EXPORT_PATH, RUN_PATH, validate_health

from .models import DatasetStats, Sample
from .runtime import RuntimeManager
from .transport import Transport


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"cannot send {type(value).__name__} to isolated Ayase")


class AyasePipeline:
    """Drop-in facade for dataset runs through an isolated Ayase worker."""

    def __init__(
        self,
        *,
        config: Optional[Any] = None,
        profile: Optional[Union[Path, str, Dict[str, Any]]] = None,
        modules: Optional[List[str]] = None,
        endpoint: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.config = config
        self.profile = profile
        self.modules = modules
        self._manager: Optional[RuntimeManager] = None
        self._transport = Transport(endpoint, token or "") if endpoint else None
        self._connection_checked = False
        self._results: Dict[str, Sample] = {}
        self._stats = DatasetStats({"total_samples": 0, "valid_samples": 0, "invalid_samples": 0, "total_size": 0})
        self._session_id: Optional[str] = None

    def _connection(self) -> Transport:
        if self._transport is None:
            from . import __version__

            self._manager = RuntimeManager(__version__)
            self._transport = self._manager.start()
            self._connection_checked = True
        elif not self._connection_checked:
            from . import __version__

            validate_health(self._transport.health(), __version__)
            self._connection_checked = True
        return self._transport

    def run(
        self,
        dataset_path: Union[str, Path],
        *,
        samples: Optional[Iterable[Any]] = None,
        recursive: bool = True,
    ) -> Dict[str, Sample]:
        payload = {
            "dataset_path": str(Path(dataset_path).resolve()),
            "config": _json_value(self.config),
            "profile": _json_value(self.profile),
            "modules": self.modules,
            "recursive": recursive,
            "samples": None if samples is None else [_json_value(item) for item in samples],
        }
        response = self._connection().request("POST", RUN_PATH, payload)
        self._session_id = response["session_id"]
        self._results = {key: Sample(value) for key, value in response["results"].items()}
        self._stats = DatasetStats(response["stats"])
        return self._results

    def export(self, path: Union[str, Path], format: str = "json") -> None:
        if self._session_id is None:
            raise RuntimeError("run() must be called before export()")
        self._connection().request(
            "POST",
            EXPORT_PATH,
            {"session_id": self._session_id, "path": str(Path(path).resolve()), "format": format},
        )

    @property
    def results(self) -> Dict[str, Sample]:
        return self._results

    @property
    def stats(self) -> DatasetStats:
        return self._stats

    def close(self) -> None:
        if self._manager is not None:
            self._manager.stop()
            self._manager = None
            self._transport = None
            self._connection_checked = False

    def __enter__(self) -> "AyasePipeline":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

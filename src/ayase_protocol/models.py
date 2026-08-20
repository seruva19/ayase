"""Dependency-free result objects shared by Ayase clients."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping


def _convert(value: Any, key: str = "") -> Any:
    if isinstance(value, dict):
        if key == "quality_metrics":
            return QualityMetrics(value)
        return RemoteModel(value)
    if isinstance(value, list):
        return [_convert(item) for item in value]
    if key in {"path", "reference_path", "source_file"} and value is not None:
        return Path(value)
    return value


def _dump(value: Any, mode: str) -> Any:
    if isinstance(value, RemoteModel):
        return value.model_dump(mode=mode)
    if isinstance(value, list):
        return [_dump(item, mode) for item in value]
    if isinstance(value, Path) and mode == "json":
        return str(value)
    return value


class RemoteModel(Mapping[str, Any]):
    """Small model with attribute access and common Pydantic-compatible methods."""

    def __init__(self, data: Dict[str, Any]) -> None:
        object.__setattr__(self, "_data", {key: _convert(value, key) for key, value in data.items()})

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            self._data[name] = _convert(value, name)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        values = ", ".join(f"{key}={value!r}" for key, value in self._data.items())
        return f"{type(self).__name__}({values})"

    def model_dump(self, *, mode: str = "python", **_: Any) -> Dict[str, Any]:
        return {key: _dump(value, mode) for key, value in self._data.items()}

    @classmethod
    def model_validate(cls, value: Any) -> "RemoteModel":
        if isinstance(value, cls):
            return value
        if not isinstance(value, dict):
            raise TypeError(f"{cls.__name__} requires a mapping")
        return cls(value)

    def model_copy(self, *, deep: bool = False, **_: Any) -> "RemoteModel":
        data = self.model_dump()
        return type(self)(copy.deepcopy(data) if deep else data.copy())


class QualityMetrics(RemoteModel):
    _NON_METRIC_FIELDS = frozenset({"metric_backends"})

    def non_null_metrics(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in self.model_dump().items()
            if value is not None and key not in self._NON_METRIC_FIELDS
        }

    def non_null_count(self) -> int:
        return len(self.non_null_metrics())

    def to_grouped_dict(self) -> Dict[str, Dict[str, Any]]:
        metrics = self.non_null_metrics()
        return {"remote": metrics} if metrics else {}

    def summary(self) -> str:
        count = self.non_null_count()
        return f"{count} metrics (remote={count})" if count else "0 metrics"


class Sample(RemoteModel):
    @property
    def is_valid(self) -> bool:
        return not any(getattr(issue, "severity", None) == "error" for issue in self.validation_issues)

    @property
    def width(self) -> Any:
        metadata = self.video_metadata or self.image_metadata
        return getattr(metadata, "width", None) if metadata else None

    @property
    def height(self) -> Any:
        metadata = self.video_metadata or self.image_metadata
        return getattr(metadata, "height", None) if metadata else None


class DatasetStats(RemoteModel):
    pass

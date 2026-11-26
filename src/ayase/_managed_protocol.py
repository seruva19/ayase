"""Internal wire format shared by the managed bridge and its worker."""

import json
from struct import Struct
from typing import Any, BinaryIO, Dict, Mapping

from .models import QualityMetrics, Sample

_NULL_METRIC_DEFAULTS = tuple(
    name for name, field in QualityMetrics.model_fields.items() if field.default is None
)
_FRAME_LENGTH = Struct("!I")


def write_message(stream: BinaryIO, message: Mapping[str, Any]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    stream.write(_FRAME_LENGTH.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def read_message(stream: BinaryIO) -> Dict[str, Any]:
    header = stream.read(_FRAME_LENGTH.size)
    if len(header) != _FRAME_LENGTH.size:
        raise EOFError("Managed worker message header ended early")
    (size,) = _FRAME_LENGTH.unpack(header)
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("Managed worker message body ended early")
    message = json.loads(payload)
    if not isinstance(message, dict):
        raise ValueError("Managed worker message must be an object")
    return message


def dump_sample(sample: Sample) -> Dict[str, Any]:
    """Omit null metric defaults that the receiver restores on validation."""
    metrics = sample.quality_metrics
    if metrics is None or metrics._legacy_lip_sync_protocol is not None:
        return sample.model_dump(mode="json")
    excluded = {name for name in _NULL_METRIC_DEFAULTS if getattr(metrics, name) is None}
    return sample.model_dump(mode="json", exclude={"quality_metrics": excluded})

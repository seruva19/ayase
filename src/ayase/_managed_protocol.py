"""Internal wire format shared by the managed bridge and its worker."""

from typing import Any, Dict

from .models import QualityMetrics, Sample

_NULL_METRIC_DEFAULTS = tuple(
    name for name, field in QualityMetrics.model_fields.items() if field.default is None
)


def dump_sample(sample: Sample) -> Dict[str, Any]:
    """Omit null metric defaults that the receiver restores on validation."""
    metrics = sample.quality_metrics
    if metrics is None or metrics._legacy_lip_sync_protocol is not None:
        return sample.model_dump(mode="json")
    excluded = {name for name in _NULL_METRIC_DEFAULTS if getattr(metrics, name) is None}
    return sample.model_dump(mode="json", exclude={"quality_metrics": excluded})

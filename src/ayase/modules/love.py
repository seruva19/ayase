"""Import LOVE perception and correspondence predictions."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.modules._result_adapter_utils import _first_float, _read_csv, _sample_keys, _index_rows

logger = logging.getLogger(__name__)

class LOVEResultModule(PipelineModule):
    """Import raw predictions produced by the LOVE inference scripts."""

    name = "love_results"
    description = "LOVE perception and text-video correspondence result adapter"
    default_config = {
        "perception_results_path": None,
        "correspondence_results_path": None,
    }
    models = [
        {
            "id": "anonymousdb/LOVE-Perception",
            "type": "huggingface",
            "task": "LOVE video perception regressor",
            "auto_download": False,
            "notes": "Run with the upstream LOVE repository; software license not published",
        },
        {
            "id": "anonymousdb/LOVE-Correspondence",
            "type": "huggingface",
            "task": "LOVE text-video correspondence regressor",
            "auto_download": False,
            "notes": "Run with the upstream LOVE repository; software license not published",
        },
    ]
    metric_info = {
        "love_perception_score": "Raw LOVE perception prediction (higher=better)",
        "love_correspondence_score": "Raw LOVE correspondence prediction (higher=better)",
    }
    metric_groups = {
        "love_perception_score": "nr_quality",
        "love_correspondence_score": "alignment",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._perception: Dict[str, Dict[str, Any]] = {}
        self._correspondence: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._backend: Optional[str] = None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for config_key, target in (
            ("perception_results_path", "_perception"),
            ("correspondence_results_path", "_correspondence"),
        ):
            raw_path = self.config.get(config_key)
            if not raw_path:
                continue
            path = Path(raw_path)
            try:
                rows = _read_csv(path)
                setattr(self, target, _index_rows(rows, ("video_name", "video", "path")))
            except Exception as exc:
                logger.warning("LOVE result import failed for %s: %s", path, exc)

    def process(self, sample: Sample) -> Sample:
        self._load()
        perception = next((self._perception[k] for k in _sample_keys(sample) if k in self._perception), None)
        correspondence = next(
            (self._correspondence[k] for k in _sample_keys(sample) if k in self._correspondence),
            None,
        )
        p_score = _first_float(perception or {}, ("pred_score", "score"))
        c_score = _first_float(correspondence or {}, ("pred_score", "score"))
        if p_score is None and c_score is None:
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.love_perception_score = p_score
        sample.quality_metrics.love_correspondence_score = c_score
        self._backend = "imported_results"
        return sample



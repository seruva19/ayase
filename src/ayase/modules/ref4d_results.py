"""Import Ref4D-VideoBench dimension predictions."""

import logging
from pathlib import Path
from statistics import fmean
from typing import Any, Dict, Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.modules._result_adapter_utils import _first_float, _index_rows, _read_csv, _sample_keys

logger = logging.getLogger(__name__)

class Ref4DResultModule(PipelineModule):
    """Import the four Ref4D-VideoBench summary CSVs."""

    name = "ref4d_results"
    description = "Ref4D semantic, event, motion, and world result adapter"
    default_config = {
        "semantic_results_path": None,
        "event_results_path": None,
        "motion_results_path": None,
        "world_results_path": None,
    }
    models = [
        {
            "id": "TAILab-W/Ref4D-VideoBench@6f79f08b359053f2697e1b91b9e38be29baf4d7e",
            "type": "other",
            "url": "https://github.com/TAILab-W/Ref4D-VideoBench",
            "task": "four-dimensional video evaluator",
            "auto_download": False,
            "notes": "Apache-2.0; run its dimension-specific environments separately",
        }
    ]
    metric_info = {
        "ref4d_semantic_score": "Ref4D semantic score (0-100)",
        "ref4d_event_score": "Ref4D event-temporal score (0-100)",
        "ref4d_motion_score": "Ref4D motion-dynamics score (0-100)",
        "ref4d_world_score": "Ref4D world-knowledge score",
        "ref4d_overall_score": "Arithmetic mean of available Ref4D dimensions",
    }
    metric_groups = {
        "ref4d_semantic_score": "alignment",
        "ref4d_event_score": "temporal",
        "ref4d_motion_score": "motion",
        "ref4d_world_score": "alignment",
        "ref4d_overall_score": "nr_quality",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._indexes: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._loaded = False
        self._backend: Optional[str] = None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        for dimension in ("semantic", "event", "motion", "world"):
            raw_path = self.config.get(f"{dimension}_results_path")
            if not raw_path:
                continue
            path = Path(raw_path)
            try:
                self._indexes[dimension] = _index_rows(
                    _read_csv(path), ("sample_id", "video_name", "video", "path")
                )
            except Exception as exc:
                logger.warning("Ref4D %s result import failed for %s: %s", dimension, path, exc)

    def process(self, sample: Sample) -> Sample:
        self._load()
        scores: Dict[str, float] = {}
        for dimension, index in self._indexes.items():
            row = next((index[k] for k in _sample_keys(sample) if k in index), None)
            if row is None:
                continue
            fields = (
                f"{dimension}_score_0_100",
                f"{dimension}_score",
            )
            score = _first_float(row, fields)
            if score is not None:
                scores[dimension] = score
        if not scores:
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.ref4d_semantic_score = scores.get("semantic")
        sample.quality_metrics.ref4d_event_score = scores.get("event")
        sample.quality_metrics.ref4d_motion_score = scores.get("motion")
        sample.quality_metrics.ref4d_world_score = scores.get("world")
        sample.quality_metrics.ref4d_overall_score = float(fmean(scores.values()))
        self._backend = "imported_results"
        return sample



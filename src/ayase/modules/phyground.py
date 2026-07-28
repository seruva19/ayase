"""Import PhyGround and PhyJudge structured predictions."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ayase.modules._result_adapter_utils import _first_float, _index_rows, _sample_keys

logger = logging.getLogger(__name__)

class PhyGroundResultModule(PipelineModule):
    """Import PhyGround/PhyJudge structured JSON results."""

    name = "phyground_results"
    description = "PhyGround general and physical-law judge result adapter"
    default_config = {"results_path": None}
    models = [
        {
            "id": "NU-World-Model-Embodied-AI/phyjudge-9B",
            "type": "huggingface",
            "task": "PhyGround physical-law video judge LoRA",
            "auto_download": False,
        }
    ]
    metric_info = {
        "phyground_spatial_alignment_score": "PhyGround SA score (1-5)",
        "phyground_prompt_temporal_validity_score": "PhyGround PTV score (1-5)",
        "phyground_persistence_score": "PhyGround persistence score (1-5)",
        "phyground_general_score": "PhyGround general-dimension mean (1-5)",
        "phyground_physical_score": "PhyGround applicable physical-law mean (1-5)",
        "phyground_physical_coverage": "Fraction of requested physical laws scored (0-1)",
    }
    metric_groups = {
        "phyground_spatial_alignment_score": "alignment",
        "phyground_prompt_temporal_validity_score": "temporal",
        "phyground_persistence_score": "temporal",
        "phyground_general_score": "nr_quality",
        "phyground_physical_score": "nr_quality",
        "phyground_physical_coverage": "nr_quality",
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._results: Dict[str, Dict[str, Any]] = {}
        self._loaded = False
        self._backend: Optional[str] = None

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        raw_path = self.config.get("results_path")
        if not raw_path:
            return
        path = Path(raw_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("results", payload) if isinstance(payload, dict) else payload
            if not isinstance(rows, list):
                raise ValueError("expected a result list or an object containing 'results'")
            self._results = _index_rows(rows, ("video", "video_name", "path"))
        except Exception as exc:
            logger.warning("PhyGround result import failed for %s: %s", path, exc)

    def process(self, sample: Sample) -> Sample:
        self._load()
        row = next((self._results[k] for k in _sample_keys(sample) if k in self._results), None)
        if row is None:
            return sample
        physical = row.get("physical") if isinstance(row.get("physical"), dict) else {}
        values = {
            "phyground_spatial_alignment_score": _first_float(row, ("SA",)),
            "phyground_prompt_temporal_validity_score": _first_float(row, ("PTV",)),
            "phyground_persistence_score": _first_float(row, ("persistence",)),
            "phyground_general_score": _first_float(row, ("general_avg",)),
            "phyground_physical_score": _first_float(physical, ("avg",)),
            "phyground_physical_coverage": _first_float(physical, ("coverage",)),
        }
        if not any(value is not None for value in values.values()):
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.phyground_spatial_alignment_score = values[
            "phyground_spatial_alignment_score"
        ]
        sample.quality_metrics.phyground_prompt_temporal_validity_score = values[
            "phyground_prompt_temporal_validity_score"
        ]
        sample.quality_metrics.phyground_persistence_score = values[
            "phyground_persistence_score"
        ]
        sample.quality_metrics.phyground_general_score = values["phyground_general_score"]
        sample.quality_metrics.phyground_physical_score = values["phyground_physical_score"]
        sample.quality_metrics.phyground_physical_coverage = values[
            "phyground_physical_coverage"
        ]
        self._backend = "imported_results"
        return sample

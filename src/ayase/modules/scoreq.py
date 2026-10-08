"""SCOREQ no-reference speech quality MOS prediction.

Uses the real SCOREQ model (alessandroragano/scoreq) in no-reference mode.
No proxy metric is substituted when the ``scoreq`` package is missing. When
the model is unavailable, ``scoreq_score`` is left ``None``.

The configured SCOREQ domain selects the published natural- or synthetic-speech
checkpoint. SCOREQ returns its native MOS prediction without normalization or
clamping (higher is better).
"""

import logging
from importlib import metadata, util
from pathlib import Path
from typing import Optional

from ayase.config import download_model_file
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_SUPPORTED_SCOREQ_VERSIONS = {"1.0.0", "1.0.1"}


def _load_native_scoreq_class():
    """Load SCOREQ's implementation without executing its eager package singleton."""
    distribution = metadata.distribution("scoreq")
    version = distribution.version
    if version not in _SUPPORTED_SCOREQ_VERSIONS:
        raise RuntimeError(
            f"Unsupported SCOREQ version {version!r}; expected one of "
            f"{sorted(_SUPPORTED_SCOREQ_VERSIONS)}"
        )

    relative_source = Path("scoreq") / "scoreq.py"
    if distribution.files is None:
        raise RuntimeError(f"SCOREQ {version} distribution has no installed-file manifest")
    distribution_paths = {str(item).replace("\\", "/") for item in distribution.files}
    if "scoreq/scoreq.py" not in distribution_paths:
        raise RuntimeError(f"SCOREQ {version} distribution does not contain {relative_source}")
    source_path = Path(distribution.locate_file(relative_source)).resolve()
    if not source_path.is_file():
        raise RuntimeError(f"SCOREQ implementation file is missing: {source_path}")

    module_name = f"_ayase_native_scoreq_{version.replace('.', '_')}"
    spec = util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load SCOREQ implementation from {source_path}")
    native_module = util.module_from_spec(spec)
    spec.loader.exec_module(native_module)
    native_class = getattr(native_module, "Scoreq", None)
    if not isinstance(native_class, type):
        raise RuntimeError(f"SCOREQ {version} implementation does not export Scoreq")
    return native_class, version, str(source_path)


class SCOREQModule(PipelineModule):
    name = "scoreq"
    provenance = "published"
    sources = {
        "scoreq_score": "SCOREQ (Ragano et al., NeurIPS 2024) — https://github.com/alessandroragano/scoreq",
    }
    description = "SCOREQ domain-dependent no-reference speech quality MOS"
    default_config = {
        "sample_rate": 16000,
        "data_domain": "natural",
        "warning_threshold": None,
    }
    models = [
        {
            "id": "scoreq",
            "type": "pip_package",
            "install": "pip install scoreq==1.0.0",
            "task": "Native SCOREQ ONNX inference runtime",
            "notes": "1.0.0 supports NumPy <2; 1.0.1 requires NumPy >=2",
        },
        {
            "id": "adapt_nr_telephone.onnx",
            "type": "other",
            "url": "https://zenodo.org/records/15739280/files/adapt_nr_telephone.onnx",
            "task": "SCOREQ no-reference MOS for data_domain=natural",
            "notes": "SCOREQ 1.0.0/1.0.1; cached under models_dir/scoreq/onnx-models",
        },
        {
            "id": "adapt_nr_synthetic.onnx",
            "type": "other",
            "url": "https://zenodo.org/records/15739280/files/adapt_nr_synthetic.onnx",
            "task": "SCOREQ no-reference MOS for data_domain=synthetic",
            "notes": "SCOREQ 1.0.0/1.0.1; cached under models_dir/scoreq/onnx-models",
        },
    ]
    metric_info = {
        "scoreq_score": "SCOREQ no-reference speech quality MOS (native output, higher=better)",
    }
    metric_groups = {
        "scoreq_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.data_domain = self.config.get("data_domain", "natural")
        self.warning_threshold = self.config.get("warning_threshold")
        self.models_dir = self.config.get("models_dir", "models")
        self._backend = None
        self._backend_version = None
        self._backend_source = None
        self._model = None

    def setup(self) -> None:
        try:
            models_dir = self.models_dir
            native_scoreq, native_version, native_source = _load_native_scoreq_class()

            class AyaseCachedScoreq(native_scoreq):
                """Native SCOREQ with only its hard-coded cache path redirected."""

                def _download_model(self, filename, url, cache_dir_name):
                    return str(
                        download_model_file(
                            f"scoreq/{cache_dir_name}/{filename}", url, models_dir
                        )
                    )

            self._model = AyaseCachedScoreq(data_domain=self.data_domain, mode="nr")
            self._backend = "scoreq"
            self._backend_version = native_version
            self._backend_source = native_source
            logger.info("SCOREQ initialised with scoreq package (nr, %s)", self.data_domain)
        except ImportError:
            self._backend = "unavailable"
            logger.warning("SCOREQ package not installed; scoreq_score left unset.")
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("SCOREQ setup failed (%s); scoreq_score left unset.", e)

    def process(self, sample: Sample) -> Sample:
        if self._backend != "scoreq" or self._model is None:
            return sample

        try:
            score = self._model.predict(test_path=str(sample.path), ref_path=None)
            if score is None:
                return sample
            score = float(score)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.scoreq_score = score

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low SCOREQ speech quality: {score:.3f}",
                        details={"scoreq_score": score},
                    )
                )
        except Exception as e:
            logger.warning("SCOREQ failed for %s: %s", sample.path, e)
        return sample

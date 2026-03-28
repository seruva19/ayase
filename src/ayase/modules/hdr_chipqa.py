"""HDR-ChipQA module.

This module runs the HDR-ChipQA feature extractor and LIVE-HDR SVR from the
bundled source tree. The pipeline expects raw YUV HDR input.
"""

import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_HDR_CHIPQA_BOOTSTRAP = r"""
import importlib
import runpy
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))
chipqa_pkg = importlib.import_module("chipqa")
sys.modules["ChipQA"] = chipqa_pkg
sys.modules["ChipQA.niqe"] = importlib.import_module("chipqa.niqe")
sys.modules["ChipQA.save_stats"] = importlib.import_module("chipqa.save_stats")
sys.argv = [
    str(repo / "hdr_chipqa.py"),
    "--input_file",
    sys.argv[2],
    "--results_file",
    sys.argv[3],
    "--width",
    sys.argv[4],
    "--height",
    sys.argv[5],
    "--bit_depth",
    sys.argv[6],
    "--color_space",
    sys.argv[7],
]
runpy.run_path(str(repo / "hdr_chipqa.py"), run_name="__main__")
"""


def _has_required_paths(base_dir: Path, required_paths: list[str]) -> bool:
    return all((base_dir / relative_path).exists() for relative_path in required_paths)


def _parse_last_float(text: str) -> float | None:
    matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _python_executable(config_value: str | None) -> str:
    return str(config_value) if config_value else sys.executable


def _run_python_inline(
    python_bin: str,
    code: str,
    args: list[str],
    cwd: Path,
    timeout_sec: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [python_bin, "-c", code, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )


def _third_party_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1].joinpath("third_party", *parts)


class HDRChipQAModule(PipelineModule):
    name = "hdr_chipqa"
    description = "HDR-ChipQA no-reference HDR video quality via official feature extractor and LIVE-HDR SVR"
    default_config = {
        "repo_path": None,
        "python_executable": None,
        "timeout_sec": 1800,
        "width": 3840,
        "height": 2160,
        "bit_depth": 10,
        "color_space": "BT2020",
        "warning_threshold": None,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.repo_path = self.config.get("repo_path")
        self.python_bin = _python_executable(self.config.get("python_executable"))
        self.timeout_sec = int(self.config.get("timeout_sec", 1800))
        self.width = int(self.config.get("width", 3840))
        self.height = int(self.config.get("height", 2160))
        self.bit_depth = int(self.config.get("bit_depth", 10))
        self.color_space = str(self.config.get("color_space", "BT2020"))
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = None
        self._repo_dir: Path | None = None

    def setup(self) -> None:
        try:
            import cv2  # noqa: F401
            import joblib  # noqa: F401
            import matplotlib  # noqa: F401
            import numba  # noqa: F401
            import numpy  # noqa: F401
            import scipy  # noqa: F401
            import sklearn  # noqa: F401
        except ImportError as exc:
            logger.warning("HDR-ChipQA unavailable: missing dependency: %s", exc)
            return

        repo_dir = Path(self.repo_path).resolve() if self.repo_path else _third_party_path("hdr_chipqa")
        required = [
            "hdr_chipqa.py",
            "testing.py",
            "hdrchipqa_livehdr_trained_svr.z",
            "hdrchipqa_livehdr_fitted_scaler.z",
            "utils/colour_utils.py",
            "utils/hdr_utils.py",
            "chipqa/__init__.py",
            "chipqa/niqe.py",
            "chipqa/save_stats.py",
        ]
        if not _has_required_paths(repo_dir, required):
            logger.warning("HDR-ChipQA unavailable: required source files are missing.")
            return

        self._repo_dir = repo_dir
        self._backend = "hdr_chipqa"

    def process(self, sample: Sample) -> Sample:
        if self._backend != "hdr_chipqa" or self._repo_dir is None:
            return sample
        if not sample.is_video or sample.path.suffix.lower() != ".yuv":
            return sample

        try:
            score = self._run_hdr_chipqa(sample.path)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.hdr_chipqa_score = score

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low HDR-ChipQA score: {score:.3f}",
                        details={"hdr_chipqa_score": score},
                    )
                )
        except Exception as exc:
            logger.warning("HDR-ChipQA failed for %s: %s", sample.path, exc)

        return sample

    def _run_hdr_chipqa(self, sample_path: Path) -> float | None:
        assert self._repo_dir is not None
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            feature_path = tmp_dir / "hdr_chipqa_features.z"

            extract = _run_python_inline(
                self.python_bin,
                _HDR_CHIPQA_BOOTSTRAP,
                [
                    str(self._repo_dir),
                    str(sample_path.resolve()),
                    str(feature_path.resolve()),
                    str(self.width),
                    str(self.height),
                    str(self.bit_depth),
                    self.color_space,
                ],
                cwd=self._repo_dir,
                timeout_sec=self.timeout_sec,
            )
            if extract.returncode != 0:
                logger.warning("HDR-ChipQA feature extraction failed: %s", extract.stderr.strip())
                return None
            if not feature_path.exists():
                logger.warning("HDR-ChipQA did not produce feature file for %s", sample_path.name)
                return None

            predict = subprocess.run(
                [self.python_bin, "testing.py", str(feature_path)],
                cwd=str(self._repo_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            if predict.returncode != 0:
                logger.warning("HDR-ChipQA prediction failed: %s", predict.stderr.strip())
                return None
            return _parse_last_float(predict.stdout)

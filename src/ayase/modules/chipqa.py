"""ChipQA module.

This module runs the ChipQA feature extractor and LIVE-Livestream SVR from the
bundled source tree.
"""

import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_CHIPQA_BOOTSTRAP = r"""
import importlib
import runpy
import sys
from pathlib import Path

repo = Path(sys.argv[1])
input_folder = sys.argv[2]
results_folder = sys.argv[3]
sys.path.insert(0, str(repo.parent))
sys.modules["niqe"] = importlib.import_module("chipqa.niqe")
sys.modules["save_stats"] = importlib.import_module("chipqa.save_stats")
sys.argv = [str(repo / "chipqa.py"), input_folder, results_folder]
runpy.run_path(str(repo / "chipqa.py"), run_name="__main__")
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


class ChipQAModule(PipelineModule):
    name = "chipqa"
    description = "ChipQA no-reference video quality via official feature extractor and LIVE-Livestream SVR"
    default_config = {
        "repo_path": None,
        "python_executable": None,
        "timeout_sec": 1800,
        "warning_threshold": None,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.repo_path = self.config.get("repo_path")
        self.python_bin = _python_executable(self.config.get("python_executable"))
        self.timeout_sec = int(self.config.get("timeout_sec", 1800))
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
            logger.warning("ChipQA unavailable: missing dependency: %s", exc)
            return

        repo_dir = (
            Path(self.repo_path).resolve()
            if self.repo_path
            else _third_party_path("hdr_chipqa", "chipqa")
        )
        required = [
            "chipqa.py",
            "testing.py",
            "LIVE_Livestream_trained_svr.z",
            "LIVE_Livestream_fitted_scaler.z",
            "frames_modelparameters.mat",
            "niqe.py",
            "save_stats.py",
        ]
        if not _has_required_paths(repo_dir, required):
            logger.warning("ChipQA unavailable: required source files are missing.")
            return

        self._repo_dir = repo_dir
        self._backend = "chipqa"

    def process(self, sample: Sample) -> Sample:
        if self._backend != "chipqa" or self._repo_dir is None:
            return sample
        if not sample.is_video or sample.path.suffix.lower() != ".mp4":
            return sample

        try:
            score = self._run_chipqa(sample.path)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.chipqa_score = score

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low ChipQA score: {score:.3f}",
                        details={"chipqa_score": score},
                    )
                )
        except Exception as exc:
            logger.warning("ChipQA failed for %s: %s", sample.path, exc)

        return sample

    def _run_chipqa(self, sample_path: Path) -> float | None:
        assert self._repo_dir is not None
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            input_dir = tmp_dir / "input"
            output_dir = tmp_dir / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            staged_path = input_dir / sample_path.name
            shutil.copy2(sample_path, staged_path)

            extract = _run_python_inline(
                self.python_bin,
                _CHIPQA_BOOTSTRAP,
                [str(self._repo_dir), str(input_dir), str(output_dir)],
                cwd=self._repo_dir.parent,
                timeout_sec=self.timeout_sec,
            )
            if extract.returncode != 0:
                logger.warning("ChipQA feature extraction failed: %s", extract.stderr.strip())
                return None

            feature_path = output_dir / f"{sample_path.stem}.z"
            if not feature_path.exists():
                logger.warning("ChipQA did not produce feature file for %s", sample_path.name)
                return None

            predict = subprocess.run(
                [self.python_bin, "testing.py", str(feature_path)],
                cwd=str(self._repo_dir),
                capture_output=True,
                text=True,
                timeout=self.timeout_sec,
            )
            if predict.returncode != 0:
                logger.warning("ChipQA prediction failed: %s", predict.stderr.strip())
                return None
            return _parse_last_float(predict.stdout)

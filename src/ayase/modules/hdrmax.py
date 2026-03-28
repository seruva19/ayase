"""HDRMAX module.

This module runs the HDRMAX scripts and model files from the bundled source
tree.
"""

import csv
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _has_required_paths(base_dir: Path, required_paths: list[str]) -> bool:
    return all((base_dir / relative_path).exists() for relative_path in required_paths)


def _python_executable(config_value: str | None) -> str:
    return str(config_value) if config_value else sys.executable


def _third_party_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1].joinpath("third_party", *parts)


class HDRMAXModule(PipelineModule):
    name = "hdrmax"
    description = "HDRMAX full-reference HDR video quality via official HDRMAX feature and prediction scripts"
    default_config = {
        "repo_path": None,
        "python_executable": None,
        "mode": "hdrvmaf",
        "timeout_sec": 3600,
        "ffmpeg_bin": "ffmpeg",
        "njobs": 1,
        "fps": None,
        "warning_threshold": None,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.repo_path = self.config.get("repo_path")
        self.python_bin = _python_executable(self.config.get("python_executable"))
        self.mode = str(self.config.get("mode", "hdrvmaf"))
        self.timeout_sec = int(self.config.get("timeout_sec", 3600))
        self.ffmpeg_bin = str(self.config.get("ffmpeg_bin", "ffmpeg"))
        self.njobs = int(self.config.get("njobs", 1))
        self.fps_override = self.config.get("fps")
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = None
        self._repo_dir: Path | None = None

    def setup(self) -> None:
        try:
            import colour  # noqa: F401
            import joblib  # noqa: F401
            import matplotlib  # noqa: F401
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import pyrtools  # noqa: F401
            import pywt  # noqa: F401
            import scipy  # noqa: F401
            import skimage  # noqa: F401
        except ImportError as exc:
            logger.warning("HDRMAX unavailable: missing dependency: %s", exc)
            return

        required_paths = [
            "hdrvmaf_features.py",
            "extract_ssim.py",
            "extract_msssim.py",
            "predict_unified.py",
        ]
        if self.mode == "hdrvmaf":
            required_paths.extend(
                [
                    "models/svr/model_svr_livehdr.pkl",
                    "models/scaler/model_scaler_livehdr.pkl",
                ]
            )
        elif self.mode == "ssim-hdrmax":
            required_paths.extend(
                [
                    "models/svr/ssim_svr.pkl",
                    "models/scaler/ssim_scaler.pkl",
                ]
            )
        elif self.mode == "msssim-hdrmax":
            required_paths.extend(
                [
                    "models/svr/msssim_svr.pkl",
                    "models/scaler/msssim_scaler.pkl",
                ]
            )
        else:
            logger.warning("HDRMAX unavailable: unsupported mode %s", self.mode)
            return

        repo_dir = Path(self.repo_path).resolve() if self.repo_path else _third_party_path("hdrmax")
        if not _has_required_paths(repo_dir, required_paths):
            logger.warning("HDRMAX unavailable: required source files are missing.")
            return

        self._repo_dir = repo_dir
        self._backend = "hdrmax"

    def process(self, sample: Sample) -> Sample:
        if self._backend != "hdrmax" or self._repo_dir is None:
            return sample
        if not sample.is_video or sample.reference_path is None:
            return sample

        reference_path = Path(sample.reference_path)
        if not reference_path.exists():
            return sample

        try:
            score = self._run_hdrmax(sample, reference_path)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.hdrmax_score = score

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low HDRMAX score: {score:.3f}",
                        details={"hdrmax_score": score},
                    )
                )
        except Exception as exc:
            logger.warning("HDRMAX failed for %s: %s", sample.path, exc)

        return sample

    def _run_hdrmax(self, sample: Sample, reference_path: Path) -> float | None:
        assert self._repo_dir is not None
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            input_dir = tmp_dir / "input"
            feature_dir = tmp_dir / "features"
            input_dir.mkdir()
            feature_dir.mkdir()

            distorted_yuv = input_dir / "dist.yuv"
            reference_yuv = input_dir / "ref.yuv"
            self._prepare_yuv(sample.path, distorted_yuv)
            self._prepare_yuv(reference_path, reference_yuv)

            fps = self._resolve_fps(sample)
            if fps is None:
                logger.warning("HDRMAX requires FPS metadata or explicit config.")
                return None

            info_csv = tmp_dir / "info.csv"
            with info_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["", "encoded_yuv", "fps", "start", "end", "ref_yuv"])
                writer.writerow([0, distorted_yuv.name, fps, 0, 0, reference_yuv.name])

            commands = self._build_commands(input_dir, feature_dir, info_csv, tmp_dir / "pred.csv")
            for command in commands:
                result = subprocess.run(
                    command,
                    cwd=str(self._repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
                if result.returncode != 0:
                    logger.warning("HDRMAX command failed: %s", result.stderr.strip())
                    return None

            pred_csv = tmp_dir / "pred.csv"
            if not pred_csv.exists():
                logger.warning("HDRMAX prediction file was not created.")
                return None

            with pred_csv.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    pred_value = row.get("pred")
                    if pred_value is not None:
                        return float(pred_value)
            return None

    def _prepare_yuv(self, source_path: Path, output_path: Path) -> None:
        if source_path.suffix.lower() == ".yuv":
            shutil.copy2(source_path, output_path)
            return

        result = subprocess.run(
            [
                self.ffmpeg_bin,
                "-y",
                "-i",
                str(source_path),
                "-vf",
                "scale=3840x2160",
                "-pix_fmt",
                "yuv420p10le",
                str(output_path),
            ],
            capture_output=True,
            text=True,
            timeout=self.timeout_sec,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "ffmpeg conversion failed")

    def _resolve_fps(self, sample: Sample) -> int | None:
        if self.fps_override is not None:
            return int(self.fps_override)
        if sample.video_metadata and sample.video_metadata.fps:
            return max(1, int(round(sample.video_metadata.fps)))
        cap = cv2.VideoCapture(str(sample.path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
        finally:
            cap.release()
        if fps and fps > 0:
            return max(1, int(round(fps)))
        return None

    def _build_commands(
        self,
        input_dir: Path,
        feature_dir: Path,
        info_csv: Path,
        pred_csv: Path,
    ) -> list[list[str]]:
        commands: list[list[str]] = []
        if self.mode == "hdrvmaf":
            commands.extend(
                [
                    [
                        self.python_bin,
                        "hdrvmaf_features.py",
                        str(input_dir),
                        str(feature_dir),
                        str(info_csv),
                        "--space",
                        "ycbcr",
                        "--nonlinear",
                        "local_m_exp",
                        "--channel",
                        "0",
                        "--vif",
                        "--dlm",
                        "--njobs",
                        str(self.njobs),
                        "--frame_range",
                        "all",
                    ],
                    [
                        self.python_bin,
                        "hdrvmaf_features.py",
                        str(input_dir),
                        str(feature_dir),
                        str(info_csv),
                        "--space",
                        "ycbcr",
                        "--nonlinear",
                        "none",
                        "--parameter",
                        "2",
                        "--channel",
                        "0",
                        "--vif",
                        "--dlm",
                        "--njobs",
                        str(self.njobs),
                        "--frame_range",
                        "all",
                    ],
                    [
                        self.python_bin,
                        "predict_unified.py",
                        str(feature_dir),
                        str(pred_csv),
                        "--model",
                        "VMAF",
                    ],
                ]
            )
        elif self.mode == "ssim-hdrmax":
            commands.extend(
                [
                    [
                        self.python_bin,
                        "hdrvmaf_features.py",
                        str(input_dir),
                        str(feature_dir),
                        str(info_csv),
                        "--space",
                        "ycbcr",
                        "--nonlinear",
                        "local_m_exp",
                        "--channel",
                        "0",
                        "--vif",
                        "--dlm",
                        "--njobs",
                        str(self.njobs),
                        "--frame_range",
                        "all",
                    ],
                    [
                        self.python_bin,
                        "extract_ssim.py",
                        str(input_dir),
                        str(feature_dir),
                        str(info_csv),
                        "--space",
                        "ycbcr",
                        "--channel",
                        "0",
                        "--njobs",
                        str(self.njobs),
                        "--frame_range",
                        "all",
                    ],
                    [
                        self.python_bin,
                        "predict_unified.py",
                        str(feature_dir),
                        str(pred_csv),
                        "--model",
                        "SSIM",
                    ],
                ]
            )
        else:
            commands.extend(
                [
                    [
                        self.python_bin,
                        "hdrvmaf_features.py",
                        str(input_dir),
                        str(feature_dir),
                        str(info_csv),
                        "--space",
                        "ycbcr",
                        "--nonlinear",
                        "local_m_exp",
                        "--channel",
                        "0",
                        "--vif",
                        "--dlm",
                        "--njobs",
                        str(self.njobs),
                        "--frame_range",
                        "all",
                    ],
                    [
                        self.python_bin,
                        "extract_msssim.py",
                        str(input_dir),
                        str(feature_dir),
                        str(info_csv),
                        "--space",
                        "ycbcr",
                        "--channel",
                        "0",
                        "--njobs",
                        str(self.njobs),
                        "--frame_range",
                        "all",
                    ],
                    [
                        self.python_bin,
                        "predict_unified.py",
                        str(feature_dir),
                        str(pred_csv),
                        "--model",
                        "MSSSIM",
                    ],
                ]
            )
        return commands

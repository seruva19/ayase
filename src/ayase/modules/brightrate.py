"""BrightRate module.

This module runs the BrightRate inference script from the bundled BrightVQ
source tree.
"""

import csv
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2

from ayase.config import download_model_file
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_LOCAL_AYASE_MODELS_REPO = Path(r"H:\models\ayase-models")
_BRIGHTRATE_MODEL_URL = (
    "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/brightvq/brightrate_brightvq.pt"
)
_CONTRIQUE_MODEL_URL = (
    "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/brightvq/CONTRIQUE_checkpoint25.tar"
)
_HDR_NIQE_PARAMS_URL = (
    "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/brightvq/frames_modelparameters.mat"
)
_CLIP_VIT_B32_URL = (
    "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/brightvq/ViT-B-32.safetensors"
)
_CLIP_VIT_L14_URL = (
    "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/brightvq/ViT-L-14.safetensors"
)
_CLIPIQA_VITL14_URL = (
    "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/brightvq/CLIPIQA+_ViTL14_512-e66488f2.pth"
)


def _has_required_paths(base_dir: Path, required_paths: list[str]) -> bool:
    return all((base_dir / relative_path).exists() for relative_path in required_paths)


def _python_executable(config_value: str | None) -> str:
    return str(config_value) if config_value else sys.executable


def _third_party_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[1].joinpath("third_party", *parts)


class BrightRateModule(PipelineModule):
    name = "brightrate"
    description = "BrightRate HDR no-reference video quality via official BrightVQ inference script"
    default_config = {
        "repo_path": None,
        "python_executable": None,
        "model_path": None,
        "timeout_sec": 3600,
        "num_frames": 30,
        "num_workers": 1,
        "parallel_level": "video",
        "ffmpeg_path": "",
        "read_yuv": False,
        "warning_threshold": None,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.repo_path = self.config.get("repo_path")
        self.python_bin = _python_executable(self.config.get("python_executable"))
        self.model_path = self.config.get("model_path")
        self.timeout_sec = int(self.config.get("timeout_sec", 3600))
        self.num_frames = int(self.config.get("num_frames", 30))
        self.num_workers = int(self.config.get("num_workers", 1))
        self.parallel_level = str(self.config.get("parallel_level", "video"))
        self.ffmpeg_path = str(self.config.get("ffmpeg_path", ""))
        self.read_yuv = bool(self.config.get("read_yuv", False))
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = None
        self._repo_dir: Path | None = None
        self._model_file: Path | None = None
        self._asset_paths: dict[str, Path] = {}

    def setup(self) -> None:
        try:
            import imageio_ffmpeg  # noqa: F401
            import joblib  # noqa: F401
            import numba  # noqa: F401
            import numpy  # noqa: F401
            import pandas  # noqa: F401
            import pyiqa  # noqa: F401
            import scipy  # noqa: F401
            import sklearn  # noqa: F401
            import torch  # noqa: F401
            import torchvision  # noqa: F401
        except ImportError as exc:
            logger.warning("BrightRate unavailable: missing dependency: %s", exc)
            return

        repo_dir = Path(self.repo_path).resolve() if self.repo_path else _third_party_path("brightvq")
        required = [
            "demo_inference.py",
            "util_hdr_10bit.py",
            "CLIP/clip_feats.py",
            "CLIP/clip/clip.py",
            "CONTRIQUE/contrique_feat.py",
            "HDR/hdr_feat.py",
        ]
        if not _has_required_paths(repo_dir, required):
            logger.warning("BrightRate unavailable: required source files are missing.")
            return

        asset_paths = self._resolve_asset_paths()
        model_file = asset_paths.get("regressor")
        if model_file is None:
            logger.warning("BrightRate unavailable: regressor weights are missing.")
            return

        self._repo_dir = repo_dir
        self._model_file = model_file.resolve()
        self._asset_paths = asset_paths
        self._backend = "brightrate"

    def process(self, sample: Sample) -> Sample:
        if self._backend != "brightrate" or self._repo_dir is None or self._model_file is None:
            return sample
        if not sample.is_video:
            return sample
        if self.read_yuv and sample.path.suffix.lower() != ".yuv":
            return sample
        if not self.read_yuv and sample.path.suffix.lower() != ".mp4":
            return sample

        try:
            score = self._run_brightrate(sample)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.brightrate_score = score

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low BrightRate score: {score:.3f}",
                        details={"brightrate_score": score},
                    )
                )
        except Exception as exc:
            logger.warning("BrightRate failed for %s: %s", sample.path, exc)

        return sample

    def _run_brightrate(self, sample: Sample) -> float | None:
        assert self._repo_dir is not None
        assert self._model_file is not None
        with tempfile.TemporaryDirectory() as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            video_dir = tmp_dir / "videos"
            output_dir = tmp_dir / "output"
            video_dir.mkdir()
            output_dir.mkdir()

            base_name = "sample"
            staged_video = video_dir / f"{base_name}{sample.path.suffix.lower()}"
            shutil.copy2(sample.path, staged_video)

            width, height = self._resolve_dimensions(sample)
            if self.read_yuv and (width is None or height is None):
                logger.warning("BrightRate YUV mode requires width and height metadata.")
                return None

            csv_path = tmp_dir / "sample_videos.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Video", "width", "height", "name"])
                writer.writerow([base_name, width or "", height or "", sample.path.stem])

            command = [
                self.python_bin,
                "demo_inference.py",
                "--model_path",
                str(self._model_file),
                "--dataset_csv",
                str(csv_path),
                "--video_path",
                str(video_dir),
                "--save_path",
                str(output_dir),
                "--parallel_level",
                self.parallel_level,
                "--num_workers",
                str(self.num_workers),
                "--num_frames",
                str(self.num_frames),
                "--ffmpeg_path",
                self.ffmpeg_path,
            ]
            if self.read_yuv:
                command.append("--read_yuv")

            result = subprocess.run(
                command,
                cwd=str(self._repo_dir),
                capture_output=True,
                text=True,
                env=self._build_subprocess_env(),
                timeout=self.timeout_sec,
            )
            if result.returncode != 0:
                logger.warning("BrightRate inference failed: %s", result.stderr.strip())
                return None

            score_file = output_dir / "brightrate" / f"{base_name}_score.txt"
            if not score_file.exists():
                logger.warning("BrightRate score file was not created.")
                return None
            return float(score_file.read_text(encoding="utf-8").strip())

    def _resolve_dimensions(self, sample: Sample) -> tuple[int | None, int | None]:
        if sample.video_metadata is not None:
            return sample.video_metadata.width, sample.video_metadata.height
        cap = cv2.VideoCapture(str(sample.path))
        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
        if width > 0 and height > 0:
            return width, height
        return None, None

    def _resolve_asset_paths(self) -> dict[str, Path]:
        models_dir = str(self.config.get("models_dir", "models"))
        assets: dict[str, Path] = {}
        for key, relative_path, url in [
            ("regressor", "brightvq/brightrate_brightvq.pt", _BRIGHTRATE_MODEL_URL),
            ("contrique", "brightvq/CONTRIQUE_checkpoint25.tar", _CONTRIQUE_MODEL_URL),
            ("hdr_niqe_params", "brightvq/frames_modelparameters.mat", _HDR_NIQE_PARAMS_URL),
            ("clip_vit_b32", "brightvq/ViT-B-32.safetensors", _CLIP_VIT_B32_URL),
            ("clip_vit_l14", "brightvq/ViT-L-14.safetensors", _CLIP_VIT_L14_URL),
            ("clipiqa_vitl14", "brightvq/CLIPIQA+_ViTL14_512-e66488f2.pth", _CLIPIQA_VITL14_URL),
        ]:
            resolved = self._resolve_asset_path(key, relative_path, url, models_dir)
            if resolved is None:
                logger.warning("Failed to resolve BrightRate asset %s", key)
                return {}
            assets[key] = resolved
        return assets

    def _resolve_asset_path(
        self,
        key: str,
        relative_path: str,
        url: str,
        models_dir: str,
    ) -> Path | None:
        config_key = f"{key}_path"
        configured_path = self.config.get(config_key)
        if key == "regressor" and self.model_path:
            configured_path = self.model_path
        if configured_path:
            candidate = Path(configured_path)
            return candidate if candidate.exists() else None
        mirror_candidate = _LOCAL_AYASE_MODELS_REPO / relative_path
        if mirror_candidate.exists():
            return mirror_candidate
        try:
            return download_model_file(relative_path, url, models_dir)
        except Exception as exc:
            logger.warning("Failed to download BrightRate asset %s: %s", key, exc)
            return None

    def _build_subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "AYASE_BRIGHTVQ_CONTRIQUE_MODEL_PATH": str(self._asset_paths["contrique"]),
                "AYASE_BRIGHTVQ_HDR_NIQE_PARAMS_PATH": str(self._asset_paths["hdr_niqe_params"]),
                "AYASE_BRIGHTVQ_CLIP_VIT_B32_PATH": str(self._asset_paths["clip_vit_b32"]),
                "AYASE_BRIGHTVQ_CLIP_VIT_L14_PATH": str(self._asset_paths["clip_vit_l14"]),
                "AYASE_BRIGHTVQ_CLIPIQA_VITL14_PATH": str(self._asset_paths["clipiqa_vitl14"]),
            }
        )
        return env

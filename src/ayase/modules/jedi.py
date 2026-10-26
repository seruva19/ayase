"""Official two-distribution JEDi using the upstream ``videojedi`` package.

JEDi compares generated and reference video distributions in the fine-tuned
V-JEPA feature space with the upstream degree-2 polynomial-kernel MMD. Lower
scores mean closer distributions. This module deliberately remains separate
from Ayase's order-sensitive ``mmd_selfsplit`` diagnostic.
"""

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from ayase.models import Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
_UPSTREAM_REVISION = "e64f3fe8572b89ea904d161def44b419024e31a0"
_MANAGED_REQUIREMENTS = (
    "videojedi==1.1.0",
    "vjepa==0.1.2",
    "torch==2.4.1+cu124",
    "torchvision==0.19.1+cu124",
    "pandas==2.2.2",
    "numpy==1.26.4",
)
_PYTORCH_INDEX = "https://download.pytorch.org/whl/cu124"


class OfficialJEDiModule(PipelineModule):
    """Run the official JEDi backend on generated and reference collections."""

    name = "jedi_official"
    implementation_status = "implemented"
    implementation_notes = "Private pinned videojedi/vjepa environment and official encoder assets are automatically provisioned. Concrete worker decodes RGB[0,1]16-frame videos, supplies real reference/generated loaders to JEDiMetric.load_features and compute_metric; decoder protocol is adapted."
    implementation_review = "Source inspection; confidence high; pretrained inference not validated."
    implementation_sources = ("../managed_runtime.py", "../_jedi_worker.py")
    provenance = {
        "jedi_score": "adapted",
        "jedi_generated_count": "utility",
        "jedi_reference_count": "utility",
        "jedi_reference_manifest_sha256": "utility",
        "jedi_preprocessing_sha256": "utility",
    }
    deviations = {
        "jedi_score": "Uses the official V-JEPA encoder/probe and JEDi kernel, with Ayase's "
        "uniform full-video 16-frame sampling and square bilinear resize. Upstream accepts "
        "caller-defined loaders; benchmark-specific decoding/sampling is not reproduced.",
    }
    input_type = "video"
    sources = {
        "jedi_score": (
            "Official JEDi implementation at commit " + _UPSTREAM_REVISION
            + " — https://github.com/oooolga/JEDi"
        )
    }
    description = "Official V-JEPA two-distribution JEDi (lower=better)"
    default_config = {
        "reference_dir": None,
        "num_samples": 5000,
        "num_frames": 16,
        "resolution": 224,
        "batch_size": 4,
        "num_workers": 0,
        "feature_path": None,
        "model_dir": None,
        "config_path": None,
        "models_dir": "models",
        "device": "auto",
        "backend_python": None,
        "backend_cuda_visible_devices": None,
        "timeout": 7200,
    }
    required_packages = []  # The official dependencies live in the backend interpreter.
    models = [
        {
            "id": "videojedi==1.1.0",
            "type": "pip_package",
            "install": "Automatically installed into Ayase's private managed JEDi environment",
            "task": "Official JEDi feature extraction and polynomial MMD",
            "notes": "Apache-2.0; source revision " + _UPSTREAM_REVISION,
        },
        {
            "id": "vith16.pth.tar",
            "type": "local",
            "url": "https://dl.fbaipublicfiles.com/jepa/vith16/vith16.pth.tar",
            "task": "Official V-JEPA ViT-H/16 encoder checkpoint",
            "auto_download": True,
        },
        {
            "id": "ssv2-probe.pth.tar",
            "type": "local",
            "url": "https://dl.fbaipublicfiles.com/jepa/vith16/ssv2-probe.pth.tar",
            "task": "Official Something-Something-v2 attentive probe",
            "auto_download": True,
        },
    ]
    metric_info = {
        "jedi_score": "Official two-distribution JEDi (polynomial MMD x100, lower=better)",
        "jedi_generated_count": "Number of generated videos supplied to JEDi",
        "jedi_reference_count": "Number of reference videos supplied to JEDi",
        "jedi_reference_manifest_sha256": "SHA-256 of the ordered reference path manifest",
        "jedi_preprocessing_sha256": "SHA-256 of Ayase's fixed JEDi loader policy",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model_dir = self.config.get("model_dir")
        self._config_path = self.config.get("config_path")
        self._backend_version = None
        self._managed_python: Optional[Path] = None
        self._generated_paths: List[Path] = []
        self._paired_reference_paths: List[Path] = []
        self.last_metadata = {}

    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            if self.config.get("device", "auto") not in ("auto", "cuda", "cuda:0"):
                raise RuntimeError(
                    "videojedi hardcodes cuda:0; use a dedicated worker with "
                    "CUDA_VISIBLE_DEVICES to select another physical GPU"
                )
            if int(self.config["num_frames"]) != 16 or int(self.config["resolution"]) != 224:
                raise ValueError("the pinned JEDi encoder configuration requires 16 frames at 224px")
            self._resolve_backend_python()
            status = self._invoke_worker(check=True)
            self._backend_version = status["videojedi_version"]
            self._prepare_assets()
            self._ml_available = True
            self._backend = "videojedi_isolated"
        except Exception as exc:
            logger.warning("jedi_official automatic backend setup failed: %s", exc)
            self._ml_available = False
            self._backend = "unavailable"

    @staticmethod
    def _host_cuda_available() -> bool:
        """Check CUDA without assuming Ayase's host Torch installation is GPU-enabled."""
        try:
            import torch

            if torch.cuda.is_available():
                return True
        except Exception:
            pass
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return False
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                shell=False,
                timeout=10,
                check=False,
                text=True,
                creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
            )
        except (OSError, subprocess.SubprocessError):
            return False
        return result.returncode == 0 and bool(result.stdout.strip())

    def _resolve_backend_python(self) -> Path:
        """Resolve an expert override or provision Ayase's private official backend."""
        override = self.config.get("backend_python")
        if override:
            return Path(override)
        if self._managed_python is not None:
            return self._managed_python
        if not self._host_cuda_available():
            raise RuntimeError(
                "JEDi requires a host-visible NVIDIA GPU; automatic environment installation "
                "was skipped"
            )
        try:
            from ayase.managed_runtime import ensure_managed_environment

            self._managed_python = ensure_managed_environment(
                "jedi-videojedi-1.1.0",
                _MANAGED_REQUIREMENTS,
                models_dir=self.config["models_dir"],
                pip_options=("--extra-index-url", _PYTORCH_INDEX),
                python_version="3.11",
            )
        except Exception as exc:
            raise RuntimeError(f"automatic JEDi environment installation failed: {exc}") from exc
        return self._managed_python

    def _prepare_assets(self) -> None:
        from ayase.config import download_model_file

        root = Path(self._model_dir or Path(self.config["models_dir"]) / "jedi")
        root.mkdir(parents=True, exist_ok=True)
        for filename in ("vith16.pth.tar", "ssv2-probe.pth.tar"):
            download_model_file(
                filename, "https://dl.fbaipublicfiles.com/jepa/vith16/" + filename, str(root)
            )
        self._model_dir = str(root.resolve())
        self._config_path = self._config_path or str(download_model_file(
            "vith16_ssv2_16x2x3.yaml",
            "https://raw.githubusercontent.com/oooolga/JEDi/" + _UPSTREAM_REVISION
            + "/configs/vith16_ssv2_16x2x3.yaml", str(root),
        ))

    def _invoke_worker(self, payload: Optional[dict] = None, check: bool = False) -> dict:
        executable = self._resolve_backend_python()
        worker = Path(__file__).with_name("_jedi_worker.py")
        environment = os.environ.copy()
        visible_devices = self.config.get("backend_cuda_visible_devices")
        if visible_devices is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(visible_devices)
        with tempfile.TemporaryDirectory(prefix="ayase-jedi-worker-") as directory:
            root = Path(directory)
            result_file = root / "result.json"
            command = [str(executable), str(worker), "--output", str(result_file)]
            if check:
                command.append("--check")
            else:
                request = root / "input.json"
                request.write_text(json.dumps(payload), encoding="utf-8")
                command.extend(["--input", str(request)])
            log_path = root / "worker.log"
            with log_path.open("wb") as log:
                result = subprocess.run(
                    command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                    shell=False, timeout=60 if check else float(self.config["timeout"]),
                    check=False,
                    creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                )
            if not result_file.is_file():
                with log_path.open("rb") as log:
                    log.seek(max(0, log_path.stat().st_size - 2048))
                    detail = log.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"JEDi worker exited {result.returncode}: {detail}")
            output = json.loads(result_file.read_text(encoding="utf-8"))
            if result.returncode != 0 or not output.get("ok"):
                raise RuntimeError(output.get("error", "JEDi worker failed"))
            return output

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample
        path = Path(sample.path)
        if path.is_file():
            self._generated_paths.append(path.resolve())
        reference = getattr(sample, "reference_path", None)
        if reference is not None:
            reference_path = Path(reference)
            if reference_path.is_file() and reference_path.suffix.lower() in _VIDEO_SUFFIXES:
                self._paired_reference_paths.append(reference_path.resolve())
        return sample

    def _reference_paths(self) -> List[Path]:
        reference_dir = self.config.get("reference_dir")
        if reference_dir:
            root = Path(reference_dir)
            if not root.is_dir():
                logger.warning("jedi_official reference_dir does not exist: %s", root)
                return []
            return sorted(
                path.resolve() for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in _VIDEO_SUFFIXES
            )
        # A shared reference may occur on every sample; distributions contain files once.
        return sorted(set(self._paired_reference_paths), key=lambda path: str(path))

    @staticmethod
    def _hash_lines(paths: Sequence[Path]) -> str:
        manifest = "\n".join(str(path.resolve()) for path in paths) + "\n"
        return hashlib.sha256(manifest.encode("utf-8")).hexdigest()

    def _preprocessing_hash(self) -> str:
        policy = {
            "frame_sampling": "uniform_linspace_including_endpoints",
            "input_layout": "TCHW",
            "input_range": "0_1",
            "num_frames": int(self.config["num_frames"]),
            "resize": "bilinear_square_align_corners_false",
            "resolution": int(self.config["resolution"]),
            "upstream_revision": _UPSTREAM_REVISION,
            "backend_identity": self._backend_identity(),
        }
        payload = json.dumps(policy, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _backend_identity(self) -> dict:
        weights = []
        if self._model_dir:
            for name in ("vith16.pth.tar", "ssv2-probe.pth.tar"):
                path = Path(self._model_dir) / name
                if path.is_file():
                    stat = path.stat()
                    weights.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
        config_hash = None
        if self._config_path:
            config_hash = hashlib.sha256(Path(self._config_path).read_bytes()).hexdigest()
        return {
            "package_version": self._backend_version,
            "weights": weights,
            "config_sha256": config_hash,
        }

    def post_process(self, all_samples: List[Sample]) -> None:
        del all_samples
        if not self._ml_available or self._backend != "videojedi_isolated":
            return
        generated = sorted(set(self._generated_paths), key=str)
        references = self._reference_paths()
        if not generated or not references:
            logger.info("jedi_official requires non-empty generated and reference collections")
            return

        limit = min(int(self.config["num_samples"]), len(generated), len(references))
        if limit < 1:
            logger.warning("jedi_official num_samples must be positive")
            return
        generated = generated[:limit]
        references = references[:limit]
        try:
            cache_root = self.config.get("feature_path")
            if cache_root:
                identity = {
                    "reference_files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                                        for p in references],
                    "generated_files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                                        for p in generated],
                    "preprocessing": self._preprocessing_hash(),
                }
                key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
                feature_dir = Path(cache_root) / key
                feature_dir.mkdir(parents=True, exist_ok=True)
                workspace = str(feature_dir)
            else:
                workspace = tempfile.TemporaryDirectory(prefix="ayase-jedi-")
            # The official API looks for train.npy/test.npy before consulting loaders.
            # A run-specific directory prevents stale features from another collection.
            if isinstance(workspace, str):
                feature_dir = workspace
                output = self._score_in_worker(generated, references, feature_dir, limit)
            else:
                with workspace as feature_dir:
                    output = self._score_in_worker(generated, references, feature_dir, limit)
            score = float(output["jedi_score"])
            if not math.isfinite(score):
                raise ValueError("JEDi backend returned a nonfinite score")
            self.last_metadata = {
                "generated_count": len(generated),
                "reference_count": len(references),
                "reference_manifest_sha256": self._hash_lines(references),
                "preprocessing_sha256": self._preprocessing_hash(),
                "upstream_revision": _UPSTREAM_REVISION,
            }
            if self.pipeline is not None:
                self.pipeline.add_dataset_metric("jedi_score", score)
                self.pipeline.add_dataset_metric("jedi_generated_count", len(generated))
                self.pipeline.add_dataset_metric("jedi_reference_count", len(references))
                self.pipeline.add_dataset_metric(
                    "jedi_reference_manifest_sha256",
                    self.last_metadata["reference_manifest_sha256"],
                )
                self.pipeline.add_dataset_metric(
                    "jedi_preprocessing_sha256",
                    self.last_metadata["preprocessing_sha256"],
                )
        except Exception as exc:
            logger.warning("jedi_official computation failed: %s", exc)

    def _score_in_worker(self, generated, references, feature_dir, limit):
        return self._invoke_worker({
            "generated_paths": [str(path) for path in generated],
            "reference_paths": [str(path) for path in references],
            "feature_path": str(feature_dir),
            "model_dir": self._model_dir,
            "config_path": self._config_path,
            "num_samples": limit,
            "num_frames": 16,
            "resolution": 224,
            "batch_size": int(self.config["batch_size"]),
            "num_workers": int(self.config["num_workers"]),
            "device": "cuda:0",
        })

    def on_dispose(self) -> None:
        self._generated_paths.clear()
        self._paired_reference_paths.clear()
        super().on_dispose()

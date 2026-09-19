"""CDPAM perceptual audio distance using the official pretrained package.

CDPAM is a full-reference learned metric for perceptual audio similarity.
The official loader resamples inputs to 22.05 kHz and applies the 16-bit-scale
preprocessing expected by the bundled model. Lower distances indicate greater
perceptual similarity; identical inputs should have a distance near zero.
"""

import hashlib
import logging
import math
import os
from pathlib import Path
import threading

import numpy as np

from ayase.audio import load_audio
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_CHECKPOINT_SHA256 = "453c8b6edee1a94f0120236156436ff28fe4d8d884485e4a67695c8e8570bdfe"
_TORCH_LOAD_LOCK = threading.Lock()


class CDPAMModule(PipelineModule):
    name = "cdpam"
    description = "CDPAM learned perceptual audio distance (full-reference)"
    default_config = {
        "device": "auto",
        "target_sr": 22050,
        "warning_threshold": None,
    }
    models = [
        {
            "id": "cdpam==0.0.6",
            "type": "pip_package",
            "url": "https://pypi.org/project/cdpam/0.0.6/",
            "install": "pip install cdpam==0.0.6",
            "task": "Full-reference learned perceptual audio similarity",
            "size": "98.3 MB",
            "notes": (
                "Official package with bundled scratchJNDdefault_best_model.pth; "
                "checkpoint SHA-256 453c8b6edee1a94f0120236156436ff28fe4d8d884485e4a67695c8e8570bdfe; "
                "source provenance: pranaymanocha/PerceptualAudio commit "
                "4bd0a842b3d7a196b0e15398b761525482b11640"
            ),
        },
    ]
    metric_info = {
        "cdpam_score": (
            "CDPAM perceptual audio distance using official 22.05 kHz "
            "preprocessing (lower=better)"
        ),
    }
    metric_groups = {
        "cdpam_score": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = str(self.config.get("device", "auto"))
        self.target_sr = int(self.config.get("target_sr", 22050))
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = "unavailable"
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import cdpam
            import torch

            if self.device_config in ("auto", ""):
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_config

            self._model = self._load_model(cdpam, device)
            self._device = device
            self._backend = "cdpam:0.0.6"
            logger.info("CDPAM initialised on %s", device)
        except ImportError:
            logger.warning(
                "CDPAM package not installed; install the optional dependency "
                "with `pip install ayase[cdpam]`."
            )
        except Exception as exc:
            logger.warning("CDPAM setup failed: %s", exc)

    @staticmethod
    def _load_model(cdpam_package, device):
        """Load the pinned legacy checkpoint safely on modern PyTorch."""
        package_file = getattr(cdpam_package, "__file__", None)
        if package_file is None:
            return cdpam_package.CDPAM(dev=device)

        checkpoint = (
            Path(package_file).resolve().parent
            / "CDPAM_trained"
            / "scratchJNDdefault_best_model.pth"
        )
        if not checkpoint.is_file():
            raise RuntimeError("CDPAM package checkpoint is missing")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != _CHECKPOINT_SHA256:
            raise RuntimeError("CDPAM package checkpoint failed SHA-256 verification")

        env_name = "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        with _TORCH_LOAD_LOCK:
            previous = os.environ.get(env_name)
            os.environ[env_name] = "1"
            try:
                return cdpam_package.CDPAM(dev=device)
            finally:
                if previous is None:
                    os.environ.pop(env_name, None)
                else:
                    os.environ[env_name] = previous

    def process(self, sample: Sample) -> Sample:
        if self._model is None:
            return sample

        reference = sample.reference_path
        if reference is None:
            return sample

        reference_path = Path(reference)
        sample_path = Path(sample.path)
        if not reference_path.exists() or not sample_path.exists():
            return sample

        try:
            reference_audio = self._prepare_audio(reference_path)
            degraded_audio = self._prepare_audio(sample_path)
            if reference_audio is None or degraded_audio is None:
                return sample

            import torch

            with torch.inference_mode():
                distance = self._model.forward(reference_audio, degraded_audio)

            if hasattr(distance, "detach"):
                distance = distance.detach()
            if hasattr(distance, "cpu"):
                distance = distance.cpu()
            if hasattr(distance, "item"):
                distance = distance.item()
            score = float(distance)
            if not math.isfinite(score):
                logger.warning("CDPAM returned a non-finite score for %s", sample.path)
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.cdpam_score = score
            sample.quality_metrics.metric_backends["cdpam_score"] = self._backend

            if self.warning_threshold is not None and score > float(self.warning_threshold):
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        issue_type="high_cdpam_distance",
                        message=f"High CDPAM perceptual distance: {score:.4f}",
                        details={
                            "cdpam_score": score,
                            "threshold": float(self.warning_threshold),
                        },
                        recommendation=(
                            "Inspect the degraded audio for perceptible changes relative "
                            "to the reference."
                        ),
                    )
                )
        except Exception as exc:
            logger.warning("CDPAM failed for %s: %s", sample.path, exc)

        return sample

    def _prepare_audio(self, path: Path):
        """Decode media and reproduce CDPAM's documented 16-bit-scale input."""
        audio = load_audio(path, target_sr=self.target_sr, mono=True)
        if audio is None:
            return None
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size == 0 or not np.all(np.isfinite(audio)):
            return None
        scaled = np.round(np.clip(audio, -1.0, 1.0) * 32768.0).astype(np.float32)
        return scaled.reshape(1, -1)

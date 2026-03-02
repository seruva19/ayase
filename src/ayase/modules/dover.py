"""DOVER (Disentangled Objective Video Quality Evaluator) module.

ICCV 2023 — state-of-the-art no-reference video quality assessment
that disentangles *technical* quality (noise, blur, compression) from
*aesthetic* quality (composition, colour harmony, content).

Supports two backends:
1. Native DOVER — loads the original model directly (``pip install dover``
   or ``pip install git+https://github.com/VQAssessment/DOVER.git``).
2. pyiqa wrapper — ``pip install pyiqa`` (if pyiqa ships DOVER).

dover_score    — fused overall quality (higher = better, 0-1 sigmoid)
dover_technical — technical sub-score (0-1 sigmoid)
dover_aesthetic — aesthetic sub-score (0-1 sigmoid)
"""

import logging
import os
from typing import Optional

import numpy as np

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _fuse_results(results):
    """Sigmoid rescaling from raw DOVER outputs (matches the original repo)."""
    t = (results[1] + 0.0758) / 0.0129
    a = (results[0] - 0.1253) / 0.0318
    x = t * 0.6104 + a * 0.3896
    return {
        "aesthetic": 1.0 / (1.0 + np.exp(-a)),
        "technical": 1.0 / (1.0 + np.exp(-t)),
        "overall": 1.0 / (1.0 + np.exp(-x)),
    }


class DOVERModule(PipelineModule):
    name = "dover"
    description = "DOVER disentangled technical + aesthetic VQA (ICCV 2023)"
    default_config = {
        "warning_threshold": 0.4,
        "weights_path": None,  # Explicit path to DOVER.pth (optional)
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.warning_threshold = self.config.get("warning_threshold", 0.4)
        self.weights_path = self.config.get("weights_path", None)
        self._device = "cpu"
        self._ml_available = False
        self._backend = None  # "native" or "pyiqa"

        # Native backend state
        self._model = None
        self._sample_types = None
        self._mean = None
        self._std = None

    def setup(self) -> None:
        # Try native DOVER first, then pyiqa fallback
        if self._try_native_setup():
            return
        if self._try_pyiqa_setup():
            return

        logger.warning(
            "DOVER unavailable. Install with: "
            "pip install git+https://github.com/VQAssessment/DOVER.git  "
            "OR  pip install pyiqa"
        )

    # ------------------------------------------------------------------ #
    #  Backend 1: Native DOVER                                            #
    # ------------------------------------------------------------------ #

    def _try_native_setup(self) -> bool:
        try:
            import torch
            from ayase.third_party.dover.models import DOVER as DOVERModel
            from ayase.third_party.dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition  # noqa: F401

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Resolve weights path
            weights = self._resolve_weights()
            if weights is None:
                logger.debug("DOVER weights not found, skipping native backend.")
                return False

            # Hardcoded config matching the original val-l1080p profile
            self._sample_types = {
                "technical": {
                    "fragments_h": 7, "fragments_w": 7,
                    "fsize_h": 32, "fsize_w": 32,
                    "aligned": 32,
                    "clip_len": 32, "frame_interval": 2, "num_clips": 3,
                },
                "aesthetic": {
                    "size_h": 224, "size_w": 224,
                    "clip_len": 32, "frame_interval": 2,
                    "t_frag": 32, "num_clips": 1,
                },
            }

            model_args = {
                "backbone": {
                    "technical": {"type": "swin_tiny_grpb", "checkpoint": True},
                    "aesthetic": {"type": "conv_tiny"},
                },
                "backbone_preserve_keys": "technical,aesthetic",
                "divide_head": True,
                "vqa_head": {"in_channels": 768, "hidden_channels": 64},
            }

            self._model = DOVERModel(**model_args).to(self._device)
            self._model.load_state_dict(
                torch.load(weights, map_location=self._device, weights_only=True)
            )
            self._model.eval()

            self._mean = torch.FloatTensor([123.675, 116.28, 103.53])
            self._std = torch.FloatTensor([58.395, 57.12, 57.375])

            self._ml_available = True
            self._backend = "native"
            logger.info(f"DOVER (native) initialised on {self._device}")
            return True

        except ImportError:
            logger.debug("dover package not installed, trying pyiqa fallback.")
            return False
        except Exception as e:
            logger.debug(f"Native DOVER setup failed: {e}")
            return False

    _DOVER_WEIGHTS_URL = "https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth"

    def _resolve_weights(self) -> Optional[str]:
        """Find DOVER.pth weights file, auto-downloading if needed."""
        if self.weights_path and os.path.exists(self.weights_path):
            return self.weights_path

        models_dir = self.config.get("models_dir", "models")

        candidates = [
            os.path.join(models_dir, "dover", "DOVER.pth"),
            os.path.join(models_dir, "DOVER", "pretrained_weights", "DOVER.pth"),
            os.path.join(models_dir, "DOVER.pth"),
        ]

        # Also check TORCH_HOME
        torch_home = os.environ.get("TORCH_HOME", "")
        if torch_home:
            candidates.append(
                os.path.join(torch_home, "DOVER", "pretrained_weights", "DOVER.pth")
            )

        for path in candidates:
            if os.path.exists(path):
                return path

        # Auto-download
        from ayase.config import download_model_file

        return download_model_file(
            "dover/DOVER.pth", self._DOVER_WEIGHTS_URL, models_dir
        )

    # ------------------------------------------------------------------ #
    #  Backend 2: pyiqa                                                   #
    # ------------------------------------------------------------------ #

    def _try_pyiqa_setup(self) -> bool:
        try:
            import pyiqa
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            models_dir = self.config.get("models_dir", None)
            if models_dir:
                os.environ["TORCH_HOME"] = str(models_dir)

            self._model = pyiqa.create_metric("dover", device=self._device, as_loss=False)
            self._ml_available = True
            self._backend = "pyiqa"
            logger.info(f"DOVER (pyiqa) initialised on {self._device}")
            return True

        except ImportError:
            return False
        except Exception as e:
            logger.debug(f"pyiqa DOVER setup failed: {e}")
            return False

    # ------------------------------------------------------------------ #
    #  Processing                                                         #
    # ------------------------------------------------------------------ #

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            if self._backend == "native":
                aesthetic, technical, overall = self._process_native(sample)
            else:
                aesthetic, technical, overall = self._process_pyiqa(sample)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.dover_score = overall
            sample.quality_metrics.dover_aesthetic = aesthetic
            sample.quality_metrics.dover_technical = technical

            if overall < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low DOVER score: {overall:.3f}",
                        details={
                            "dover_score": overall,
                            "dover_technical": technical,
                            "dover_aesthetic": aesthetic,
                        },
                        recommendation=(
                            "Video quality is below threshold. "
                            "Technical issues: noise/blur/compression. "
                            "Aesthetic issues: composition/colour."
                        ),
                    )
                )

            logger.debug(
                f"DOVER for {sample.path.name}: "
                f"overall={overall:.3f} tech={technical:.3f} aes={aesthetic:.3f}"
            )

        except Exception as e:
            logger.warning(f"DOVER failed for {sample.path}: {e}")

        return sample

    def _process_native(self, sample: Sample):
        """Run inference using the native DOVER model."""
        import torch
        from ayase.third_party.dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition

        # Build temporal samplers
        temporal_samplers = {}
        for stype, sopt in self._sample_types.items():
            if "t_frag" not in sopt:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
                )
            else:
                temporal_samplers[stype] = UnifiedFrameSampler(
                    sopt["clip_len"] // sopt["t_frag"],
                    sopt["t_frag"],
                    sopt["frame_interval"],
                    sopt["num_clips"],
                )

        # Decompose video into views
        views, _ = spatial_temporal_view_decomposition(
            str(sample.path), self._sample_types, temporal_samplers
        )

        # Normalize and reshape
        processed = {}
        for k, v in views.items():
            num_clips = self._sample_types[k].get("num_clips", 1)
            processed[k] = (
                ((v.permute(1, 2, 3, 0) - self._mean) / self._std)
                .permute(3, 0, 1, 2)
                .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
                .transpose(0, 1)
                .to(self._device)
            )

        with torch.no_grad():
            outputs = self._model(processed, reduce_scores=False)
            results = [float(np.mean(out.detach().cpu().numpy())) for out in outputs]

        rescaled = _fuse_results(results)
        return rescaled["aesthetic"], rescaled["technical"], rescaled["overall"]

    def _process_pyiqa(self, sample: Sample):
        """Run inference using pyiqa's DOVER wrapper."""
        import torch

        with torch.no_grad():
            result = self._model(str(sample.path))

        if isinstance(result, torch.Tensor):
            result = result.squeeze().cpu().tolist()

        if isinstance(result, (list, tuple)):
            if len(result) >= 2:
                aesthetic = float(result[0])
                technical = float(result[1])
                overall = (aesthetic + technical) / 2.0
            else:
                aesthetic = technical = overall = float(result[0])
        else:
            aesthetic = technical = overall = float(result)

        return aesthetic, technical, overall

"""Motion smoothness via RIFE video frame interpolation reconstruction error (VBench).

Interpolates middle frames from neighbours (RIFE HD v3 VFI) and measures L1
error; ``motion_smoothness = 1 - mean_error`` (0-1, higher = smoother). This is
the VBench definition, which is specifically tied to RIFE VFI, so when RIFE is
unavailable the metric is left unset rather than approximated with a
differently-scaled optical-flow warping proxy."""

import logging
import os
import cv2
import numpy as np
from typing import Optional, List

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MotionSmoothnessModule(PipelineModule):
    name = "motion_smoothness"
    description = "Motion smoothness via RIFE VFI reconstruction error (VBench)"

    default_config = {
        "vfi_error_threshold": 0.08,
        "max_frames": 64,
    }
    models = [
        {
            "id": "flownet.pkl",
            "type": "local",
            "url": "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/motion_smoothness/flownet.pkl",
            "task": "Bundled RIFE HD v3 interpolation weights",
        },
        {
            "id": "rife_model",
            "type": "pip_package",
            "install": "pip install rife-model",
            "task": "Optional external RIFE interpolation fallback",
        },
    ]
    metric_info = {
        "motion_smoothness": "VFI or optical-flow reconstruction smoothness (0-1, higher=better)",
    }
    metric_groups = {
        "motion_smoothness": "motion",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.vfi_error_threshold = self.config.get("vfi_error_threshold", 0.08)
        self.max_frames = self.config.get("max_frames", 64)

        self._rife_model = None
        self._device = "cpu"
        self._rife_available = False
        self._backend = None

    def setup(self):
        try:
            from ayase.runtime import resolve_torch_device
            self._device = resolve_torch_device(self.config.get("device", "auto"))

            # Try bundled RIFE HD v3 (ayase.third_party.rife)
            try:
                from ayase.third_party.rife.RIFE_HDv3 import Model as RIFEModel
                from ayase.config import download_model_file

                models_dir = self.config.get("models_dir", "models")
                rife_dir = os.path.join(models_dir, "rife")
                weights_path = os.path.join(rife_dir, "flownet.pkl")

                if not os.path.exists(weights_path):
                    download_model_file(
                        "rife/flownet.pkl",
                        "https://huggingface.co/AkaneTendo25/ayase-models/resolve/main/motion_smoothness/flownet.pkl",
                        models_dir,
                    )

                self._rife_model = RIFEModel()
                self._rife_model.load_model(rife_dir, rank=-1)
                self._rife_model.eval()
                self._rife_available = True
                self._backend = "rife"
                logger.info("RIFE HD v3 loaded on %s", self._device)
                return
            except ImportError:
                pass
            except Exception as exc:
                logger.debug("Bundled RIFE load failed: %s", exc)

            # Fallback: external rife_model package
            try:
                from rife_model import load_rife_model
                self._rife_model = load_rife_model(device=self._device)
                self._rife_available = True
                self._backend = "rife"
                logger.info("RIFE model loaded on %s", self._device)
                return
            except ImportError:
                pass
            except Exception as exc:
                logger.debug("rife_model import failed: %s", exc)

            # Neither import path worked -- disable (VBench motion_smoothness is
            # defined via RIFE VFI; do not approximate with a flow proxy).
            self._backend = "unavailable"
            logger.warning(
                "RIFE VFI model not available (neither bundled 'ayase.third_party.rife' "
                "nor 'rife_model' found); motion_smoothness left unset."
            )

        except ImportError:
            self._backend = "unavailable"
            logger.warning(
                "PyTorch is not installed; RIFE model cannot be loaded. "
                "motion_smoothness left unset."
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("Motion smoothness setup failed: %s; motion_smoothness left unset.", e)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video or not self._rife_available:
            return sample

        try:
            self._analyze_rife(sample)
        except Exception as e:
            logger.warning(f"Motion smoothness analysis failed: {e}")

        return sample

    def _analyze_rife(self, sample: Sample) -> None:
        """RIFE-based motion smoothness (VBench approach).

        For triplets (I0, I1, I2): interpolate I1_pred from (I0, I2),
        then measure |I1_pred - I1_gt|.
        """
        import torch

        frames = self._load_frames(sample)
        if len(frames) < 3:
            return

        errors = []

        h, w = frames[0].shape[:2]
        # Pad to multiple of 32 (RIFE requirement)
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32
        need_pad = (ph != h or pw != w)

        with torch.no_grad():
            for i in range(1, len(frames) - 1):
                I0 = torch.from_numpy(frames[i - 1]).permute(2, 0, 1).float().unsqueeze(0).to(self._device) / 255.0
                I1_gt = torch.from_numpy(frames[i]).permute(2, 0, 1).float().unsqueeze(0).to(self._device) / 255.0
                I2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).float().unsqueeze(0).to(self._device) / 255.0

                if need_pad:
                    I0 = torch.nn.functional.pad(I0, (0, pw - w, 0, ph - h))
                    I1_gt = torch.nn.functional.pad(I1_gt, (0, pw - w, 0, ph - h))
                    I2 = torch.nn.functional.pad(I2, (0, pw - w, 0, ph - h))

                # RIFE interpolation at t=0.5
                I1_pred = self._rife_model.inference(I0, I2)

                # Crop back to original size and compute L1 error
                diff = torch.mean(torch.abs(I1_pred[:, :, :h, :w] - I1_gt[:, :, :h, :w])).item()
                errors.append(diff)

        avg_error = float(np.mean(errors))
        smoothness = max(0.0, 1.0 - avg_error)

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.motion_smoothness = smoothness

        if avg_error > self.vfi_error_threshold:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low motion smoothness (RIFE error): {avg_error:.3f}",
                    details={"vfi_error": avg_error},
                )
            )

    def _load_frames(self, sample: Sample) -> List[np.ndarray]:
        frames = []
        cap = cv2.VideoCapture(str(sample.path))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if total <= 0:
                return []

            n = min(self.max_frames, total)
            indices = np.linspace(0, total - 1, n, dtype=int)

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)
        finally:
            cap.release()
        return frames



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

    def __init__(self, config=None):
        super().__init__(config)
        self.vfi_error_threshold = self.config.get("vfi_error_threshold", 0.08)
        self.max_frames = self.config.get("max_frames", 64)

        self._rife_model = None
        self._device = "cpu"
        self._rife_available = False

    def setup(self):
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

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
                logger.info("RIFE model loaded on %s", self._device)
                return
            except ImportError:
                pass
            except Exception as exc:
                logger.debug("rife_model import failed: %s", exc)

            # Neither import path worked -- fall back gracefully
            logger.warning(
                "RIFE VFI model not available (neither 'rife_model' nor "
                "'model.RIFE' packages found). Falling back to Farneback "
                "optical-flow warping proxy for motion smoothness."
            )

        except ImportError:
            logger.warning(
                "PyTorch is not installed; RIFE model cannot be loaded. "
                "Falling back to Farneback optical-flow warping proxy."
            )
        except Exception as e:
            logger.warning("Motion smoothness setup failed: %s. "
                           "Falling back to flow-based proxy.", e)

    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        try:
            if self._rife_available:
                self._analyze_rife(sample)
            else:
                self._analyze_flow_proxy(sample)
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

    def _analyze_flow_proxy(self, sample: Sample) -> None:
        """Farneback flow-based warping proxy when RIFE is unavailable.

        For triplets (I0, I1, I2): warp I0 forward and I2 backward by 0.5*flow,
        blend, and compare to I1.
        """
        frames = self._load_frames(sample)
        if len(frames) < 3:
            return

        errors = []

        for i in range(1, len(frames) - 1):
            I0 = frames[i - 1]
            I1 = frames[i]
            I2 = frames[i + 1]

            prev_gray = cv2.cvtColor(I0, cv2.COLOR_RGB2GRAY)
            next_gray = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            h, w = prev_gray.shape
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            map0_x = (x + flow[..., 0] * 0.5).astype(np.float32)
            map0_y = (y + flow[..., 1] * 0.5).astype(np.float32)
            I0_warped = cv2.remap(I0, map0_x, map0_y, cv2.INTER_LINEAR)

            map2_x = (x - flow[..., 0] * 0.5).astype(np.float32)
            map2_y = (y - flow[..., 1] * 0.5).astype(np.float32)
            I2_warped = cv2.remap(I2, map2_x, map2_y, cv2.INTER_LINEAR)

            I1_pred = cv2.addWeighted(I0_warped, 0.5, I2_warped, 0.5, 0)

            diff = np.mean(np.abs(I1.astype(np.float32) - I1_pred.astype(np.float32))) / 255.0
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
                    message=f"Low motion smoothness (flow proxy error): {avg_error:.3f}",
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



"""Google UVQ 1.5 no-reference perceptual video quality.

UVQ 1.5 predicts a human-aligned mean-opinion score for user-generated video
without requiring a pristine reference. The upstream model samples at 1 FPS
and returns a score in [1, 5], where higher is better.
"""

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2

from ayase.config import download_model_file
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_UVQ_COMMIT = "811b6b1b7c085a9ac59ee5e3a03c560be18fe91c"
_UVQ_RAW_ROOT = f"https://raw.githubusercontent.com/google/uvq/{_UVQ_COMMIT}"
_WEIGHTS = {
    "content": (
        "uvq1p5/content_net.pth",
        f"{_UVQ_RAW_ROOT}/uvq1p5_pytorch/checkpoints/content_net.pth",
    ),
    "distortion": (
        "uvq1p5/distortion_net.pth",
        f"{_UVQ_RAW_ROOT}/uvq1p5_pytorch/checkpoints/distortion_net.pth",
    ),
    "aggregation": (
        "uvq1p5/aggregation_net.pth",
        f"{_UVQ_RAW_ROOT}/uvq1p5_pytorch/checkpoints/aggregation_net.pth",
    ),
}


class UVQModule(PipelineModule):
    name = "uvq"
    description = "Google UVQ 1.5 no-reference perceptual video MOS"
    default_config = {
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {
            "id": "uvq1p5/content_net.pth",
            "type": "local",
            "url": _WEIGHTS["content"][1],
            "task": "Google UVQ 1.5 content network",
            "size": "15.3 MB",
            "auto_download": True,
            "notes": f"Apache-2.0; pinned to google/uvq commit {_UVQ_COMMIT}",
        },
        {
            "id": "uvq1p5/distortion_net.pth",
            "type": "local",
            "url": _WEIGHTS["distortion"][1],
            "task": "Google UVQ 1.5 distortion network",
            "size": "15.3 MB",
            "auto_download": True,
            "notes": f"Apache-2.0; pinned to google/uvq commit {_UVQ_COMMIT}",
        },
        {
            "id": "uvq1p5/aggregation_net.pth",
            "type": "local",
            "url": _WEIGHTS["aggregation"][1],
            "task": "Google UVQ 1.5 aggregation network",
            "size": "0.3 MB",
            "auto_download": True,
            "notes": f"Apache-2.0; pinned to google/uvq commit {_UVQ_COMMIT}",
        },
    ]
    metric_info = {
        "uvq1p5_score": (
            "Google UVQ 1.5 no-reference perceptual video MOS "
            "(1-5, higher=better)"
        ),
    }
    metric_groups = {"uvq1p5_score": "nr_quality"}

    def __init__(self, config=None):
        super().__init__(config)
        self.device_config = str(self.config.get("device", "auto"))
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._backend = "unavailable"
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            logger.debug("UVQ 1.5: test mode, skipping model setup")
            return

        try:
            import torch

            from ayase.third_party.uvq.uvq1p5_pytorch.utils.uvq1p5 import UVQ1p5

            if self.device_config in ("", "auto"):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            elif self.device_config in ("cpu", "cuda"):
                device = self.device_config
            else:
                logger.warning("UVQ 1.5 device must be auto, cpu, or cuda")
                return
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("UVQ 1.5 requested CUDA, but CUDA is unavailable")
                return

            paths = {
                component: download_model_file(relative_path, url, self.models_dir)
                for component, (relative_path, url) in _WEIGHTS.items()
            }

            model = UVQ1p5(eval_mode=True, pretrained=False)
            model.content_net.load_state_dict(paths["content"])
            model.distortion_net.load_state_dict(paths["distortion"])
            model.aggregation_net.load_state_dict(paths["aggregation"])
            model.to(device)
            model.eval()

            self._model = model
            self._device = device
            self._backend = "uvq1p5"
            logger.info("Google UVQ 1.5 initialised on %s", device)
        except ImportError as exc:
            logger.warning("UVQ 1.5 requires torch and torchvision: %s", exc)
        except Exception as exc:
            logger.warning("UVQ 1.5 backend initialisation failed: %s", exc)

    @staticmethod
    def _video_geometry(path: Path) -> Optional[Tuple[int, bool, float]]:
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return None
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            capture.release()

        if frame_count <= 0 or fps <= 0 or width <= 0 or height <= 0:
            return None
        return max(1, math.ceil(frame_count / fps)), height > width, fps

    def process(self, sample: Sample) -> Sample:
        if self._backend != "uvq1p5" or self._model is None:
            return sample
        if not sample.is_video:
            return sample

        try:
            geometry = self._video_geometry(sample.path)
            if geometry is None:
                return sample
            video_length, transpose, original_fps = geometry
            result = self._model.infer(
                str(sample.path),
                video_length,
                transpose,
                fps=1,
                orig_fps=original_fps,
                device=self._device,
            )
            score = float(result["uvq1p5_score"])
            if not math.isfinite(score) or not 1.0 <= score <= 5.0:
                logger.warning("UVQ 1.5 produced invalid score: %r", score)
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.uvq1p5_score = score
        except Exception as exc:
            logger.warning("UVQ 1.5 failed for %s: %s", sample.path, exc)
        return sample

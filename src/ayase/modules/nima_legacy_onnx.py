"""NIMA legacy ONNX module.

Frozen-ONNX inference wrapper for the original NIMA (Neural Image Assessment,
Talebi & Milanfar 2017) MobileNet-based aesthetic predictor. The original
implementation was trained with TensorFlow; this module loads a pre-exported
.onnx file via onnxruntime so the metric can run without pulling in a heavy
TensorFlow dependency, preserving numerical reproducibility of legacy
leaderboard runs.

The ONNX export step itself is performed offline and is not part of this
module — only the inference path lives here.

The model head outputs a 10-bin probability distribution over the MOS scale
1..10; the final scalar score is the distribution mean:

    score = sum(p[i] * (i + 1) for i in range(10))   # in [1, 10]

Stores the score in ``QualityMetrics.aesthetic_score_legacy``.
"""

import logging
from pathlib import Path
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class NIMALegacyONNXModule(PipelineModule):
    name = "nima_legacy_onnx"
    description = (
        "Legacy NIMA aesthetic score (1-10) via a frozen ONNX MobileNet export"
    )
    default_config = {
        "model_path": "nima/nima_mobilenet_aesthetic.onnx",
        "models_dir": "models",
        "device": "auto",
        "image_size": 224,
        "preprocess": "mobilenet",
    }
    models = [
        {
            "id": "cromsc/nima-mobilenet-aesthetic",
            "type": "huggingface",
            "task": "Frozen ONNX export of the NIMA MobileNet aesthetic predictor",
        },
    ]
    metric_info = {
        "aesthetic_score_legacy": (
            "NIMA legacy aesthetic score (1-10), frozen ONNX MobileNet backend"
        ),
    }
    metric_groups = {
        "aesthetic_score_legacy": "aesthetic",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._session = None
        self._input_name: Optional[str] = None
        self._input_layout = "nhwc"
        self._image_size = int(self.config.get("image_size", 224))
        self._device = "cpu"
        self._backend = None

    _WEIGHTS_URLS = {
        "nima_mobilenet_aesthetic.onnx": (
            "https://huggingface.co/cromsc/nima-mobilenet-aesthetic/resolve/main/"
            "nima_mobilenet_aesthetic.onnx"
        ),
    }

    def setup(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            self._backend = "unavailable"
            logger.warning(
                "NIMA legacy ONNX unavailable: onnxruntime is not installed"
            )
            return

        model_path = self._resolve_model_path()
        if model_path is None or not model_path.exists():
            self._backend = "unavailable"
            logger.warning(
                "NIMA legacy ONNX unavailable: model file not found at %s",
                model_path,
            )
            return

        try:
            providers = self._select_providers(ort)
            self._session = ort.InferenceSession(
                str(model_path), providers=providers
            )
            inputs = self._session.get_inputs()
            if not inputs:
                self._backend = "unavailable"
                logger.warning(
                    "NIMA legacy ONNX unavailable: model %s has no inputs",
                    model_path,
                )
                self._session = None
                return
            self._input_name = inputs[0].name
            self._input_layout = self._infer_input_layout(inputs[0].shape)
            active = self._session.get_providers()
            self._device = (
                "cuda" if any("CUDA" in p for p in active) else "cpu"
            )
            self._ml_available = True
            self._backend = "onnxruntime"
            logger.info(
                "NIMA legacy ONNX loaded from %s (providers=%s)",
                model_path,
                active,
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("NIMA legacy ONNX setup failed: %s", e)
            self._session = None

    def _resolve_model_path(self) -> Optional[Path]:
        raw = self.config.get("model_path")
        if not raw:
            return None
        path = Path(raw)
        if path.is_absolute() or path.exists():
            return path
        models_dir = self.config.get("models_dir", "models")
        if models_dir:
            candidate = Path(models_dir) / path.name
            if candidate.exists():
                return candidate
            # Also try the raw relative path under models_dir
            joined = Path(models_dir) / path
            if joined.exists():
                return joined
            return self._ensure_model_file(path, models_dir)
        return self._ensure_model_file(path, ".")

    def _ensure_model_file(self, path: Path, models_dir) -> Path:
        url = self._WEIGHTS_URLS.get(path.name)
        if not url:
            return path
        try:
            from ayase.config import download_model_file

            return download_model_file(str(path), url, str(models_dir))
        except Exception as e:
            logger.warning("NIMA legacy ONNX download failed: %s", e)
            return path

    @staticmethod
    def _infer_input_layout(shape) -> str:
        dims = list(shape or [])
        if len(dims) != 4:
            return "nhwc"
        if dims[1] == 3:
            return "nchw"
        if dims[-1] == 3:
            return "nhwc"
        return "nhwc"

    def _select_providers(self, ort) -> list:
        device_cfg = str(self.config.get("device", "auto")).lower()
        available = set(ort.get_available_providers())

        cuda_available = False
        if device_cfg in ("auto", "cuda", "gpu"):
            try:
                import torch

                cuda_available = bool(torch.cuda.is_available())
            except Exception:
                cuda_available = "CUDAExecutionProvider" in available

        if device_cfg == "cpu":
            return ["CPUExecutionProvider"]
        if device_cfg in ("cuda", "gpu"):
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]
        # auto
        if cuda_available and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available:
            return sample
        # Image-only first cut; video samples are out of scope for this
        # legacy wrapper. Keyframe support can be added later via the same
        # pattern used by laion_aesthetic._load_frames.
        if sample.is_video:
            return sample

        try:
            import numpy as np
            from PIL import Image

            with Image.open(sample.path) as img:
                rgb = img.convert("RGB").resize(
                    (self._image_size, self._image_size), Image.BILINEAR
                )
                arr = np.asarray(rgb, dtype=np.float32)

            preprocess = str(self.config.get("preprocess", "mobilenet")).lower()
            if preprocess == "mobilenet":
                arr = (arr / 127.5) - 1.0
            elif preprocess in {"none", "raw"}:
                pass
            else:
                logger.warning("Unknown NIMA preprocess %r; using mobilenet", preprocess)
                arr = (arr / 127.5) - 1.0

            if self._input_layout == "nchw":
                tensor = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
            else:
                tensor = arr[None, ...].astype(np.float32)

            outputs = self._session.run(None, {self._input_name: tensor})
            if not outputs:
                logger.warning(
                    "NIMA legacy ONNX produced no outputs for %s", sample.path
                )
                return sample
            probs = np.asarray(outputs[0]).reshape(-1).astype(np.float64)
            if probs.size != 10:
                logger.warning(
                    "NIMA legacy ONNX: expected 10-bin output, got shape %s "
                    "for %s",
                    np.asarray(outputs[0]).shape,
                    sample.path,
                )
                return sample
            total = float(probs.sum())
            if total > 0 and not np.isclose(total, 1.0, atol=1e-3):
                probs = probs / total
            score = float(
                sum(float(probs[i]) * (i + 1) for i in range(10))
            )
            sample.quality_metrics.aesthetic_score_legacy = float(score)
        except Exception as e:
            logger.warning(
                "NIMA legacy ONNX processing failed for %s: %s", sample.path, e
            )
        return sample

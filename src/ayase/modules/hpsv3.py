"""HPSv3 module.

This module loads HPSv3 on top of Qwen2-VL for prompt-conditioned reward
scoring. Images are scored directly; videos are scored by uniformly sampling
frames and averaging frame scores.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class HPSv3Module(PipelineModule):
    name = "hpsv3"
    description = "HPSv3 wide-spectrum human preference scoring (frame-averaged on video)"
    default_config = {
        "num_frames": 5,
        "device": "auto",
        "warning_threshold": None,
    }
    models = [
        {
            "id": "MizzenAI/HPSv3",
            "type": "huggingface",
            "task": "HPSv3 prompt-conditioned reward model",
        },
        {
            "id": "Qwen/Qwen2-VL-7B-Instruct",
            "type": "huggingface",
            "task": "Vision-language backbone used by HPSv3",
        },
    ]
    metric_info = {
        "hpsv3_score": "HPSv3 human preference reward mean over sampled frames (higher=better)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.num_frames = self.config.get("num_frames", 5)
        self.device_config = self.config.get("device", "auto")
        self.warning_threshold = self.config.get("warning_threshold")
        self._backend = None
        self._model = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import huggingface_hub  # noqa: F401
            import safetensors  # noqa: F401
            import torch
            import transformers  # noqa: F401

            from ayase.third_party.hpsv3 import HPSv3RewardInferencer

            self._device = self._resolve_device(torch)
            models_dir = self.config.get("models_dir")
            if models_dir:
                os.environ.setdefault("HF_HOME", str(models_dir))
                os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(models_dir))

            self._model = HPSv3RewardInferencer(device=self._device, cache_dir=models_dir)
            self._backend = "hpsv3"
            logger.info("HPSv3 model initialized on %s", self._device)
        except ImportError:
            logger.warning("HPSv3 unavailable: missing dependency.")
        except Exception as e:
            logger.warning("Failed to load HPSv3 model: %s", e)

    def process(self, sample: Sample) -> Sample:
        if self._backend != "hpsv3":
            return sample

        caption_text = self._get_caption_text(sample)
        if not caption_text:
            return sample

        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample

            score = self._score_hpsv3(frames, caption_text)
            if score is None:
                return sample

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.hpsv3_score = score

            if self.warning_threshold is not None and score < self.warning_threshold:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low HPSv3 score: {score:.3f}",
                        details={"hpsv3_score": score},
                    )
                )
        except Exception as e:
            logger.warning("HPSv3 inference failed: %s", e)

        return sample

    def _resolve_device(self, torch_module) -> str:
        if self.device_config == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        return str(self.device_config)

    def _get_caption_text(self, sample: Sample) -> Optional[str]:
        if sample.caption:
            return sample.caption.text
        txt_path = sample.path.with_suffix(".txt")
        if not txt_path.exists():
            return None
        try:
            return txt_path.read_text(encoding="utf-8").strip()
        except Exception:
            logger.debug("Failed to read caption file: %s", txt_path)
            return None

    def _score_hpsv3(self, frames: List[Image.Image], caption: str) -> Optional[float]:
        temp_paths: List[Path] = []
        try:
            for frame in frames:
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                path_obj = Path(tmp_path)
                frame.save(path_obj)
                temp_paths.append(path_obj)

            rewards = self._model.reward(
                [caption] * len(temp_paths),
                image_paths=[str(path) for path in temp_paths],
            )
            scores = []
            for reward in rewards:
                value = reward[0] if isinstance(reward, (list, tuple)) else reward
                if hasattr(value, "item"):
                    value = value.item()
                elif isinstance(value, list):
                    value = value[0]
                scores.append(float(value))
            return float(np.mean(scores)) if scores else None
        except Exception as e:
            logger.debug("HPSv3 native scoring failed: %s", e)
            return None
        finally:
            for path in temp_paths:
                path.unlink(missing_ok=True)

    def _load_frames(self, sample: Sample) -> List[Image.Image]:
        try:
            if not sample.is_video:
                bgr = cv2.imread(str(sample.path))
                if bgr is None:
                    return []
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return [Image.fromarray(rgb)]

            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []

            frame_count = min(self.num_frames, total)
            indices = np.linspace(0, total - 1, frame_count, dtype=int)
            frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()
                if ok:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(rgb))
            cap.release()
            return frames
        except Exception:
            return []

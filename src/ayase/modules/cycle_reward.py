"""CycleReward image-text alignment reward (ICCV 2025).

Uses the upstream CycleReward-Combo model trained from cycle-consistency
preferences. Images are scored directly; videos are represented by uniformly
sampled frames and the frame rewards are averaged. Higher is better.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class CycleRewardModule(PipelineModule):
    name = "cycle_reward"
    description = "CycleReward-Combo image-text alignment reward (ICCV 2025)"
    default_config = {
        "model_type": "CycleReward-Combo",
        "num_frames": 5,
        "device": "auto",
        "models_dir": "models",
    }
    models = [
        {
            "id": "cyclereward",
            "type": "pip_package",
            "install": "pip install cyclereward==0.1.7",
            "task": "CycleReward implementation",
        },
        {
            "id": "CycleReward-Combo.pth",
            "type": "local",
            "url": (
                "https://github.com/hjbahng/cyclereward/releases/download/"
                "v1.0.0/CycleReward-Combo.pth"
            ),
            "task": "CycleReward combined I2T/T2I preference checkpoint",
            "auto_download": True,
        },
    ]
    metric_info = {
        "cycle_reward_score": (
            "CycleReward-Combo image-text alignment reward, averaged over "
            "sampled video frames (higher=better)"
        ),
    }
    metric_groups = {"cycle_reward_score": "alignment"}

    def __init__(self, config=None):
        super().__init__(config)
        self.model_type = str(self.config.get("model_type", "CycleReward-Combo"))
        self.num_frames = int(self.config.get("num_frames", 5))
        self.device_config = str(self.config.get("device", "auto"))
        self.models_dir = Path(self.config.get("models_dir", "models"))
        self._model = None
        self._preprocess = None
        self._torch = None
        self._device = "cpu"
        self._backend = "unavailable"

    def setup(self) -> None:
        if self.model_type != "CycleReward-Combo":
            logger.warning(
                "CycleReward exposes only the validated Combo model; got %r",
                self.model_type,
            )
            return
        try:
            import torch

            if self.device_config in ("auto", ""):
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device_config

            cache_dir = self.models_dir / "cyclereward"
            cache_dir.mkdir(parents=True, exist_ok=True)
            # Set these before importing cyclereward/transformers: both libraries
            # resolve their default cache paths at import time.
            os.environ["HF_HOME"] = str(cache_dir / "huggingface")
            os.environ["HF_HUB_CACHE"] = str(cache_dir / "huggingface" / "hub")
            os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ["HF_HUB_CACHE"]
            os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "huggingface" / "transformers")
            torch.hub.set_dir(str(cache_dir / "torch_hub"))
            from cyclereward import cyclereward

            model, preprocess = cyclereward(
                device=device,
                model_type=self.model_type,
                cache_dir=str(cache_dir),
            )
            self._torch = torch
            self._device = device
            self._model = model
            self._preprocess = preprocess
            self._backend = "cyclereward"
            logger.info("CycleReward-Combo initialized on %s", device)
        except ImportError as exc:
            logger.warning("CycleReward requires `pip install cyclereward==0.1.7`: %s", exc)
        except Exception as exc:
            logger.warning("CycleReward initialization failed: %s", exc)

    def process(self, sample: Sample) -> Sample:
        if self._backend != "cyclereward":
            return sample
        caption = self._caption_text(sample)
        if not caption:
            return sample
        try:
            frames = self._load_frames(sample)
            if not frames:
                return sample
            scores = []
            for frame in frames:
                tensor = self._preprocess(frame).unsqueeze(0).to(self._device)
                reward = self._model.score(tensor, caption)
                scores.append(float(reward.detach().cpu().reshape(-1)[0].item()))
            if not scores:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.cycle_reward_score = float(np.mean(scores))
        except Exception as exc:
            logger.warning("CycleReward inference failed for %s: %s", sample.path, exc)
        return sample

    def _caption_text(self, sample: Sample) -> Optional[str]:
        if sample.caption and sample.caption.text:
            return sample.caption.text
        sidecar = sample.path.with_suffix(".txt")
        try:
            return sidecar.read_text(encoding="utf-8").strip() if sidecar.exists() else None
        except Exception:
            return None

    def _load_frames(self, sample: Sample) -> List[Image.Image]:
        try:
            from ayase.image import arrays_to_pil, sample_frames

            arrays = sample_frames(sample.path, max_frames=self.num_frames, color="rgb")
            return arrays_to_pil(arrays)
        except Exception:
            return []

    def on_dispose(self) -> None:
        self._model = None
        self._preprocess = None
        if self._torch is not None and self._device.startswith("cuda"):
            try:
                self._torch.cuda.empty_cache()
            except Exception:
                pass
        super().on_dispose()

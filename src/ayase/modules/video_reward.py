"""VideoReward preference scoring for prompt-conditioned generated videos.

Runs the official VideoAlign ``VideoVLMRewardInference``: a Qwen2-VL-2B reward
model with a reward head read at the <|VQ_reward|>/<|MQ_reward|>/<|TA_reward|>
special tokens, the checkpoint's ``detailed_special`` prompt template, uniform
video sampling at the configured fps/max_pixels from ``model_config.json``, and
per-dimension normalization by the checkpoint's ``inference_config`` means/stds.
``video_reward_score`` is the normalized Overall (VQ+MQ+TA); higher indicates
stronger learned preference. Images are skipped. Scores are produced only by
the official model; no proxy is used when unavailable.

Source: https://github.com/KlingAIResearch/VideoAlign (VideoReward)
"""

import logging
import os
from typing import Optional

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/KlingTeam/VideoReward/resolve/main"


class VideoRewardModule(PipelineModule):
    name = "video_reward"
    provenance = "published"
    sources = {
        "video_reward_score": "VideoAlign/VideoReward, Liu et al. NeurIPS 2025 — https://github.com/KlingAIResearch/VideoAlign",
    }
    description = "VideoAlign human preference reward model (NeurIPS 2025)"
    default_config = {
        "model_name": "KlingTeam/VideoReward",
        "checkpoint_step": -1,
        "use_norm": True,
    }
    metric_groups = {
        "video_reward_score": "alignment",
    }
    models = [
        {
            "id": "KlingTeam/VideoReward",
            "type": "huggingface",
            "task": "Qwen2-VL-2B video reward model (VQ/MQ/TA)",
        },
    ]

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._inferencer = None
        self._device = "cpu"

    def setup(self) -> None:
        try:
            import torch

            from ayase.runtime import resolve_torch_device
            from ayase.third_party.videoalign import VideoVLMRewardInference

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            ckpt_dir = self._resolve_checkpoint()
            if ckpt_dir is None:
                logger.warning("VideoReward checkpoint not found; metric skipped.")
                return

            self._inferencer = VideoVLMRewardInference(
                ckpt_dir,
                load_from_pretrained_step=int(self.config.get("checkpoint_step", -1)),
                device=self._device,
                dtype=torch.bfloat16,
            )
            self._ml_available = True
            self._backend = "videoreward_official"
            logger.info("VideoAlign reward model loaded on %s", self._device)
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("VideoAlign unavailable: %s", e)

    def _resolve_checkpoint(self) -> Optional[str]:
        """Return a local dir holding model_config.json + checkpoint-*/model.pth."""
        explicit = self.config.get("checkpoint_dir")
        if explicit and os.path.exists(os.path.join(explicit, "model_config.json")):
            return explicit

        models_dir = self.config.get("models_dir", "models")
        local = os.path.join(models_dir, "videoreward")
        if os.path.exists(os.path.join(local, "model_config.json")):
            return local

        # Download the official checkpoint layout (model_config.json at root,
        # weights under checkpoint-*/model.pth).
        from ayase.config import download_model_file

        files = [
            ("model_config.json",),
            ("checkpoint-11352", "model.pth"),
            ("checkpoint-11352", "tokenizer", "added_tokens.json"),
            ("checkpoint-11352", "tokenizer", "merges.txt"),
            ("checkpoint-11352", "tokenizer", "special_tokens_map.json"),
            ("checkpoint-11352", "tokenizer", "tokenizer.json"),
            ("checkpoint-11352", "tokenizer", "tokenizer_config.json"),
            ("checkpoint-11352", "tokenizer", "vocab.json"),
        ]
        try:
            for rel in files:
                rel_path = os.path.join("videoreward", *rel)
                url = f"{_HF_BASE}/{'/'.join(rel)}"
                download_model_file(rel_path, url, models_dir)
        except Exception as e:
            logger.warning("VideoReward checkpoint download failed: %s", e)
            return None
        return local

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._inferencer is None:
            return sample
        # VideoReward is a video model; images have no defined reward.
        if not sample.is_video:
            return sample

        try:
            caption = sample.caption.text if sample.caption else ""
            rewards = self._inferencer.reward(
                [str(sample.path)],
                [caption if caption else "the video"],
                use_norm=bool(self.config.get("use_norm", True)),
            )
            r = rewards[0]
            if r.get("Overall") is not None:
                sample.quality_metrics.video_reward_score = float(r["Overall"])
        except Exception as e:
            logger.warning("VideoAlign processing failed: %s", e)
        return sample

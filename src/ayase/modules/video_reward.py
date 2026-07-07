"""VideoAlign reward model module.

NeurIPS 2025. Human preference alignment scoring for video generation.
Based on QWen2-VL reward model.
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VideoRewardModule(PipelineModule):
    name = "video_reward"
    description = "VideoAlign human preference reward model (NeurIPS 2025)"
    default_config = {
        "model_name": "KlingTeam/VideoReward",
        "subsample": 8,
        "trust_remote_code": True,
        "model_revision": None,
    }
    metric_groups = {
        "video_reward_score": "alignment",
    }

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        # VideoReward (KwaiVGI/VideoReward) is a Qwen2-VL-based multi-dimensional
        # preference reward model (VQ/MQ/TA). Its custom reward-head class is not a
        # standard transformers Auto class, so we load it generically via AutoModel
        # + trust_remote_code (resolved through the checkpoint's auto_map, when the
        # checkpoint ships HF-compatible config/remote code). Real-or-none: any
        # failure (including the public raw-.pth checkpoint that has no HF config)
        # leaves the metric None with backend "unavailable".
        try:
            from transformers import AutoModel, AutoProcessor
            from ayase.runtime import resolve_torch_device

            model_name = self.config.get("model_name", "KwaiVGI/VideoReward")
            device = resolve_torch_device(self.config.get("device", "auto"))

            trc = self.config.get("trust_remote_code", True)
            rev = self.config.get("model_revision", None)
            self._processor = AutoProcessor.from_pretrained(
                model_name, trust_remote_code=trc, revision=rev
            )
            self._model = AutoModel.from_pretrained(
                model_name, trust_remote_code=trc, revision=rev
            ).to(device)
            self._model.eval()
            self._device = device
            self._ml_available = True
            self._backend = "videoreward_hf"
            logger.info("VideoAlign reward model loaded on %s", device)
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("VideoAlign unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._model is None:
            return sample
        # VideoReward is a video model (Qwen2-VL video path); images have no
        # defined reward, so leave the metric None for non-video samples.
        if not sample.is_video:
            return sample

        try:
            import torch
            from qwen_vl_utils import process_vision_info

            caption = sample.caption.text if sample.caption else ""
            prompt = caption if caption else "the video"
            nframes = int(self.config.get("subsample", 8))
            max_pixels = int(self.config.get("max_pixels", 448 * 448))

            messages = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": f"file://{sample.path}",
                                "max_pixels": max_pixels,
                                "nframes": nframes,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            batch = self._processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                videos_kwargs={"do_rescale": True},
            ).to(self._device)

            with torch.no_grad():
                rewards = self._model(return_dict=True, **batch)["logits"]

            # Each row holds the three dimensions: VQ, MQ, TA. The module declares a
            # single overall reward, defined (per the reference inference) as the sum.
            reward = rewards[0]
            vq = float(reward[0].item())
            mq = float(reward[1].item())
            ta = float(reward[2].item())
            sample.quality_metrics.video_reward_score = vq + mq + ta
        except Exception as e:
            logger.warning("VideoAlign processing failed: %s", e)
        return sample

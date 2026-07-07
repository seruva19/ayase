"""VideoReward — Kling Multi-Dimensional Reward Model (NeurIPS 2025).

HuggingFace: https://huggingface.co/KlingTeam/VideoReward

The published VideoReward model is a Qwen2-VL-based reward model producing
Visual Quality (VQ), Motion Quality (MQ) and Text Alignment (TA) rewards.
Encoding frames with CLIP and comparing them to "high quality" vs "blurry"
text prompts is a proxy, not VideoReward, so it is not emitted under the
VideoReward name.

The real KlingTeam/VideoReward model is loaded by the sibling ``video_reward``
module (AutoModelForSequenceClassification). This module stays unavailable
rather than emit CLIP-prompt proxy values under the VideoReward fields.

videoreward_vq / videoreward_mq / videoreward_ta — populated only with a real
backend.
"""

import logging
from typing import Dict, Optional

from ayase.models import Sample, QualityMetrics  # noqa: F401 (kept for API parity)
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VideoRewardModule(PipelineModule):
    name = "videoreward"
    provisional = True  # no turnkey real backend in a standard install
    description = "VideoReward Kling multi-dim reward model (NeurIPS 2025)"
    default_config = {
        "subsample": 8,
        "clip_model": "openai/clip-vit-base-patch32",
    }
    metric_groups = {
        "videoreward_mq": "motion",
        "videoreward_ta": "alignment",
        "videoreward_vq": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = None
        self._ml_available = False

    def setup(self) -> None:
        if self.test_mode:
            return
        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "VideoReward unavailable: the Qwen2-VL VideoReward model is not wired "
            "here (use the 'video_reward' module for the real KlingTeam/VideoReward "
            "backend); videoreward_vq/mq/ta will not be populated by this module."
        )

    def process(self, sample: Sample) -> Sample:
        return sample

    def _compute_rewards(self, frames, sample: Sample) -> Dict[str, Optional[float]]:
        return {"vq": None, "mq": None, "ta": None}

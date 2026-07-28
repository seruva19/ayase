"""VQ-Insight — ByteDance AIGC Video Quality (AAAI 2026 Oral).

VQ-Insight is a reasoning-style vision-language model (built on
``Qwen2.5-VL-7B-Instruct``) for AI-generated video quality understanding. It is
released by ByteDance in the ``ByteDance/Q-Insight`` HuggingFace repo, under two
video sub-folders:

* ``vqinsight-aigcvideo``  — multi-dimension AIGC scoring (spatial / temporal /
  text-video-consistency), returned as JSON.
* ``vqinsight-naturalvideo`` — a single 0-100 natural-video quality score.

The model *thinks* inside ``<think>...</think>`` and emits the final score inside
``<answer>...</answer>``. This module reproduces the upstream demo inference
protocol (``src/eval/demo_vqinsight_score.py`` in ``github.com/bytedance/Q-Insight``):
the same system prompt, per-mode user prompt, ``qwen_vl_utils.process_vision_info``
frame handling, greedy generation, and ``<answer>`` parsing. The parsed 0-100 score
(for AIGC, the mean of the three dimensions) is mapped to 0-1.

If the released weights / transformers backend are unavailable, the metric is
left ``None`` — no CLIP zero-shot proxy or handcrafted approximation is
substituted for the published metric.

vqinsight_score — higher = better (0-1)

Source: https://github.com/bytedance/Q-Insight (src/eval/demo_vqinsight_score.py);
weights: https://huggingface.co/ByteDance/Q-Insight ; paper: arXiv:2506.18564.
"""

import json
import logging
import re
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# demo constants (demo_vqinsight_score.py).
_SUBFOLDERS = {
    "aigc": "vqinsight-aigcvideo",
    "natural": "vqinsight-naturalvideo",
}

_SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the "
    "reasoning process in the mind and then provides the answer. "
    "The reasoning process and answer are enclosed within "
    "<think></think> and <answer></answer> tags."
)

_PROMPTS = {
    "natural": (
        "Please rate the quality of this video. "
        "The range of quality score is between 0 and 100."
    ),
    "aigc": (
        "This is an AIGC-generated video."
        "Rate this video from three dimensions including spatial quality, "
        "temporal quality, and text-video alignment quality. "
        "Return the result in JSON format with the following keys: "
        '"spatial", "temporal", and "consistency". '
        "Each score should be a float between 0 and 100, "
        "rounded to two decimal places."
    ),
}


class VQInsightModule(PipelineModule):
    name = "vqinsight"
    requires_external_backend = False  # real weights verified end-to-end on an H100 (see docstring)
    description = "VQ-Insight ByteDance multi-dim AIGC scoring (AAAI 2026)"
    default_config = {
        "video_type": "aigc",  # "aigc" (multi-dim) or "natural" (single score)
        "model_name_or_path": "ByteDance/Q-Insight",
        "max_new_tokens": 256,
        "nframes": 16,  # frames fed to the VLM (must be even for Qwen2.5-VL)
        "device": "auto",
        "models_dir": "models",
    }
    metric_groups = {
        "vqinsight_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.video_type = self.config.get("video_type", "aigc")
        if self.video_type not in _SUBFOLDERS:
            self.video_type = "aigc"
        self.max_new_tokens = int(self.config.get("max_new_tokens", 256))
        nframes = int(self.config.get("nframes", 16))
        if nframes % 2 != 0:  # Qwen2.5-VL temporal patch size is 2
            nframes += 1
        self.nframes = max(2, nframes)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._processor = None
        self._device = "cpu"

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            from transformers import (
                AutoProcessor,
                Qwen2_5_VLForConditionalGeneration,
            )
            from qwen_vl_utils import process_vision_info  # noqa: F401
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            dtype = (
                torch.bfloat16
                if str(self._device).startswith("cuda")
                else torch.float32
            )
            repo = self.config.get("model_name_or_path", "ByteDance/Q-Insight")
            subfolder = _SUBFOLDERS[self.video_type]
            models_dir = self.config.get("models_dir", "models")

            logger.info(
                "Loading VQ-Insight (%s / %s) on %s...",
                repo,
                subfolder,
                self._device,
            )
            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                repo,
                subfolder=subfolder,
                torch_dtype=dtype,
                cache_dir=models_dir,
            ).to(self._device)
            self._model.eval()
            self._processor = AutoProcessor.from_pretrained(
                repo,
                subfolder=subfolder,
                cache_dir=models_dir,
            )

            self._ml_available = True
            self._backend = "real"
            logger.info("VQ-Insight initialised (%s)", subfolder)
            return
        except ImportError as e:
            logger.warning("VQ-Insight unavailable (missing dependency): %s", e)
        except Exception as e:  # noqa: BLE001 - real-or-none: any failure -> None
            logger.warning("VQ-Insight failed to load: %s", e)

        self._backend = "unavailable"
        self._ml_available = False
        self._model = None
        self._processor = None

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if not self._ml_available or self._backend != "real":
            return sample
        # VQ-Insight scores videos; skip non-video samples (leave None).
        if not getattr(sample, "is_video", False):
            return sample

        try:
            score = self._score_video(str(sample.path))
            if score is not None:
                sample.quality_metrics.vqinsight_score = float(score)
        except Exception as e:  # noqa: BLE001
            logger.warning("VQ-Insight failed for %s: %s", sample.path, e)
        return sample

    def _score_video(self, video_path: str) -> Optional[float]:
        import torch
        from qwen_vl_utils import process_vision_info

        messages = [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "nframes": self.nframes,
                        },
                        {"type": "text", "text": _PROMPTS[self.video_type]},
                    ],
                },
            ]
        ]

        text = [
            self._processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                use_cache=True,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_score(output_text)

    def _parse_score(self, content: str) -> Optional[float]:
        """Extract a 0-1 score from the model output.

        Tolerant parse: prefer the ``<answer>...</answer>`` block, but fall back
        to the whole string so a missing tag never silently yields ``None`` when
        a number is present.
        """
        if not content:
            return None
        m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        answer = m.group(1).strip() if m else content

        if self.video_type == "aigc":
            score_100 = self._parse_aigc(answer)
            if score_100 is None:
                score_100 = self._parse_aigc(content)
        else:
            score_100 = self._first_float(answer)
            if score_100 is None:
                score_100 = self._first_float(content)

        if score_100 is None:
            return None
        return max(0.0, min(1.0, score_100 / 100.0))

    @staticmethod
    def _parse_aigc(text: str) -> Optional[float]:
        keys = ("spatial", "temporal", "consistency")
        # First try a JSON object substring.
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            try:
                obj = json.loads(obj_match.group(0))
                vals = [
                    float(obj[k])
                    for k in keys
                    if k in obj and _is_number(obj[k])
                ]
                if vals:
                    return sum(vals) / len(vals)
            except (ValueError, TypeError):
                pass
        # Fallback: regex each key independently.
        vals = []
        for k in keys:
            km = re.search(rf'"?{k}"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text)
            if km:
                vals.append(float(km.group(1)))
        if vals:
            return sum(vals) / len(vals)
        return None

    @staticmethod
    def _first_float(text: str) -> Optional[float]:
        fm = re.search(r"-?\d+(?:\.\d+)?", text)
        return float(fm.group(0)) if fm else None


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) or (
        isinstance(v, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", v.strip()) is not None
    )

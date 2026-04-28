"""VideoScore2 multi-dimensional generative video evaluation module.

3 dimensions: visual quality, text-to-video alignment, and
physical/common-sense consistency. arXiv 2025.
"""

import logging
from string import Template
from typing import Dict, Optional

from ayase.config import resolve_model_path
from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class VideoScore2Module(PipelineModule):
    name = "videoscore2"
    description = "VideoScore2 3-dimensional generative video evaluation"
    default_config = {
        "model_name": "TIGER-Lab/VideoScore2",
        "infer_fps": 2.0,
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "do_sample": True,
        "trust_remote_code": True,
        "model_revision": None,
    }
    models = [
        {"id": "TIGER-Lab/VideoScore2", "type": "huggingface", "task": "VideoScore2 VLM for 3D generative video evaluation"},
    ]
    metric_info = {
        "videoscore2_visual": "Visual quality subscore (0-10, higher=better)",
        "videoscore2_alignment": "Text-to-video alignment subscore (0-10, higher=better)",
        "videoscore2_physical": "Physical/common-sense consistency subscore (0-10, higher=better)",
    }

    _QUERY_TEMPLATE = Template(
        "You are an expert for evaluating AI-generated videos from three dimensions: "
        "(1) visual quality - clarity, smoothness, artifacts; "
        "(2) text-to-video alignment - fidelity to the prompt; "
        "(3) physical/common-sense consistency - naturalness and physics plausibility.\n\n"
        "Video prompt: $t2v_prompt\n"
        "Please output in this format: visual quality: ; "
        "text-to-video alignment: ; physical/common-sense consistency: "
    )

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._backend = None
        self._device = "cpu"
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._process_vision_info = None

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
            from qwen_vl_utils import process_vision_info

            model_name = self.config.get("model_name", "TIGER-Lab/VideoScore2")
            models_dir = self.config.get("models_dir", "models")
            resolved = resolve_model_path(model_name, models_dir)
            revision = self.config.get("model_revision", None)
            trust_remote_code = self.config.get("trust_remote_code", True)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            elif torch.cuda.is_available():
                dtype = torch.float16
            else:
                dtype = torch.float32

            self._processor = AutoProcessor.from_pretrained(
                resolved,
                trust_remote_code=trust_remote_code,
                revision=revision,
            )
            self._tokenizer = getattr(self._processor, "tokenizer", None) or AutoTokenizer.from_pretrained(
                resolved,
                trust_remote_code=trust_remote_code,
                revision=revision,
                use_fast=False,
            )
            self._model = AutoModelForVision2Seq.from_pretrained(
                resolved,
                trust_remote_code=trust_remote_code,
                revision=revision,
                torch_dtype=dtype,
            ).to(self._device)
            self._model.eval()
            self._process_vision_info = process_vision_info
            self._backend = "transformers"
            logger.info("VideoScore2 model loaded on %s", self._device)
        except (ImportError, Exception) as e:
            logger.warning("VideoScore2 unavailable: %s", e)

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        if self._backend is None:
            return sample

        try:
            scores = self._compute_scores(sample)
            if scores:
                sample.quality_metrics.videoscore2_visual = scores.get("visual_quality")
                sample.quality_metrics.videoscore2_alignment = scores.get("text_to_video_alignment")
                sample.quality_metrics.videoscore2_physical = scores.get(
                    "physical_common_sense_consistency"
                )
        except Exception as e:
            logger.warning("VideoScore2 processing failed: %s", e)
        return sample

    def _compute_scores(self, sample: Sample) -> Optional[Dict[str, float]]:
        import re

        infer_fps = self.config.get("infer_fps", 2.0)
        caption = sample.caption.text if sample.caption and sample.caption.text else "a video"
        user_prompt = self._QUERY_TEMPLATE.substitute(t2v_prompt=caption)
        messages = [{
            "role": "user",
            "content": [
                self._build_media_content(sample, infer_fps),
                {"type": "text", "text": user_prompt},
            ],
        }]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)

        processor_kwargs = {
            "text": [text],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if video_inputs:
            processor_kwargs["fps"] = infer_fps

        inputs = self._processor(**processor_kwargs).to(self._device)

        generate_kwargs = {
            "max_new_tokens": self.config.get("max_new_tokens", 1024),
            "output_scores": True,
            "return_dict_in_generate": True,
            "do_sample": self.config.get("do_sample", True),
        }
        if generate_kwargs["do_sample"]:
            generate_kwargs["temperature"] = self.config.get("temperature", 0.7)

        gen_out = self._model.generate(**inputs, **generate_kwargs)
        input_len = inputs["input_ids"].shape[1]
        generated_ids = gen_out.sequences[:, input_len:]
        generated_token_ids = generated_ids[0].tolist()
        output_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        pattern = (
            r"visual quality:\s*(\d+).*?"
            r"text-to-video alignment:\s*(\d+).*?"
            r"physical/common-sense consistency:\s*(\d+)"
        )
        match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
        hard_scores = {
            "visual_quality": int(match.group(1)) if match else None,
            "text_to_video_alignment": int(match.group(2)) if match else None,
            "physical_common_sense_consistency": int(match.group(3)) if match else None,
        }

        prompt_map = {
            "visual_quality": "visual quality:",
            "text_to_video_alignment": "text-to-video alignment:",
            "physical_common_sense_consistency": "physical/common-sense consistency:",
        }

        parsed_scores = {}
        for key, prompt_text in prompt_map.items():
            token_idx = self._find_score_token_index_by_prompt(
                prompt_text,
                self._tokenizer,
                generated_token_ids,
            )
            soft_score = self._ll_based_soft_score_normed(
                hard_scores[key],
                token_idx,
                gen_out.scores,
                self._tokenizer,
            )
            final_score = soft_score if soft_score is not None else hard_scores[key]
            if final_score is not None:
                parsed_scores[key] = float(final_score)

        return parsed_scores or None

    def _build_media_content(self, sample: Sample, infer_fps: float) -> Dict[str, object]:
        path = str(sample.path)
        if sample.is_video:
            return {"type": "video", "video": path, "fps": infer_fps}
        return {"type": "image", "image": path}

    def _find_score_token_index_by_prompt(self, prompt_text, tokenizer, generated_token_ids) -> int:
        import re

        generated_text = tokenizer.decode(generated_token_ids, skip_special_tokens=False)
        pattern = r"(?:\(\d+\)\s*|\n\s*)?" + re.escape(prompt_text)
        match = re.search(pattern, generated_text, flags=re.IGNORECASE)
        if not match:
            return -1

        tail = generated_text[match.end():]
        digit_match = re.search(r"\d", tail)
        if not digit_match:
            return -1

        target_text = generated_text[:match.end() + digit_match.start() + 1]
        for idx in range(len(generated_token_ids)):
            partial = tokenizer.decode(generated_token_ids[: idx + 1], skip_special_tokens=False)
            if partial == target_text:
                return idx
        return -1

    def _ll_based_soft_score_normed(self, hard_value, token_idx, scores, tokenizer) -> Optional[float]:
        import numpy as np
        import torch

        if hard_value is None or token_idx < 0 or token_idx >= len(scores):
            return None

        logits = scores[token_idx][0]
        score_probs = []
        for score_value in range(1, 6):
            token_ids = tokenizer.encode(str(score_value), add_special_tokens=False)
            if len(token_ids) != 1:
                continue
            token_id = token_ids[0]
            log_prob = torch.log_softmax(logits, dim=-1)[token_id].item()
            score_probs.append((score_value, float(np.exp(log_prob))))

        if not score_probs:
            return None

        discrete_scores, probs = zip(*score_probs)
        total_prob = sum(probs)
        if total_prob <= 0:
            return None
        max_prob = max(probs)
        best_score = discrete_scores[probs.index(max_prob)]
        return round(best_score * (max_prob / total_prob), 4)

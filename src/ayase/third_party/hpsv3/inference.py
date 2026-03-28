"""HPSv3 inference utilities.

This module provides the prompt-conditioned image and frame scoring path used by
Ayase for HPSv3.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
CHECKPOINT_REPO = "MizzenAI/HPSv3"
CHECKPOINT_FILE = "HPSv3.safetensors"

INSTRUCTION = """
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best.

**Visual Quality:**
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

**Text Alignment:**
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}
"""

PROMPT_WITH_SPECIAL_TOKEN = """
Please provide the overall ratings of this image: <|Reward|>

END
"""

IMAGE_FACTOR = 28
FIXED_PIXELS = 256 * 28 * 28
MAX_RATIO = 200


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = FIXED_PIXELS,
    max_pixels: int = FIXED_PIXELS,
) -> tuple[int, int]:
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _fetch_image(image: str | Path | Image.Image) -> Image.Image:
    if isinstance(image, Image.Image):
        image_obj = image.convert("RGB")
    else:
        image_obj = Image.open(image).convert("RGB")
    width, height = image_obj.size
    resized_height, resized_width = _smart_resize(height, width)
    return image_obj.resize((resized_width, resized_height), Image.BICUBIC)


class Qwen2VLRewardModel(Qwen2VLForConditionalGeneration):
    def __init__(
        self,
        config,
        output_dim: int = 2,
        reward_token: str = "special",
        special_token_ids: Optional[Iterable[int]] = None,
        rm_head_type: str = "ranknet",
        rm_head_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.output_dim = output_dim
        self.reward_token = reward_token
        self.special_token_ids = list(special_token_ids or [])
        if self.special_token_ids:
            self.reward_token = "special"

        if rm_head_type == "default":
            self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        elif rm_head_type == "ranknet":
            if rm_head_kwargs is not None:
                layers: list[nn.Module] = []
                hidden_size = rm_head_kwargs["hidden_size"]
                num_layers = rm_head_kwargs.get("num_layers", 3)
                dropout = rm_head_kwargs.get("dropout", 0.1)
                bias = rm_head_kwargs.get("bias", False)
                for layer_index in range(num_layers):
                    if layer_index == 0:
                        layers.extend(
                            [
                                nn.Linear(config.hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                            ]
                        )
                    elif layer_index < num_layers - 1:
                        layers.extend(
                            [
                                nn.Linear(hidden_size, hidden_size),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                            ]
                        )
                    else:
                        layers.append(nn.Linear(hidden_size, output_dim, bias=bias))
                self.rm_head = nn.Sequential(*layers)
            else:
                self.rm_head = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                )
        else:
            raise ValueError(f"Unsupported rm_head_type: {rm_head_type}")

        self.rm_head.to(torch.float32)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.rm_head(hidden_states.to(torch.float32))

        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = (sequence_lengths % input_ids.shape[-1]).to(logits.device)

        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack(
                [logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)]
            )
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask |= input_ids == special_token_id
            pooled_logits = logits[special_token_mask, ...].view(batch_size, 1, -1).view(batch_size, -1)
        else:
            raise ValueError(f"Invalid reward_token: {self.reward_token}")

        return {"logits": pooled_logits}


class HPSv3RewardInferencer:
    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        cache_dir: str | Path | None = None,
    ) -> None:
        self.device = device
        self.use_special_tokens = True

        cache_dir_str = str(cache_dir) if cache_dir else None
        if checkpoint_path is None:
            checkpoint_path = hf_hub_download(
                repo_id=CHECKPOINT_REPO,
                filename=CHECKPOINT_FILE,
                repo_type="model",
                cache_dir=cache_dir_str,
            )

        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            padding_side="right",
            cache_dir=cache_dir_str,
        )
        special_tokens = ["<|Reward|>"]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        special_token_ids = processor.tokenizer.convert_tokens_to_ids(special_tokens)

        torch_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        model = Qwen2VLRewardModel.from_pretrained(
            MODEL_NAME,
            output_dim=2,
            reward_token="special",
            special_token_ids=special_token_ids,
            rm_head_type="ranknet",
            torch_dtype=torch_dtype,
            attn_implementation="sdpa",
            use_cache=False,
            cache_dir=cache_dir_str,
        )
        model.resize_token_embeddings(len(processor.tokenizer))

        import safetensors.torch

        state_dict = safetensors.torch.load_file(str(checkpoint_path), device="cpu")
        if "model" in state_dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        model.eval().to(device)

        self.model = model
        self.processor = processor

    def _prepare_input(self, data):
        if isinstance(data, Mapping):
            return type(data)({key: self._prepare_input(value) for key, value in data.items()})
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(value) for value in data)
        if isinstance(data, torch.Tensor):
            return data.to(device=self.device)
        return data

    def _prepare_inputs(self, inputs):
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError("Empty HPSv3 batch.")
        return inputs

    def prepare_batch(self, image_paths: Iterable[str | Path | Image.Image], prompts: Iterable[str]):
        messages = []
        for prompt, image in zip(prompts, image_paths):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                                "min_pixels": FIXED_PIXELS,
                                "max_pixels": FIXED_PIXELS,
                            },
                            {
                                "type": "text",
                                "text": INSTRUCTION.format(text_prompt=prompt) + PROMPT_WITH_SPECIAL_TOKEN,
                            },
                        ],
                    }
                ]
            )

        image_inputs = [_fetch_image(message[0]["content"][0]["image"]) for message in messages]
        batch = self.processor(
            text=self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        return self._prepare_inputs(batch)

    def reward(self, prompts: Iterable[str], image_paths: Iterable[str | Path | Image.Image]):
        batch = self.prepare_batch(image_paths=image_paths, prompts=prompts)
        return self.model(return_dict=True, **batch)["logits"]

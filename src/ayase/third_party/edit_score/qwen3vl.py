"""Pinned native Qwen3-VL adapter for EditScore."""

import random

import numpy as np


class Qwen3VL:
    def __init__(self, vlm_model, temperature=0.7, seed=42, lora_path=None):
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
        )
        if lora_path:
            model = PeftModel.from_pretrained(
                model, lora_path, local_files_only=True
            ).merge_and_unload()
        self.model = model.eval()
        self.processor = AutoProcessor.from_pretrained(
            vlm_model, local_files_only=True
        )
        self.temperature = temperature
        self.seed = seed

    def prepare_input(self, images, text_prompt=""):
        if not isinstance(images, list):
            images = [images]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image} for image in images
                ]
                + [{"type": "text", "text": text_prompt}],
            }
        ]
        return self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda")

    def inference(self, inputs, seed=None):
        import torch

        seed = self.seed if seed is None else seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generated = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9,
            top_k=20,
        )
        trimmed = [
            output[len(prompt) :]
            for prompt, output in zip(inputs.input_ids, generated)
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

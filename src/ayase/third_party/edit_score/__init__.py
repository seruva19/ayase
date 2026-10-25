"""Pinned EditScore inference core with strict score parsing.

Adapted from VectorSpaceLab/EditScore at commit
4609c5d2ebb62fdebf665d3c924686d896ef1f74. See LICENSE and NOTICE.
"""

import math

import numpy as np

from . import vie_prompts
from .utils import mllm_output_to_dict


class EditScore:
    """Source-compatible single-sample EditScore evaluator."""

    def __init__(
        self,
        backbone="qwen3vl",
        model_name_or_path="",
        score_range=25,
        temperature=0.7,
        num_pass=1,
        reduction="average_last",
        seed=42,
        lora_path=None,
    ):
        if backbone != "qwen3vl":
            raise ValueError("Ayase's pinned EditScore supports only qwen3vl")
        from .qwen3vl import Qwen3VL

        self.backbone = backbone
        self.score_range = score_range
        self.reduction = reduction
        self.seed = seed
        self.num_pass = num_pass
        self.model = Qwen3VL(
            model_name_or_path,
            temperature=temperature,
            seed=seed,
            lora_path=lora_path,
        )
        context = vie_prompts.CONTEXT
        self.SC_prompt = "\n".join(
            [
                context,
                vie_prompts.TWO_IMAGE_RULE,
                vie_prompts.SC_RULE.replace("10", str(score_range)),
            ]
        )
        self.PQ_prompt = "\n".join(
            [context, vie_prompts.PQ_RULE.replace("10", str(score_range))]
        )

    def evaluate(self, image_prompts, text_prompt):
        if not isinstance(image_prompts, list):
            image_prompts = [image_prompts]
        if len(image_prompts) != 2:
            raise ValueError("EditScore requires exactly [source, edited] images")

        _SC_prompt = self.SC_prompt.replace("<instruction>", text_prompt)

        SC_prompt_final = self.model.prepare_input(image_prompts, _SC_prompt)
        PQ_prompt_final = self.model.prepare_input(image_prompts[-1], self.PQ_prompt)

        outputs_multi_pass = []

        for i in range(self.num_pass):
            SC_dict = False
            PQ_dict = False
            tries = 0
            max_tries = 2
            while SC_dict is False or PQ_dict is False:
                tries += 1
                give_up_parsing = True if tries > max_tries else False

                result_SC = self.model.inference(SC_prompt_final, seed=self.seed + i)
                result_PQ = self.model.inference(PQ_prompt_final, seed=self.seed + i)

                if result_SC in ["I'm sorry, but I can't assist with that request."] or result_PQ in ["I'm sorry, but I can't assist with that request."]:
                    give_up_parsing = True

                SC_dict = mllm_output_to_dict(result_SC, give_up_parsing=give_up_parsing, text_prompt=text_prompt, score_range=self.score_range)
                PQ_dict = mllm_output_to_dict(result_PQ, give_up_parsing=give_up_parsing, text_prompt=text_prompt, score_range=self.score_range)

            if SC_dict == "rate_limit_exceeded" or PQ_dict == "rate_limit_exceeded":
                print("rate_limit_exceeded")
                raise ValueError("rate_limit_exceeded")

            try:
                SC_score = min(SC_dict['score']) / (self.score_range / 10)
                PQ_score = min(PQ_dict['score']) / (self.score_range / 10)
                O_score = math.sqrt(SC_score * PQ_score)
            except Exception as e:
                print(f"{e=} {SC_dict['score']=} {PQ_dict['score']=}")
                raise e

            try:
                outputs_multi_pass.append({
                    'prompt_following': SC_dict['score'][0] / (self.score_range / 10),
                    'consistency': SC_dict['score'][1] / (self.score_range / 10),
                    'perceptual_quality': PQ_score,
                    'overall': O_score,
                })
            except Exception as e:
                print(f"{e=} {SC_dict['score']=} {PQ_dict['score']=}")
                raise e

        output = {
                    "prompt_following": np.mean([output_per_pass["prompt_following"] for output_per_pass in outputs_multi_pass]),
                    "consistency": np.mean([output_per_pass["consistency"] for output_per_pass in outputs_multi_pass]),
                    "perceptual_quality": np.mean([output_per_pass["perceptual_quality"] for output_per_pass in outputs_multi_pass]),
                    "overall": np.mean([output_per_pass["overall"] for output_per_pass in outputs_multi_pass]),
                    "SC_reasoning": SC_dict["reasoning"],
                    "PQ_reasoning": PQ_dict["reasoning"],
                }
        if self.reduction == "average_first":
            output["overall"] = math.sqrt(output["prompt_following"] * output["perceptual_quality"])
        return output


__all__ = ["EditScore"]

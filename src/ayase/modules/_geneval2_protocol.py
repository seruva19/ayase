"""Python 3.9-compatible port of the GenEval2 Soft-TIFA protocol.

Adapted from Meta Platforms, Inc. and affiliates' GenEval2 evaluator under
CC BY-NC 4.0. The retained source and license are in ``third_party/geneval2``.
"""

# SPDX-License-Identifier: CC-BY-NC-4.0

import math
from typing import Any, List, Sequence, Tuple


_NUMERIC = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def answer_variants(question: str, answer: str) -> List[str]:
    """Return the exact answer variants used by upstream GenEval2."""
    if question.startswith("How many"):
        numeric = _NUMERIC.get(answer, "other")
        return [
            answer,
            answer.capitalize(),
            " " + answer,
            " " + answer.capitalize(),
            numeric,
            " " + numeric,
        ]
    return ["Yes", "yes", " yes", " Yes"]


def score_question(
    processor: Any,
    model: Any,
    torch: Any,
    image_path: str,
    question: str,
    answer: str,
) -> float:
    """Score one VQA atom with upstream's one-token full-vocabulary protocol."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question + " Answer in one word."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    probabilities = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
    probability = 0.0
    for variant in answer_variants(question, answer):
        token_id = processor.tokenizer.encode(variant)[0]
        probability += probabilities[0, token_id].item()
    return probability


def aggregate_atom_scores(scores: Sequence[float]) -> Tuple[float, float]:
    """Return per-prompt arithmetic and geometric means without epsilon flooring."""
    if not scores:
        raise ValueError("Soft-TIFA requires at least one atom score")
    arithmetic = sum(scores) / len(scores)
    geometric = 0.0 if any(score == 0 for score in scores) else math.exp(
        sum(math.log(score) for score in scores) / len(scores)
    )
    return arithmetic, geometric

"""Scientific validity gate around EditScore's byte-preserved source parser."""

import math

from . import _upstream_utils

REFUSAL = "I'm sorry, but I can't assist with that request."


def _forbid_zero_padding(_value):
    """Block upstream's final fallback, which pads absent scores with zeros."""
    raise ValueError("EditScore last-resort score repair is disabled")


_upstream_utils.fallback_repair_json = _forbid_zero_padding


def mllm_output_to_dict(
    value, give_up_parsing=False, text_prompt=None, score_range=25
):
    """Run the source parser, rejecting fabricated or scientifically invalid scores."""
    if give_up_parsing:
        raise ValueError("EditScore exhausted parsing retries")
    if not isinstance(value, str) or value.strip() in {
        "",
        REFUSAL,
        "rate_limit_exceeded",
    }:
        raise ValueError("EditScore model refused or returned no usable response")
    parsed = _upstream_utils.mllm_output_to_dict(
        value,
        give_up_parsing=False,
        text_prompt=text_prompt,
        score_range=score_range,
    )
    if parsed is False:
        return False
    if not isinstance(parsed, dict):
        raise ValueError("EditScore response is not an object")
    reasoning = parsed.get("reasoning")
    scores = parsed.get("score")
    if not isinstance(reasoning, str) or reasoning.startswith("guess_if_cannot_parse"):
        raise ValueError("EditScore response has invalid reasoning")
    if not isinstance(scores, list) or len(scores) != 2:
        return False
    if any(
        isinstance(score, bool) or not isinstance(score, (int, float))
        for score in scores
    ):
        return False
    normalized = [float(score) for score in scores]
    if any(
        not math.isfinite(score) or not 0 <= score <= score_range
        for score in normalized
    ):
        return False
    return {"reasoning": reasoning, "score": normalized}


strict_mllm_output_to_dict = mllm_output_to_dict

__all__ = ["mllm_output_to_dict", "strict_mllm_output_to_dict"]

"""Bundled AudioJudge prompts and Ayase protocol adapter."""

from .protocol import (
    PAIRWISE_METHODS,
    POINTWISE_METHODS,
    PairwiseExample,
    PointwiseExample,
    ResponseContract,
    build_gemini_messages,
    build_openai_messages,
    canonical_contract,
    concatenate_pair_audio,
    concatenate_pointwise_audio,
    custom_contract,
    normalize_wav,
    parse_response,
    resolve_signal_asset,
)

__all__ = [
    "PAIRWISE_METHODS",
    "POINTWISE_METHODS",
    "PairwiseExample",
    "PointwiseExample",
    "ResponseContract",
    "build_gemini_messages",
    "build_openai_messages",
    "canonical_contract",
    "concatenate_pair_audio",
    "concatenate_pointwise_audio",
    "custom_contract",
    "normalize_wav",
    "parse_response",
    "resolve_signal_asset",
]

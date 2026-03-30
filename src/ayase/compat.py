"""Compatibility helpers for different library versions."""

from __future__ import annotations

import torch


def extract_features(output: object) -> torch.Tensor:
    """Extract feature tensor from CLIP get_*_features() output.

    transformers v4.x returns torch.Tensor directly.
    transformers v5.x returns BaseModelOutputWithPooling.

    Args:
        output: Return value of get_image_features() or get_text_features().

    Returns:
        torch.Tensor with shape [batch, dim].
    """
    if isinstance(output, torch.Tensor):
        return output
    # transformers v5+: BaseModelOutputWithPooling
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0]
    raise TypeError(f"Cannot extract features from {type(output)}")

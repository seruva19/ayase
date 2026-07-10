"""Vendored GroundingDINO open-vocabulary detector (SwinB variant).

GroundingDINO — "Marrying DINO with Grounded Pre-Training for Open-Set Object
Detection" (IDEA-Research, arXiv:2303.05499). Upstream:
github.com/IDEA-Research/GroundingDINO, licensed Apache-2.0.

This vendored copy is self-contained: it uses the pure-PyTorch multi-scale
deformable attention (no compiled CUDA extension), fetches the SwinB checkpoint
and the ``bert-base-uncased`` text encoder from the Hugging Face Hub at runtime,
and adds no third-party dependencies beyond those already required by the
toolkit. See :mod:`ayase.vendor.groundingdino.api` for the public API.
"""

from .api import GroundingDinoWrapper, GroundingResult, load_grounding_dino

__all__ = ["load_grounding_dino", "GroundingDinoWrapper", "GroundingResult"]

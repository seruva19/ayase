"""Shared strict checkpoint loading for MiDaS-based Ayase metrics."""

from pathlib import Path


def load_midas_model(model_type: str, checkpoint_path: Path, device):
    """Build a MiDaS model and strict-load an Ayase-mirrored checkpoint."""
    import torch

    model = torch.hub.load(
        "intel-isl/MiDaS",
        model_type,
        pretrained=False,
        trust_repo=True,
    )
    state_dict = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()

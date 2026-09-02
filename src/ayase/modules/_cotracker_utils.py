"""Shared CoTracker loading: architecture from torch.hub, weights from HuggingFace.

Three modules track points with CoTracker. Left to itself, ``torch.hub.load`` pulls
the checkpoint from Meta's file server, which is slow or unreachable on some
networks and bypasses the HuggingFace hosting the rest of the pipeline relies on.
The authors publish the same checkpoint on the Hub, so the weights are fetched from
there and placed in the torch.hub checkpoint cache under the exact filename the hub
entrypoint expects; the entrypoint then finds them cached and downloads nothing.

``vbench2`` already loads CoTracker this way. This module exists so the other three
do not each repeat it.
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Official checkpoint published by the authors. Weights live on HuggingFace; the
#: architecture code still comes from the torch.hub repo.
COTRACKER_WEIGHTS_URL = "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth"
COTRACKER_FILENAME = "cotracker2.pth"
COTRACKER_ENTRYPOINT = "cotracker2"


def ensure_cotracker_weights(models_dir: str = "models") -> Optional[Path]:
    """Put the CoTracker checkpoint where the torch.hub entrypoint will find it.

    Args:
        models_dir (str): Directory the pipeline keeps downloaded weights in.

    Returns:
        Optional[Path]: Path to the cached checkpoint, or ``None`` when the
        download failed -- in which case the caller may still proceed and let
        torch.hub fall back to its own source.
    """
    import torch

    from ayase.config import download_model_file

    cache = Path(torch.hub.get_dir()) / "checkpoints" / COTRACKER_FILENAME
    if cache.exists():
        return cache
    try:
        downloaded = Path(
            download_model_file(
                f"cotracker/{COTRACKER_FILENAME}", COTRACKER_WEIGHTS_URL, models_dir
            )
        )
    except Exception as exc:  # pragma: no cover - network failure
        logger.warning("cotracker: could not fetch weights from HuggingFace: %s", exc)
        return None

    cache.parent.mkdir(parents=True, exist_ok=True)
    try:
        # A hard link keeps one copy on disk; a copy is the fallback when the
        # models directory and the hub cache sit on different filesystems.
        import os

        os.link(downloaded, cache)
    except OSError:
        import shutil

        shutil.copy2(downloaded, cache)
    return cache


def load_cotracker(device: Any, models_dir: str = "models") -> Any:
    """Load CoTracker with weights taken from HuggingFace.

    Args:
        device (Any): Torch device to place the model on.
        models_dir (str): Directory the pipeline keeps downloaded weights in.

    Returns:
        Any: The model in evaluation mode.
    """
    import torch

    ensure_cotracker_weights(models_dir)
    model = torch.hub.load("facebookresearch/co-tracker", COTRACKER_ENTRYPOINT)
    return model.to(device).eval()

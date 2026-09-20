"""Load shared CoTracker2 weights for point-tracking callers.

The helper combines Ayase's vendored CoTracker architecture with the authors'
``cotracker2.pth`` checkpoint fetched from HuggingFace and returns an offline
predictor in evaluation mode. It does not track a ``Sample``, emit
``QualityMetrics``, or perform validation; callers own point selection, score
interpretation, and issue semantics.
"""

import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Checkpoint published by the authors. Weights only; the architecture is in-tree.
COTRACKER_WEIGHTS_URL = "https://huggingface.co/facebook/cotracker/resolve/main/cotracker2.pth"
COTRACKER_FILENAME = "cotracker2.pth"
#: Temporal window of the second-generation model, per the upstream entrypoint.
COTRACKER_WINDOW_LEN = 8


def ensure_cotracker_weights(models_dir: str = "models") -> Optional[Path]:
    """Fetch the CoTracker2 checkpoint into ``models_dir``.

    Args:
        models_dir (str): Directory the pipeline keeps downloaded weights in.

    Returns:
        Optional[Path]: Path to the checkpoint, or ``None`` when the download
        failed.
    """
    from ayase.config import download_model_file

    try:
        return Path(
            download_model_file(
                f"cotracker/{COTRACKER_FILENAME}", COTRACKER_WEIGHTS_URL, models_dir
            )
        )
    except Exception as exc:  # pragma: no cover - network failure
        logger.warning("cotracker: could not fetch weights from HuggingFace: %s", exc)
        return None


def load_cotracker(device: Any, models_dir: str = "models") -> Any:
    """Load CoTracker2 from the vendored architecture and the fetched weights.

    Args:
        device (Any): Torch device to place the model on.
        models_dir (str): Directory the pipeline keeps downloaded weights in.

    Returns:
        Any: The predictor in evaluation mode.

    Raises:
        RuntimeError: The checkpoint could not be obtained.
    """
    from ayase.vendor.cotracker.predictor import CoTrackerPredictor

    checkpoint = ensure_cotracker_weights(models_dir)
    if checkpoint is None:
        raise RuntimeError("CoTracker2 checkpoint is unavailable")
    predictor = CoTrackerPredictor(
        checkpoint=str(checkpoint),
        v2=True,
        offline=True,
        window_len=COTRACKER_WINDOW_LEN,
    )
    return predictor.to(device).eval()

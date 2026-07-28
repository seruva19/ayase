"""DeSync — Synchformer-predicted audio-visual temporal offset (seconds).

Reports the **DeSync** metric popularised by Movie Gen, MMAudio, and
HunyuanVideo-Foley for video-to-audio / audio-visual generation: the absolute
temporal offset between the audio and video streams as predicted by
**Synchformer** (Iashin, Xie, Rahtu, Zisserman — "Synchformer: Efficient
Synchronization from Sparse Cues", ICASSP 2024;
https://github.com/v-iashin/Synchformer). Lower = better synchronised.

Synchformer classifies the A/V offset over a ±2 s grid (21 classes at 0.2 s
steps) and returns the ``argmax`` offset in seconds; DeSync is its absolute
value. Following the MMAudio evaluation protocol, DeSync is the *absolute*
predicted offset (MMAudio averages ``|offset|`` over the first/last windows of
a clip — see fidelity notes in the accompanying tests).

Only the real Synchformer backend produces this metric. There is deliberately
**no** energy-correlation heuristic here — the honest energy cross-correlation
proxy lives in the separate ``audio_visual_sync`` module. When Synchformer
weights (or a working CUDA device) are unavailable the backend is
``"unavailable"`` and ``desync_score`` is left ``None``.

The Synchformer weights are vendored/loaded exactly as by the ``av_sync``
module (``ayase/vendor/verse_bench/syncformer``); the checkpoint is fetched
from the Ayase HuggingFace mirror on first real use.
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
from ayase.runtime import resolve_torch_device

logger = logging.getLogger(__name__)


def _has_audio_stream(video_path: str) -> bool:
    """Return True iff *video_path* contains an audio stream (via ffprobe)."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return "audio" in result.stdout.lower()
    except Exception:
        return False


class AVDesyncModule(PipelineModule):
    name = "av_desync"
    description = (
        "DeSync — Synchformer |predicted A/V offset| in seconds "
        "(Movie Gen / MMAudio / HunyuanVideo-Foley; real model only, lower=better)"
    )
    default_config = {
        "device": "auto",
        "syncformer_model_path": None,  # defaults to <models_dir>/syncformer
        "allow_download": True,         # fetch weights from the HF mirror on first use
    }
    models = [
        {
            "id": "Synchformer",
            "type": "local",
            "task": "Audio-visual offset prediction (v-iashin/Synchformer, ±2s / 21-class grid)",
        },
    ]
    metric_info = {
        "desync_score": "Synchformer |predicted A/V offset| in seconds (lower=better)",
    }
    metric_groups = {
        "desync_score": "audio",
    }

    # Same checkpoint / mirror the av_sync module uses for the vendored inferencer.
    _WEIGHTS_URLS = {
        "24-01-04T16-39-21.pt": (
            "https://huggingface.co/AkaneTendo25/ayase-runtime-assets/resolve/main/"
            "synchformer/24-01-04T16-39-21.pt"
        ),
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._inferencer = None
        self._backend = "unavailable"
        self._device = "cpu"

    # ------------------------------------------------------------------
    def _ensure_synchformer_weights(self, model_path: Path) -> bool:
        """Pre-position ``<model_path>/24-01-04T16-39-21.pt`` from the Ayase HF
        mirror. The vendored ``SyncformerInferencer`` expects the file in place
        (it does NOT auto-download). Returns True iff weights are present. No
        download is attempted when ``allow_download`` is False."""
        target = Path(model_path) / "24-01-04T16-39-21.pt"
        if target.exists() and target.stat().st_size > 1_000_000:
            return True
        if not self.config.get("allow_download", True):
            return False
        url = self._WEIGHTS_URLS.get("24-01-04T16-39-21.pt")
        if not url:
            return False
        try:
            from ayase.config import download_model_file

            download_model_file("24-01-04T16-39-21.pt", url, str(model_path))
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Synchformer weights download failed: %s", e)
            return False
        return target.exists() and target.stat().st_size > 1_000_000

    def setup(self) -> None:
        self._device = resolve_torch_device(self.config.get("device", "auto"))
        try:
            vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "verse_bench"
            if str(vendor_root) not in sys.path:
                sys.path.insert(0, str(vendor_root))
            from syncformer.syncformer_inferencer import SyncformerInferencer

            models_dir = Path(self.config.get("models_dir", "models"))
            model_path = Path(
                self.config.get("syncformer_model_path") or (models_dir / "syncformer")
            )
            model_path.mkdir(parents=True, exist_ok=True)
            if not self._ensure_synchformer_weights(model_path):
                logger.warning(
                    "av_desync: Synchformer weights unavailable "
                    "(cannot fetch AkaneTendo25/ayase-runtime-assets mirror); "
                    "desync_score left unset."
                )
                self._backend = "unavailable"
                return

            self._inferencer = SyncformerInferencer(str(model_path))
            self._backend = "synchformer"
            logger.info("av_desync initialised with Synchformer backend on %s", self._device)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning(
                "av_desync: Synchformer backend unavailable (%s); desync_score left unset.",
                e,
            )
            self._backend = "unavailable"
            self._inferencer = None

    # ------------------------------------------------------------------
    def process(self, sample: Sample) -> Sample:
        if not sample.is_video:
            return sample

        if not _has_audio_stream(str(sample.path)):
            logger.debug("av_desync: no audio stream in %s, skipping", sample.path.name)
            return sample

        if self._backend != "synchformer" or self._inferencer is None:
            return sample

        try:
            desync = self._compute_desync(sample.path)
            if desync is None:
                return sample
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.desync_score = desync
            logger.debug("av_desync for %s: DeSync=%.3fs", sample.path.name, desync)
        except Exception as e:  # pylint: disable=broad-except
            logger.error("av_desync failed for %s: %s", sample.path, e)
        return sample

    def _compute_desync(self, video_path: Path) -> Optional[float]:
        """Run Synchformer and return DeSync = |predicted offset| in seconds."""
        try:
            import torch

            with torch.inference_mode():
                offset_sec = float(self._inferencer.infer(str(video_path)))
            return abs(offset_sec)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Synchformer inference failed for %s: %s", video_path, e)
            return None

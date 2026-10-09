"""SyncNet entry point: tracked face crops to per-track LSE values."""

import gc
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import torch

from .face_tracks import FaceTrackParams, extract_face_tracks
from .model import SyncNetModel, load_syncnet
from .s3fd import S3FD
from .scoring import DEFAULT_BATCH_SIZE, DEFAULT_VSHIFT, TrackScore, score_track

logger = logging.getLogger(__name__)

SYNCNET_WEIGHTS_FILE = "syncnet_v2.model"
S3FD_WEIGHTS_FILE = "sfd_face.pth"


@contextmanager
def deterministic_backends() -> Iterator[None]:
    """Disable TF32 and make cuDNN deterministic while scoring; restore afterwards."""
    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
        ) = previous


class Wav2LipLipSyncScorer:
    """SyncNet and S3FD loaded once; ``score_video`` returns one ``TrackScore`` per face track."""

    def __init__(self, syncnet: SyncNetModel, detector: S3FD, device: torch.device) -> None:
        self._syncnet: Optional[SyncNetModel] = syncnet
        self._detector: Optional[S3FD] = detector
        self.device = device

    @classmethod
    def load(
        cls, syncnet_weights: Path, s3fd_weights: Path, device: torch.device
    ) -> "Wav2LipLipSyncScorer":
        logger.info("Loading SyncNet and S3FD on %s", device)
        return cls(load_syncnet(syncnet_weights, device), S3FD(s3fd_weights, device), device)

    def score_video(
        self,
        video_path: Path,
        params: FaceTrackParams,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        vshift: int = DEFAULT_VSHIFT,
    ) -> List[TrackScore]:
        """Score every face track of ``video_path``; an empty list means no usable track."""
        if self._syncnet is None or self._detector is None:
            raise RuntimeError("scorer is closed")

        with tempfile.TemporaryDirectory(prefix="ayase_lse_") as tmp, deterministic_backends():
            work_dir = Path(tmp)
            crops = extract_face_tracks(video_path, work_dir, self._detector, params)
            logger.debug("%s: %d face tracks", video_path.name, len(crops))
            return [
                score_track(
                    self._syncnet,
                    crop,
                    work_dir / "pytmp" / crop.stem,
                    self.device,
                    batch_size=batch_size,
                    vshift=vshift,
                )
                for crop in crops
            ]

    def close(self) -> None:
        self._syncnet = None
        self._detector = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

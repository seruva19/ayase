"""Reference-free audio-visual lip synchronization metrics.

``lip_sync_verse`` retains the VERSE inferencer path. ``lip_sync_syncnet`` uses
the MIT SyncNet/S3FD source port, including 25 fps preparation, tracked face
crops, MFCC windows, and per-track SyncNet scoring. LSE-C is confidence (higher
is better) and LSE-D is distance (lower is better).

The unregistered ``LipSyncModule`` facade preserves the ``lip_sync`` request
and its ``verse_bench`` / ``wav2lip`` protocol tokens while returning one of
the two explicitly named canonical modules.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Weights are mirrored under <models_dir>/lip_sync/ and fetched on first use,
# matching the other weight-backed modules.
_MODELS_BASE = "https://huggingface.co/AkaneTendo25/ayase-assets/resolve/main/"
_SYNCNET_REL = "lip_sync/syncnet_v2.model"
_S3FD_REL = "lip_sync/sfd_face.pth"
_S3FD_URL = "https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth"
# Exact sizes of the original files (the mirror copy of syncnet_v2.model is identical);
# a different size means a broken download.
_SYNCNET_SIZE = 54_573_114
_S3FD_SIZE = 89_844_381
_SYNCNET_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
# Filled only from a verified primary/cache byte identity; size alone is not acceptance.
_S3FD_SHA256 = "d54a87c2b7543b64729c9a25eafd188da15fd3f6e02f0ecec76ae1b30d86c491"

# A fetcher takes (relative path under models_dir, url, expected size) and returns the
# local path, or None when the file could not be obtained or has the wrong size.
WeightFetcher = Callable[[str, str, Optional[int]], Optional[Path]]
# (LSE-C, LSE-D) of one clip.
LseScores = Tuple[float, float]


def _megabytes(size_bytes: int) -> str:
    return f"{size_bytes / 1e6:.1f} MB"


class LipSyncProtocol:
    """Interface of a face-preparation + scoring protocol.

    ``load`` obtains weights through ``fetch`` and builds the backend; it returns
    ``False`` (after logging why) when the protocol cannot run. ``score`` returns
    ``(LSE-C, LSE-D)`` or ``None`` when the clip has no usable talking face.
    """

    name = ""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def load(self, fetch: WeightFetcher) -> bool:
        raise NotImplementedError

    def score(self, video_path: Path) -> Optional[LseScores]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class Wav2LipProtocol(LipSyncProtocol):
    """Compatibility-token implementation using MIT SyncNet and S3FD source."""

    name = "wav2lip"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.device = str(config.get("device", "auto"))
        self.min_face_size = int(config.get("min_face_size", 100))
        self.facedet_scale = float(config.get("facedet_scale", 0.25))
        self.min_track = int(config.get("min_track", 100))
        self.crop_scale = float(config.get("crop_scale", 0.40))
        self.num_failed_det = int(config.get("num_failed_det", 25))
        self.batch_size = int(config.get("batch_size", 20))
        self.vshift = int(config.get("vshift", 15))
        self._scorer = None
        self._track_params = None

    def load(self, fetch: WeightFetcher) -> bool:
        try:
            import torch

            from ayase.runtime import resolve_torch_device
            from ayase.vendor.syncnet_python.face_tracks import FaceTrackParams
            from ayase.vendor.syncnet_python.protocol import Wav2LipLipSyncScorer
        except Exception as e:
            logger.warning("Lip Sync: wav2lip backend import failed (%s); LSE left unset", e)
            return False

        syncnet_path = fetch(_SYNCNET_REL, _MODELS_BASE + _SYNCNET_REL, _SYNCNET_SIZE)
        s3fd_path = fetch(_S3FD_REL, _S3FD_URL, _S3FD_SIZE)
        if syncnet_path is None or s3fd_path is None:
            return False

        device_name = resolve_torch_device(self.device)
        try:
            self._scorer = Wav2LipLipSyncScorer.load(
                syncnet_path, s3fd_path, torch.device(device_name)
            )
        except Exception as e:
            logger.warning("Lip Sync: wav2lip backend init failed (%s); LSE left unset", e)
            return False

        self._track_params = FaceTrackParams(
            facedet_scale=self.facedet_scale,
            crop_scale=self.crop_scale,
            min_track=self.min_track,
            num_failed_det=self.num_failed_det,
            min_face_size=self.min_face_size,
        )
        logger.info("Lip Sync module initialised (wav2lip protocol: S3FD tracks + SyncNet on %s)", device_name)
        return True

    def score(self, video_path: Path) -> Optional[LseScores]:
        """``(LSE-C, LSE-D)`` averaged over face tracks, or ``None`` without a usable track."""
        tracks = self._scorer.score_video(
            Path(video_path), self._track_params, batch_size=self.batch_size, vshift=self.vshift
        )
        if not tracks:
            return None
        lse_c = sum(t.lse_c for t in tracks) / len(tracks)
        lse_d = sum(t.lse_d for t in tracks) / len(tracks)
        return lse_c, lse_d

    def close(self) -> None:
        if self._scorer is not None:
            self._scorer.close()
            self._scorer = None


class VerseBenchProtocol(LipSyncProtocol):
    """insightface crops + SyncNet scoring as bundled with the VERSE-Bench inferencer."""

    name = "verse_bench"

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._inferencer = None

    def load(self, fetch: WeightFetcher) -> bool:
        # The VERSE-Bench SyncNet implementation is bundled under the verse_bench
        # vendor tree; add its root to the import path so the ``syncnet`` package
        # resolves to it (not an external install).
        vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "verse_bench"
        if not vendor_root.exists():
            logger.warning("Lip Sync: bundled SyncNet not found at %s; LSE left unset", vendor_root)
            return False
        vendor_root_str = str(vendor_root)
        if vendor_root_str not in sys.path:
            sys.path.insert(0, vendor_root_str)

        try:
            from syncnet.syncnet_inferencer import SyncnetInferencer
        except Exception as e:
            logger.warning("Lip Sync: SyncNet backend import failed (%s); LSE left unset", e)
            return False

        weight_path = fetch(_SYNCNET_REL, _MODELS_BASE + _SYNCNET_REL, _SYNCNET_SIZE)
        if weight_path is None:
            return False

        # SyncnetInferencer loads ``<model_dir>/syncnet_v2.model``, so hand it
        # the directory the weight was cached into.
        try:
            self._inferencer = SyncnetInferencer(str(Path(weight_path).parent))
        except Exception as e:
            logger.warning("Lip Sync: SyncNet inferencer init failed (%s); LSE left unset", e)
            return False

        logger.info("Lip Sync module initialised (verse_bench protocol: insightface + SyncNet)")
        return True

    def score(self, video_path: Path) -> Optional[LseScores]:
        # infer -> (offset, conf, dists); conf is LSE-C, min(dists) is LSE-D.
        offset, conf, dists = self._inferencer.infer(str(video_path))
        if conf is None or dists is None or len(dists) == 0:
            return None
        return float(conf), float(min(dists))

    def close(self) -> None:
        self._inferencer = None


#: Registered protocols by their config name. Register a new ``LipSyncProtocol``
#: subclass here to make it selectable through ``protocol``.
PROTOCOLS: Dict[str, Type[LipSyncProtocol]] = {
    Wav2LipProtocol.name: Wav2LipProtocol,
    VerseBenchProtocol.name: VerseBenchProtocol,
}
DEFAULT_PROTOCOL = VerseBenchProtocol.name



class _CanonicalLipSyncModule(PipelineModule):
    """Shared lifecycle for the two explicitly named lip-sync protocols."""

    name = "unnamed_module"
    provenance = "adapted"
    default_config = {"models_dir": "models", "device": "auto"}
    protocol_class: Type[LipSyncProtocol]
    canonical_fields: Tuple[str, str]
    protocol_token: str

    def __init__(self, config=None):
        super().__init__(config)
        self.models_dir = str(self.config.get("models_dir", "models"))
        self._impl: Optional[LipSyncProtocol] = None
        self._ml_available = False
        self._backend = None

    def _fetch_weight(self, relative_path: str, url: str, expected_size: Optional[int]) -> Optional[Path]:
        import hashlib
        from ayase.config import download_model_file, resolve_assets_url

        try:
            path = download_model_file(
                relative_path,
                resolve_assets_url(url, self.config),
                self.models_dir,
            )
        except Exception as exc:
            logger.warning("Lip Sync: could not fetch %s (%s)", relative_path, exc)
            return None
        if expected_size is not None and path.stat().st_size != expected_size:
            logger.warning("Lip Sync: invalid size for %s", path)
            return None
        expected_sha = _SYNCNET_SHA256 if relative_path == _SYNCNET_REL else _S3FD_SHA256
        if expected_sha is None:
            logger.warning("Lip Sync: no approved SHA-256 is configured for %s", relative_path)
            return None
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha:
            logger.warning("Lip Sync: SHA-256 mismatch for %s", path)
            return None
        return path

    def setup(self) -> None:
        if self.test_mode:
            return
        impl = self.protocol_class(self.config)
        if not impl.load(self._fetch_weight):
            impl.close()
            return
        self._impl = impl
        self._ml_available = True
        self._backend = impl.name

    def _assign_scores(
        self, quality_metrics: QualityMetrics, lse_c: float, lse_d: float
    ) -> None:
        raise NotImplementedError

    def process(self, sample: Sample) -> Sample:
        legacy_requested = getattr(self, "_requested_module_name", None) == "lip_sync"
        try:
            if legacy_requested and sample.quality_metrics is not None:
                setter = getattr(
                    sample.quality_metrics, "_set_lip_sync_legacy_protocol", None
                )
                if not callable(setter):
                    raise RuntimeError("QualityMetrics lacks legacy lip-sync protocol support")
                setter(self.protocol_token)
        except Exception as exc:
            logger.error("Lip Sync failed on %s: %s", sample.path.name, exc)
            return sample
        if not sample.is_video or not self._ml_available or self._impl is None:
            return sample
        try:
            scores = self._impl.score(sample.path)
            if scores is None:
                return sample
            lse_c, lse_d = float(scores[0]), float(scores[1])
            created_metrics = sample.quality_metrics is None
            if created_metrics:
                sample.quality_metrics = QualityMetrics()
            self._assign_scores(sample.quality_metrics, lse_c, lse_d)
            if legacy_requested and created_metrics:
                setter = getattr(
                    sample.quality_metrics, "_set_lip_sync_legacy_protocol", None
                )
                if not callable(setter):
                    raise RuntimeError("QualityMetrics lacks legacy lip-sync protocol support")
                setter(self.protocol_token)
        except Exception as exc:
            logger.error("Lip Sync failed on %s: %s", sample.path.name, exc)
            return sample
        return sample

    def on_dispose(self) -> None:
        if self._impl is not None:
            self._impl.close()
        self._impl = None
        self._ml_available = False
        super().on_dispose()


class LipSyncVerseModule(_CanonicalLipSyncModule):
    """VERSE-Bench face preparation and SyncNet scoring path."""

    name = "lip_sync_verse"
    description = "VERSE-Bench SyncNet lip-sync confidence and distance"
    protocol_class = VerseBenchProtocol
    protocol_token = "verse_bench"
    canonical_fields = ("lse_c_verse", "lse_d_verse")
    metric_info = {
        "lse_c_verse": "VERSE SyncNet lip-sync confidence (higher=better)",
        "lse_d_verse": "VERSE SyncNet lip-sync distance (lower=better)",
    }
    metric_groups = {"lse_c_verse": "temporal", "lse_d_verse": "temporal"}
    sources = {
        "lse_c_verse": "SyncNet LSE-C/LSE-D (Chung & Zisserman 2016), evaluated through the bundled VERSE-Bench inferencer — https://github.com/joonson/syncnet_python/tree/6efbb1c305c23f47a62b09cf4215a8ac45e97d49",
        "lse_d_verse": "SyncNet LSE-C/LSE-D (Chung & Zisserman 2016), evaluated through the bundled VERSE-Bench inferencer — https://github.com/joonson/syncnet_python/tree/6efbb1c305c23f47a62b09cf4215a8ac45e97d49",
    }
    deviations = {
        "lse_c_verse": (
            "VERSE preprocessing resamples to 25 fps and mono 16 kHz, uses the first "
            "InsightFace detection per frame, splits tracks when no face is detected, "
            "discards segments shorter than two seconds, applies an unsmoothed 0.30-scale "
            "224x224 face crop, rejects segments with absolute offset >=14 frames, and "
            "averages confidence across retained segments."
        ),
        "lse_d_verse": (
            "VERSE preprocessing resamples to 25 fps and mono 16 kHz, uses the first "
            "InsightFace detection per frame, splits tracks when no face is detected, "
            "discards segments shorter than two seconds, applies an unsmoothed 0.30-scale "
            "224x224 face crop, rejects segments with absolute offset >=14 frames, and "
            "averages distance vectors across retained segments before taking their minimum."
        ),
    }
    models = [{"id": "syncnet_v2.model", "type": "local", "url": _MODELS_BASE + _SYNCNET_REL, "task": "SyncNet v2 lip-sync model", "size": _megabytes(_SYNCNET_SIZE), "sha256": _SYNCNET_SHA256, "notes": "Checkpoint bytes are identity-pinned; checkpoint license is not established."}]

    def _assign_scores(
        self, quality_metrics: QualityMetrics, lse_c: float, lse_d: float
    ) -> None:
        quality_metrics.lse_c_verse = lse_c
        quality_metrics.lse_d_verse = lse_d


class LipSyncSyncNetModule(_CanonicalLipSyncModule):
    """MIT SyncNet/S3FD reference preparation and scoring path."""

    name = "lip_sync_syncnet"
    description = "MIT SyncNet reference protocol with S3FD face tracks"
    protocol_class = Wav2LipProtocol
    protocol_token = "wav2lip"
    canonical_fields = ("lse_c_syncnet", "lse_d_syncnet")
    default_config = {**_CanonicalLipSyncModule.default_config, "min_face_size": 100, "facedet_scale": 0.25, "min_track": 100, "crop_scale": 0.40, "num_failed_det": 25, "batch_size": 20, "vshift": 15}
    metric_info = {
        "lse_c_syncnet": "SyncNet lip-sync confidence (higher=better)",
        "lse_d_syncnet": "SyncNet lip-sync distance (lower=better)",
    }
    metric_groups = {"lse_c_syncnet": "temporal", "lse_d_syncnet": "temporal"}
    sources = {
        "lse_c_syncnet": "SyncNet and S3FD source (MIT), commit 6efbb1c305c23f47a62b09cf4215a8ac45e97d49 — https://github.com/joonson/syncnet_python/tree/6efbb1c305c23f47a62b09cf4215a8ac45e97d49",
        "lse_d_syncnet": "SyncNet and S3FD source (MIT), commit 6efbb1c305c23f47a62b09cf4215a8ac45e97d49 — https://github.com/joonson/syncnet_python/tree/6efbb1c305c23f47a62b09cf4215a8ac45e97d49",
    }
    deviations = {
        "lse_c_syncnet": (
            "Adapted for checked I/O and configured device execution. By default, input is "
            "prepared at 25 fps with mono 16 kHz audio, S3FD scene-aware face tracking, a "
            "0.40-scale crop with 13-tap median box smoothing, and vshift=15; confidence is "
            "the unweighted mean across eligible tracks. Tracking, crop, batch, and shift "
            "configuration changes alter preparation or aggregation."
        ),
        "lse_d_syncnet": (
            "Adapted for checked I/O and configured device execution. By default, input is "
            "prepared at 25 fps with mono 16 kHz audio, S3FD scene-aware face tracking, a "
            "0.40-scale crop with 13-tap median box smoothing, and vshift=15; distance is "
            "the unweighted mean across eligible tracks. Tracking, crop, batch, and shift "
            "configuration changes alter preparation or aggregation."
        ),
    }
    models = [
        {"id": "syncnet_v2.model", "type": "local", "url": _MODELS_BASE + _SYNCNET_REL, "task": "SyncNet v2 lip-sync model", "size": _megabytes(_SYNCNET_SIZE), "sha256": _SYNCNET_SHA256, "notes": "Checkpoint bytes are identity-pinned; checkpoint license is not established."},
        {"id": "sfd_face.pth", "type": "local", "url": _S3FD_URL, "task": "S3FD face detector", "size": _megabytes(_S3FD_SIZE), "sha256": _S3FD_SHA256, "notes": "Author-hosted checkpoint bytes are identity-pinned; checkpoint license is not established."},
    ]

    def _assign_scores(
        self, quality_metrics: QualityMetrics, lse_c: float, lse_d: float
    ) -> None:
        quality_metrics.lse_c_syncnet = lse_c
        quality_metrics.lse_d_syncnet = lse_d


class LipSyncModule(PipelineModule):
    """Unregistered compatibility facade selecting one canonical module."""

    name = "unnamed_module"

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Describe the legacy alias through its default canonical backend."""
        return LipSyncVerseModule.get_metadata()

    def __new__(cls, config=None):
        values = dict(config or {})
        token = values.pop("protocol", "verse_bench")
        target = {"verse_bench": LipSyncVerseModule, "wav2lip": LipSyncSyncNetModule}.get(token)
        if target is None:
            raise ValueError("lip_sync protocol must be 'verse_bench' or 'wav2lip'")
        instance = target(values)
        instance._requested_module_name = "lip_sync"
        instance._legacy_output_aliases = {instance.canonical_fields[0]: "lse_c", instance.canonical_fields[1]: "lse_d"}
        instance._legacy_lip_sync_protocol = token
        return instance

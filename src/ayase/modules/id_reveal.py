"""ID-Reveal identity distance (Cozzolino et al., ICCV-W 2021).

Published metric for talking-head / deepfake identity analysis, implemented
through the official GRIP-UNINA code vendored under
``third_party/idreveal`` (``grip_unina`` + ``retinaface`` + ``TDDFA``):

* the video is resampled to 25 fps; faces are detected with RetinaFace
  (ResNet-50) on every frame and linked into tracks by IoU matching;
* a TDDFA (3DDFA_V2 MobileNet-1) regressor produces a 62-dim 3DMM
  shape/expression feature per detected face;
* the ID-Reveal temporal network embeds overlapping 100-frame windows
  (stride 50 for the candidate, stride 1 for reference material);
* per-window distance = minimum squared L2 distance to the reference embeddings;
  per track, distances are smoothed with a uniform filter of length 7 and
  summarised by the 5th percentile; the video score is the minimum over
  tracks (upstream ``main_test.py`` aggregation).

``sample.path`` must be a video. ``sample.reference_path`` must be a
directory of reference videos (or a single video file). Reference videos
are embedded once and cached under ``models/id_reveal/references/`` in the
upstream ``embs_track*.npz`` layout. Ayase excludes the candidate only when it
is the exact same file; distinct files with the same stem remain valid
references. Without a backend, a valid
reference, or any face track long enough, the metric is left unset.

id_reveal_distance — lower = better (identity closer to the reference).

Source: Cozzolino et al., "ID-Reveal: Identity-aware DeepFake Video
Detection", arXiv:2012.02512; official code github.com/grip-unina/id-reveal
and github.com/grip-unina/poi-forensics. The vendored code and weights are
released for non-commercial research use (see
``third_party/idreveal/LICENSE.txt``).
"""

import hashlib
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule
from ._tddfa_coeffs import ensure_tddfa_weights

logger = logging.getLogger(__name__)

_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class IDRevealModule(PipelineModule):
    name = "id_reveal"
    provenance = {
        "id_reveal_distance": "adapted",
        "id_reveal_tracks": "utility",
    }
    sources = {
        "id_reveal_distance": (
            "ID-Reveal (Cozzolino et al. 2021, arXiv:2012.02512); official code "
            "https://github.com/grip-unina/id-reveal + https://github.com/grip-unina/poi-forensics (vendored, "
            "non-commercial research license)"
        ),
    }
    deviations = {
        "id_reveal_distance": "Official inference is retained, but Ayase accepts a file or arbitrary reference directory, uses protocol/weight/media-aware persistent embedding caches, and excludes only the exact candidate path rather than every reference sharing its stem",
    }
    description = (
        "ID-Reveal identity distance to reference videos (lower=better; "
        "RetinaFace + TDDFA + temporal ID-Reveal net)"
    )
    default_config = {
        # upstream defaults (poi-forensics config.py / idreveal.yaml)
        "fps": 25,
        "read_stride": 96,
        "rec_stride": 32,
        "clip_length": 100,
        "clip_stride": 50,
        "clip_ref_stride": 1,
        "final_mean": 7,
        "percentile": 5,
        "det_size_threshold": 75,
        "det_target_size": 1280,
        "det_score_threshold": 0.7,
        "track_iou_threshold": 0.4,
        "models_dir": "models",
    }
    models = [
        {
            "id": "akhaliq/RetinaFace-R50",
            "type": "huggingface",
            "url": "https://huggingface.co/akhaliq/RetinaFace-R50/resolve/main/RetinaFace-R50.pth",
            "task": "RetinaFace ResNet-50 face detector",
            "size": "104 MB",
            "auto_download": True,
        },
        {
            "id": "Stable-Human/3ddfa_v2",
            "type": "huggingface",
            "url": "https://huggingface.co/Stable-Human/3ddfa_v2/resolve/main/mb1_120x120.pth",
            "task": "TDDFA (3DDFA_V2) MobileNet-1 3DMM regressor",
            "size": "13 MB",
            "auto_download": True,
        },
        {
            "id": "grip-unina/id-reveal model.th",
            "type": "other",
            "url": "https://raw.githubusercontent.com/grip-unina/id-reveal/main/model.th",
            "task": "ID-Reveal temporal embedding network",
            "size": "29 MB",
            "auto_download": True,
            "notes": "GRIP-UNINA license: non-commercial research use only",
        },
    ]
    metric_info = {
        "id_reveal_distance": (
            "ID-Reveal squared-L2 distance to the closest reference identity window "
            "(lower=better; Cozzolino et al. 2021)"
        ),
        "id_reveal_tracks": "Face tracks contributing to the ID-Reveal score",
    }
    metric_groups = {
        "id_reveal_distance": "face",
        "id_reveal_tracks": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = None
        self._det = None
        self._trk = None
        self._mm = None
        self._ref_temporal = None
        self._opt = None
        self._resources = None
        self._ref_cache_root = None
        self._idreveal = None
        self._idreveal_refdir = None
        self._weight_identity_cache = {}

    # ------------------------------------------------------------------ setup



    def setup(self) -> None:
        if self.test_mode:
            return
        try:
            import torch
            import yaml  # noqa: F401
            from tqdm import tqdm  # noqa: F401

            from ayase.third_party.idreveal.grip_unina.id_reveal.detector import (  # noqa: F401
                IdReveal,
            )
            from ayase.third_party.idreveal.grip_unina.id_reveal.util_3dmm import (
                Compute3DMMtracked,
            )
            from ayase.third_party.idreveal.grip_unina.id_reveal.util_idreavel import (
                ComputeIdReveal,
            )
            from ayase.third_party.idreveal.grip_unina.util_dist import ComputeTemporal
            from ayase.third_party.idreveal.grip_unina.util_face import (
                ComputeTrack,
                DetectFace,
            )
            from ayase.third_party.idreveal.grip_unina.util_read import (  # noqa: F401
                ReadingResampledVideo,
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("id_reveal: vendored backend unavailable (%s); metric left unset", e)
            return

        resources = ensure_tddfa_weights(self.config.get("models_dir", "models"))
        if resources is None:
            self._backend = "unavailable"
            return
        self._resources = resources

        device = str(self.config.get("device", "auto"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # upstream opt (poi-forensics config.py DEFAULT_OPT + config/idreveal.yaml)
        self._opt = {
            "resources_path": str(resources),
            "fps": int(self.config["fps"]),
            "read_stride": int(self.config["read_stride"]),
            "rec_stride": int(self.config["rec_stride"]),
            "det_stride": max(int(self.config["rec_stride"]) // 2, 1),
            "face_det": {
                "size_threshold": int(self.config["det_size_threshold"]),
                "score_threshold": float(self.config["det_score_threshold"]),
                "iou_threshold": float(self.config["track_iou_threshold"]),
            },
            "model": {
                "type": "id_reveal",
                "clip_length": int(self.config["clip_length"]),
                "clip_stride": int(self.config["clip_stride"]),
                "clip_ref_stride": int(self.config["clip_ref_stride"]),
            },
            "percentile": int(self.config["percentile"]),
            "final_mean": int(self.config["final_mean"]),
            "dist_normalization": False,
        }
        self._device = device
        try:
            self._det = DetectFace(
                device,
                str(resources / "Resnet50_Final.pth"),
                size_threshold=int(self.config["det_size_threshold"]),
                target_size=int(self.config["det_target_size"]),
                batch_size=int(self.config["rec_stride"]),
                score_threshold=float(self.config["det_score_threshold"]),
                return_frame=True,
            )
            self._trk = ComputeTrack(float(self.config["track_iou_threshold"]))
            self._mm = Compute3DMMtracked(device, str(resources), return_frame=False)
            clip_op = ComputeIdReveal(
                int(self.config["clip_length"]), device,
                str(resources / "model_idreveal.th"),
            )
            self._ref_temporal = ComputeTemporal(
                int(self.config["clip_length"]),
                int(self.config["clip_ref_stride"]),
                {"3dmm": clip_op},
            )
        except Exception as e:
            self._backend = "unavailable"
            logger.warning("id_reveal: model setup failed (%s); metric left unset", e)
            return
        self._ref_cache_root = resources / "references"
        self._backend = "idreveal"

    # ------------------------------------------------- reference preparation

    def _extract_windows(self, video: Path, temporal) -> Optional[dict]:
        """Run the upstream extraction chain on a video; return dict_out.

        Chain (verbatim upstream order): ReadingResampledVideo -> DetectFace ->
        ComputeTrack -> Compute3DMMtracked -> ComputeTemporal. With
        ``clip_ref_stride`` this matches upstream reference generation; the
        candidate path uses ``IdReveal.compute_distance_video`` instead.
        """
        from ayase.third_party.idreveal.grip_unina.util_read import (
            ReadingResampledVideo,
        )

        opt = self._opt
        with ReadingResampledVideo(str(video), opt["fps"], opt["read_stride"]) as rd:
            ops = [
                rd,
                self._det.reset(),
                self._trk.reset(),
                self._mm.reset(),
                temporal.reset(),
            ]
            dict_out: Dict[str, list] = {"embs_track": []}
            count = 0
            while True:
                try:
                    out = count
                    for op in ops:
                        out = op(out)
                except StopIteration:
                    break
                count += 1
                for key, val in out.items():
                    if len(val) == 0:
                        continue
                    dict_out.setdefault(key, []).extend(list(val))
        return dict_out if dict_out.get("embs_track") else None

    def _cache_protocol_identity(self) -> str:
        """Stable identity for every setting or weight affecting embeddings."""
        protocol_keys = (
            "fps", "read_stride", "rec_stride", "clip_length",
            "clip_ref_stride", "det_size_threshold", "det_target_size",
            "det_score_threshold", "track_iou_threshold",
        )
        parts = ["cache_format=2"]
        parts.extend(
            f"{key}={self.config.get(key, self.default_config.get(key))!r}"
            for key in protocol_keys
        )
        if self._resources is not None:
            for name in ("Resnet50_Final.pth", "mb1_120x120.pth", "model_idreveal.th"):
                path = Path(self._resources) / name
                try:
                    stat = path.stat()
                    stat_key = (stat.st_size, stat.st_mtime_ns)
                    cached = self._weight_identity_cache.get(path)
                    if cached is None or cached[:2] != stat_key:
                        digest = hashlib.sha256()
                        with path.open("rb") as handle:
                            for chunk in iter(lambda: handle.read(1 << 20), b""):
                                digest.update(chunk)
                        cached = (*stat_key, digest.hexdigest())
                        self._weight_identity_cache[path] = cached
                    parts.append(f"{name}:{cached[2]}")
                except OSError:
                    parts.append(f"{name}:missing")
        return "\n".join(parts)

    @staticmethod
    def _media_identity(path: Path) -> str:
        stat = path.stat()
        return f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"

    @staticmethod
    def _reference_label(path: Path) -> str:
        suffix = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:10]
        return f"{path.stem}-{suffix}"

    def _reference_dir(self, reference: Path, exclude_path: Optional[Path] = None) -> Optional[Path]:
        """Ensure upstream-format reference embeddings exist for ``reference``.

        ``reference`` may be a video file or a directory of videos. Each video's
        track embeddings are cached under ``models/id_reveal/references/<hash>/``
        as ``<stem>-<path-hash>/embs_track{t}.npz`` — the layout upstream
        ``ComputeDistance`` reads while keeping distinct same-stem files apart.
        Only a reference resolving to ``exclude_path`` is skipped.
        """
        if reference.is_dir():
            videos = sorted(
                p for p in reference.iterdir() if p.suffix.lower() in _VIDEO_SUFFIXES
            )
        elif reference.suffix.lower() in _VIDEO_SUFFIXES:
            videos = [reference]
        else:
            logger.warning("id_reveal: reference must be video(s), got %s", reference)
            return None
        if not videos:
            return None
        if exclude_path is not None:
            try:
                excluded = exclude_path.resolve()
                videos = [v for v in videos if v.resolve() != excluded]
            except OSError:
                return None
        if not videos:
            return None

        key = hashlib.sha1(
            (self._cache_protocol_identity() + "\n" + "\n".join(
                self._media_identity(p) for p in videos
            )).encode()
        ).hexdigest()[:16]
        ref_dir = self._ref_cache_root / key
        done_marker = ref_dir / ".complete"
        if done_marker.exists():
            return ref_dir if (ref_dir / ".has_embeddings").exists() else None

        ref_dir.mkdir(parents=True, exist_ok=True)
        staging_root = ref_dir / ".staging"
        staging_root.mkdir(exist_ok=True)
        any_embs = False
        for video in videos:
            sub = ref_dir / self._reference_label(video)
            if sub.is_dir():
                if (sub / ".complete").exists():
                    if list(sub.glob("embs_track*.npz")):
                        any_embs = True
                    continue
                # Never trust a directory without its atomic completion marker.
                # Preserve it for inspection outside ComputeDistance's scan.
                quarantine = staging_root / f"quarantine-{sub.name}-{uuid.uuid4().hex}"
                try:
                    sub.replace(quarantine)
                except OSError as exc:
                    logger.warning("id_reveal: cannot quarantine partial cache %s: %s", sub, exc)
                    return None
            stage = Path(tempfile.mkdtemp(prefix=f"{sub.name}-", dir=staging_root))
            try:
                out = self._extract_windows(video, self._ref_temporal)
            except Exception as e:
                logger.warning("id_reveal: reference extraction failed on %s: %s",
                               video.name, e)
                return None
            if out is None:
                (stage / ".empty").touch()
                (stage / ".complete").touch()
                stage.replace(sub)
                continue
            embs_track = np.asarray(out["embs_track"])
            for t in np.unique(embs_track):
                np.savez(
                    stage / f"embs_track{t}.npz",
                    **{
                        k: np.asarray(v)[embs_track == t]
                        for k, v in out.items()
                        if k != "embs_track"
                    },
                )
            (stage / ".complete").touch()
            stage.replace(sub)
            any_embs = True

        if any_embs:
            (ref_dir / ".has_embeddings").touch()
        done_marker.touch()
        return ref_dir if any_embs else None

    def _get_scorer(self, ref_dir: Path):
        """IdReveal instance bound to the prepared reference folder."""
        if self._idreveal is not None and self._idreveal_refdir == ref_dir:
            return self._idreveal
        from ayase.third_party.idreveal.grip_unina.id_reveal.detector import IdReveal

        self._idreveal = IdReveal(logger, {"reference": str(ref_dir)}, self._opt,
                                  self._device)
        self._idreveal_refdir = ref_dir
        return self._idreveal

    # -------------------------------------------------------------- process

    def process(self, sample: Sample) -> Sample:
        if self._backend in (None, "unavailable"):
            return sample
        path = Path(sample.path)
        if path.suffix.lower() not in _VIDEO_SUFFIXES or not path.exists():
            return sample
        ref = getattr(sample, "reference_path", None)
        if ref is None:
            return sample
        ref_path = Path(ref)
        try:
            if ref_path.is_file() and ref_path.resolve() == path.resolve():
                return sample  # never compare a sample against itself
        except OSError:
            return sample
        try:
            ref_dir = self._reference_dir(ref_path, exclude_path=path)
            if ref_dir is None:
                return sample

            scorer = self._get_scorer(ref_dir)
            dict_out, _info = scorer.compute_distance_video(
                str(path), list_poi=["reference"], verbose=False
            )

            dists = np.asarray(dict_out.get("embs_dists_reference", []))
            tracks = np.asarray(dict_out.get("embs_track", []))
            rangs = np.asarray(dict_out.get("embs_range", []))
            if len(dists) == 0 or len(tracks) == 0:
                return sample

            # upstream aggregation (main_test.py): per-track merge -> nanmin
            scores: List[float] = []
            for t in np.unique(tracks):
                d, _loc = scorer.merge_track(
                    dists[tracks == t], rangs[tracks == t]
                )
                scores.extend(d)
            if not scores:
                return sample
            score = float(np.nanmin(scores))
        except Exception as e:
            logger.warning("id_reveal failed on %s: %s", path.name, e)
            return sample

        if not np.isfinite(score):
            return sample
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.id_reveal_distance = score
        sample.quality_metrics.id_reveal_tracks = float(len(scores))
        return sample

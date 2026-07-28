"""STREAM — Spatio-TempoRal Evaluation and Analysis Metric (ICLR 2024).

GitHub: https://github.com/pro2nit/STREAM
PyPI:   https://pypi.org/project/v-stream/  (``pip install v-stream``, module ``stream``)
Paper:  arXiv:2403.09669

STREAM is a *reference-based, dataset-level* generative-video metric. It embeds
every frame with a self-supervised backbone (SwAV resnet50 or DINOv2 vits14,
both fetched at runtime via ``torch.hub`` — no weight mirror needed), then:

* STREAM-T (temporal naturalness): fits a power law to each embedding channel's
  temporal power spectrum, derives a per-video skewness vector, and compares the
  *distribution* of skewness between the real and generated sets via histogram
  correlation. Higher = generated temporal dynamics match real ones.
* STREAM-S (spatial fidelity/diversity): compares the distribution of per-video
  mean feature signals between real and generated sets with prdc
  (precision/recall). Returns fidelity (``stream_F``) and diversity
  (``stream_D``).

Because it is reference-based, both a set of real videos and a set of generated
videos are required. Real videos are supplied per-sample via
``sample.reference_path`` (the BatchMetricModule reference plumbing). With no
reference set the metric is not defined and nothing is emitted (no proxy).

Dataset-level fields (populated only with the real ``v-stream`` backend):
    stream_temporal  — STREAM-T temporal naturalness (histogram correlation)
    stream_spatial   — STREAM-S spatial quality: the harmonic mean (F1) of the
                       backend's real stream_F (fidelity) and stream_D
                       (diversity) outputs, since a single field must summarise
                       both. Derived purely from real backend outputs.
"""
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

from ayase.models import Sample, QualityMetrics  # noqa: F401 (kept for API parity)
from ayase.base_modules import BatchMetricModule

logger = logging.getLogger(__name__)


class STREAMModule(BatchMetricModule):
    name = "stream_metric"
    description = "STREAM spatial/temporal generation eval (ICLR 2024)"
    default_config = {"num_frame": 16, "model": "swav"}
    metric_info = {
        "stream_spatial": "STREAM-S spatial fidelity/diversity (dataset-level, real backend only)",
        "stream_temporal": "STREAM-T temporal naturalness (dataset-level, real backend only)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        # num_frame must be even (the power-law fit uses num_frame // 2 spectral
        # bins) and each video is temporally resampled to exactly this length.
        num_frame = int(self.config.get("num_frame", 16))
        if num_frame % 2 == 1:
            num_frame -= 1
        self.num_frame = max(4, num_frame)
        self.model = self.config.get("model", "swav")
        self._backend = None
        self._stream_instance = None
        self._device = "cpu"
        self._warned_no_ref = False

    def setup(self) -> None:
        if getattr(self, "test_mode", False):
            return
        self._warned_no_ref = False
        try:
            import torch  # noqa: F401
            from stream import STREAM  # upstream v-stream backend
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))
            # Constructing STREAM downloads the SwAV/DINOv2 backbone via torch.hub.
            self._stream_instance = STREAM(num_frame=self.num_frame, model=self.model)
            self._align_embedder_dim()
            self._backend = "v-stream"
            logger.info(
                "STREAM initialised (v-stream backend, model=%s, num_frame=%d, device=%s)",
                self.model,
                self.num_frame,
                self._device,
            )
            return
        except ImportError as e:
            logger.info(
                "STREAM unavailable: v-stream/torch not installed (%s); "
                "stream_spatial/stream_temporal will not be populated.",
                e,
            )
        except Exception as e:  # torch.hub backbone fetch failed (offline, etc.)
            logger.info(
                "STREAM unavailable: backbone load failed (%s); "
                "stream_spatial/stream_temporal will not be populated.",
                e,
            )

        self._backend = "unavailable"
        self._stream_instance = None

    def _align_embedder_dim(self) -> None:
        """Make the backbone emit exactly ``num_embed`` features.

        STREAM's power-law fit is sized to ``num_embed`` (2048 for SwAV). The
        current ``facebookresearch/swav`` hub resnet50 ships a
        ``Linear(2048, 1000)`` ImageNet classifier head, so its forward returns
        1000-d logits instead of the 2048-d backbone features STREAM expects.
        Replace that head with Identity so the embedding dimension matches.
        (DINOv2 vits14 already returns its 384-d CLS token — no head to strip.)
        """
        emb = getattr(self._stream_instance, "embedder", None)
        expected = getattr(self._stream_instance, "num_embed", None)
        if emb is None or not expected:
            return
        fc = getattr(emb, "fc", None)
        if fc is not None and getattr(fc, "out_features", expected) != expected:
            import torch.nn as nn

            emb.fc = nn.Identity()

    # ------------------------------------------------------------------ #
    # Per-video feature extraction                                        #
    # ------------------------------------------------------------------ #
    def extract_features(self, sample: Sample) -> Optional[dict]:
        """Compute the per-video (skewness, mean_signal) via the real backend."""
        if self._stream_instance is None:
            return None
        if not getattr(sample, "is_video", False):
            return None
        try:
            arr = self._load_video_array(Path(sample.path))
            if arr is None:
                return None
            skew, mean = self._skewness_for_array(arr)
            if skew is None or mean is None:
                return None
            return {"skewness": skew, "mean_signal": mean}
        except Exception as e:
            logger.warning("STREAM feature extraction failed for %s: %s", sample.path, e)
            return None

    def _load_video_array(self, path: Path) -> Optional[np.ndarray]:
        """Load exactly ``num_frame`` uniformly-spaced RGB frames as (f, h, w, 3) uint8.

        VidDataset loads every frame in the .npy and the power-law fit assumes
        exactly ``num_frame`` frames, so the clip is resampled (with repetition
        for short clips) to that fixed length via uniform temporal indexing.
        """
        from ayase.image import sample_frames

        frames = sample_frames(path, max_frames=self.num_frame, color="rgb")
        if not frames:
            return None
        n = len(frames)
        idx = np.linspace(0, n - 1, self.num_frame).round().astype(int)
        arr = np.stack([np.ascontiguousarray(frames[i]) for i in idx]).astype(np.uint8)
        if arr.ndim != 4 or arr.shape[-1] != 3:
            return None
        return arr

    def _skewness_for_array(self, arr: np.ndarray):
        """Run STREAM's backbone + power-law fit on one clip written as a .npy."""
        import contextlib
        import io
        import shutil
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="ayase_stream_")
        try:
            np.save(os.path.join(tmpdir, "vid_00000.npy"), arr)
            # calculate_skewness prints progress; keep pipeline logs clean.
            with contextlib.redirect_stdout(io.StringIO()):
                skew, mean = self._stream_instance.calculate_skewness(
                    tmpdir, device=self._device, batch_size=1, num_workers=0
                )
            skew_np = np.asarray(skew[0].detach().cpu().numpy(), dtype=np.float64)
            mean_np = np.asarray(mean[0].detach().cpu().numpy(), dtype=np.float64)
            return skew_np, mean_np
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Dataset-level STREAM-T / STREAM-S                                   #
    # ------------------------------------------------------------------ #
    def compute_distribution_metric(
        self, features: List, reference_features: Optional[List] = None
    ) -> Optional[float]:
        """Compute STREAM-T / STREAM-S for the generated set vs the real set."""
        if self._stream_instance is None or not features:
            return None
        if not reference_features:
            if not self._warned_no_ref:
                logger.info(
                    "STREAM requires a real reference video set "
                    "(sample.reference_path); stream_spatial/stream_temporal not computed."
                )
                self._warned_no_ref = True
            return None

        try:
            fake_skew = np.stack([f["skewness"] for f in features])
            fake_mean = np.stack([f["mean_signal"] for f in features])
            real_skew = np.stack([f["skewness"] for f in reference_features])
            real_mean = np.stack([f["mean_signal"] for f in reference_features])
        except Exception as e:
            logger.warning("STREAM feature stacking failed: %s", e)
            return None

        temporal: Optional[float] = None
        spatial: Optional[float] = None

        # STREAM-T: histogram correlation of per-video skewness distributions.
        try:
            temporal = float(self._stream_instance.stream_T(real_skew, fake_skew))
        except Exception as e:
            logger.warning("STREAM-T computation failed: %s", e)

        # STREAM-S: prdc precision (fidelity) / recall (diversity). Needs enough
        # samples per set for the k=5 nearest-neighbour manifolds.
        try:
            s = self._stream_instance.stream_S(real_mean, fake_mean)
            f_val = float(s["stream_F"])
            d_val = float(s["stream_D"])
            spatial = (2.0 * f_val * d_val / (f_val + d_val)) if (f_val + d_val) > 0 else 0.0
        except Exception as e:
            logger.warning("STREAM-S computation failed: %s", e)

        self._store_metric("stream_temporal", temporal)
        self._store_metric("stream_spatial", spatial)
        return spatial

    def _store_metric(self, name: str, value: Optional[float]) -> None:
        if value is None:
            return
        pipeline = getattr(self, "pipeline", None)
        if pipeline is not None and hasattr(pipeline, "add_dataset_metric"):
            pipeline.add_dataset_metric(name, value)

    def on_dispose(self) -> None:
        """Finalise STREAM at end-of-run, then release resources.

        Overrides BatchMetricModule.on_dispose so the two real dataset fields are
        stored directly (avoiding a bogus ``stream_metric`` dataset key) and so a
        two-sample minimum is not silently imposed on a reference-based metric.
        """
        try:
            if self._feature_cache and self._stream_instance is not None:
                self.compute_distribution_metric(
                    self._feature_cache,
                    self._reference_cache if self._reference_cache else None,
                )
        except Exception as e:
            logger.warning("STREAM finalization failed: %s", e)
        finally:
            self._feature_cache = []
            self._reference_cache = []

        from ayase.pipeline import PipelineModule

        PipelineModule.on_dispose(self)

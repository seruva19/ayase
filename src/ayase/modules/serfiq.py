"""SER-FIQ -- Stochastic Embedding Robustness for Face Image Quality (CVPR 2020).

Terhoerst et al. "SER-FIQ: Unsupervised Estimation of Face Image Quality Based
on Stochastic Embedding Robustness" -- quality is the robustness of a face
embedding under stochastic (dropout-enabled) forward passes of the *face
recognition* network.

Faithful backend (github.com/pterhoer/FaceImageQuality, ``face_image_quality.py``):

    1. Detect + align the face (InsightFace ``buffalo_l`` detector + ArcFace
       5-point alignment to a 112x112 RGB crop).
    2. Feed the SAME aligned crop through a *dropout-enabled* ArcFace network
       ``T`` times.  The published model is a legacy **MXNet** symbol whose
       Dropout node is baked with ``mode="always"``, so dropout stays active at
       inference and each of the ``T`` passes yields a different embedding.
    3. L2-normalise the ``T`` stochastic embeddings, take the pairwise Euclidean
       distances of the upper triangle, and score::

           s     = 2 * (1 / (1 + exp(mean(pairwise_euclidean))))
           score = 1 / (1 + exp(-(alpha * (s - r))))     # alpha=130, r=0.88

       Low embedding spread (stable identity under dropout) -> high quality.

The stochastic-dropout-stability IS the metric, so there is no faithful
substitute: a deterministic (non-dropout) ArcFace or an ONNX approximation would
not reproduce SER-FIQ.  The published dropout weights are MXNet-only.  When
MXNet (and/or InsightFace) is unavailable the module is code-complete but
cleanly reports ``_backend="unavailable"`` and leaves ``serfiq_score`` unset.

serfiq_score -- higher = better quality (0-1)

REVIVAL NOTES (provisional -- EXTERNAL): already code-complete (faithful T=100 stochastic-pass
computation). Blocker: requires MXNet, which has no installable wheel on Windows/Python-3.10. To
revive: on Linux, ``pip install mxnet insightface``; the module then auto-downloads
AkaneTendo25/ayase-models::serfiq/serfiq_model.zip and works. No code change needed.
Source: https://github.com/pterhoer/FaceImageQuality
"""

import logging
import os
import zipfile
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# HuggingFace mirror of the official SER-FIQ dropout-ArcFace weights (MXNet).
_WEIGHTS_REPO = "AkaneTendo25/ayase-models"
_WEIGHTS_FILE = "serfiq/serfiq_model.zip"
_SYMBOL_NAME = "insightface-symbol.json"
_PARAMS_NAME = "insightface-0000.params"


class SERFIQModule(PipelineModule):
    name = "serfiq"
    provisional = True  # no turnkey real backend in a standard install
    description = "SER-FIQ face quality via dropout embedding robustness (CVPR 2020)"
    default_config = {
        "subsample": 4,
        "face_model": "buffalo_l",
        "det_size": 640,
        # Faithful SER-FIQ defaults (face_image_quality.py::get_score).
        "n_forward_passes": 100,  # T stochastic passes
        "alpha": 130.0,           # score stretching factor
        "r": 0.88,                # score displacement
    }
    metric_groups = {
        "serfiq_score": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 4)
        self.face_model = self.config.get("face_model", "buffalo_l")
        self.det_size = self.config.get("det_size", 640)
        self.n_forward_passes = int(self.config.get("n_forward_passes", 100))
        self.alpha = float(self.config.get("alpha", 130.0))
        self.r = float(self.config.get("r", 0.88))
        self._mx = None
        self._ctx = None
        self._insightface_net = None  # MXNet dropout-ArcFace SymbolBlock
        self._face_app = None         # InsightFace detector/aligner
        self._norm_crop = None
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        self._backend = "unavailable"
        self._ml_available = False

        # (1) The published SER-FIQ dropout weights are an MXNet symbol; without
        #     MXNet there is no faithful stochastic-dropout backbone.  Do not
        #     substitute a non-dropout ArcFace -- that would not be SER-FIQ.
        try:
            import mxnet as mx  # type: ignore
        except Exception:
            logger.warning(
                "SER-FIQ unavailable: the official dropout-ArcFace model is MXNet-only "
                "and 'mxnet' is not importable. Install a compatible mxnet build to enable it."
            )
            return

        # (2) InsightFace provides the detector + ArcFace 5-point alignment that
        #     SER-FIQ feeds the network.
        try:
            from insightface.app import FaceAnalysis
            from insightface.utils import face_align
        except Exception:
            logger.warning(
                "SER-FIQ unavailable: face detection/alignment requires InsightFace "
                "(pip install insightface onnxruntime)."
            )
            return

        try:
            symbol_path, params_path = self._materialise_weights()
        except Exception as e:
            logger.warning("SER-FIQ unavailable: could not fetch/unpack weights: %s", e)
            return

        try:
            self._ctx = self._resolve_ctx(mx)
            # SER-FIQ loads the symbol with a single 'data' input; the Dropout
            # node in this symbol is mode="always", so it stays active at
            # inference and drives the T-pass stochasticity.
            self._insightface_net = mx.gluon.nn.SymbolBlock.imports(
                symbol_path, ["data"], params_path, ctx=self._ctx
            )

            self._face_app = FaceAnalysis(
                name=self.face_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))

            self._mx = mx
            self._norm_crop = face_align.norm_crop
            self._ml_available = True
            self._backend = "real"
            logger.info(
                "SER-FIQ initialised (MXNet dropout-ArcFace, T=%d, detector=%s)",
                self.n_forward_passes,
                self.face_model,
            )
        except Exception as e:
            self._ml_available = False
            self._backend = "unavailable"
            logger.warning("SER-FIQ setup failed: %s", e)

    def _resolve_ctx(self, mx):
        """Pick an MXNet context (GPU if the build supports it, else CPU)."""
        try:
            if mx.context.num_gpus() > 0:
                return mx.gpu(0)
        except Exception:
            pass
        return mx.cpu()

    def _materialise_weights(self):
        """Download the SER-FIQ weight zip and extract symbol + params.

        Returns ``(symbol_path, params_path)`` on the local filesystem.
        """
        from huggingface_hub import hf_hub_download

        zip_path = hf_hub_download(repo_id=_WEIGHTS_REPO, filename=_WEIGHTS_FILE)
        extract_dir = os.path.join(os.path.dirname(zip_path), "serfiq_extracted")
        os.makedirs(extract_dir, exist_ok=True)

        symbol_path = os.path.join(extract_dir, _SYMBOL_NAME)
        params_path = os.path.join(extract_dir, _PARAMS_NAME)
        if not (os.path.exists(symbol_path) and os.path.exists(params_path)):
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Flatten any internal directory structure onto the members we need.
                for member in zf.namelist():
                    base = os.path.basename(member)
                    if base in (_SYMBOL_NAME, _PARAMS_NAME):
                        with zf.open(member) as src:
                            target = os.path.join(extract_dir, base)
                            with open(target, "wb") as dst:
                                dst.write(src.read())

        if not (os.path.exists(symbol_path) and os.path.exists(params_path)):
            raise FileNotFoundError(
                f"{_SYMBOL_NAME}/{_PARAMS_NAME} not found inside {_WEIGHTS_FILE}"
            )
        return symbol_path, params_path

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            scores = []
            for frame in frames:
                score = self._compute_serfiq(frame)
                if score is not None:
                    scores.append(score)

            if scores:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.serfiq_score = float(
                    np.clip(np.mean(scores), 0.0, 1.0)
                )
        except Exception as e:
            logger.warning("SER-FIQ failed for %s: %s", sample.path, e)

        return sample

    def _compute_serfiq(self, frame_bgr: np.ndarray) -> Optional[float]:
        """SER-FIQ for a single BGR frame: detect + align the largest face,
        then score its dropout embedding stability."""
        faces = self._face_app.get(frame_bgr)
        if not faces:
            return None

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        kps = getattr(face, "kps", None)
        if kps is None:
            return None

        # ArcFace 5-point alignment -> 112x112 BGR crop (fresh array).
        aligned_bgr = self._norm_crop(frame_bgr, landmark=kps, image_size=112)
        # SER-FIQ expects an RGB aligned image.
        aligned_rgb = aligned_bgr[:, :, ::-1]
        return self._serfiq_score(aligned_rgb)

    def _serfiq_score(self, aligned_rgb: np.ndarray) -> float:
        """Verbatim SER-FIQ score (face_image_quality.py::get_score).

        ``aligned_rgb`` is an HxWx3 RGB uint8 aligned face crop.
        """
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.preprocessing import normalize

        mx = self._mx
        T = self.n_forward_passes

        # (3, H, W) as the network expects.
        img = np.transpose(aligned_rgb, (2, 0, 1)).astype(np.float32)
        input_blob = np.expand_dims(img, axis=0)
        repeated = np.repeat(input_blob, T, axis=0)
        data = mx.nd.array(repeated, ctx=self._ctx)

        # Dropout is mode="always" in the symbol -> T stochastic embeddings.
        X = self._insightface_net(data).asnumpy()

        norm = normalize(X, axis=1)
        eucl_dist = euclidean_distances(norm, norm)[np.triu_indices(T, k=1)]

        s = 2 * (1 / (1 + np.exp(np.mean(eucl_dist))))
        return float(1 / (1 + np.exp(-(self.alpha * (s - self.r)))))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        """Extract frames (BGR) from a video or image via the shared cache.

        Frames from ``ayase.image`` are read-only; alignment (``norm_crop``)
        returns fresh arrays, so no in-place mutation of the cache occurs.
        """
        from ayase.image import sample_frames

        return list(sample_frames(sample.path, max_frames=self.subsample, color="bgr"))

    def on_dispose(self) -> None:
        self._insightface_net = None
        self._face_app = None
        self._mx = None
        self._ctx = None
        import gc

        gc.collect()

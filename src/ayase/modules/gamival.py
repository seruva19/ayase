"""GAMIVAL -- GAMIng Video Quality EVALuator (no-reference, cloud gaming).

Yu et al., "GAMIVAL: Video Quality Prediction on Mobile Cloud Gaming
Content", IEEE Signal Processing Letters 2023.
GitHub: https://github.com/utlive/GAMIVAL

Real pipeline (two-branch, 2180-d feature vector -> SVR):

  Branch 1 -- NSS features (1156-d), MATLAB only.
    ``calc_GAMIVAL_features.m`` -> ``GAMIVAL_spatial_features.m`` computes
    680-d spatial NSS per frame (MSCN/paired-product NSS in Y plus HSI/LAB/
    LMS colour opponency and wavelet-packet-transform sub-band statistics
    using the shipped WPT filter banks) together with temporal NSS, mean-
    pooled over 1-second chunks at 2 fps on 540p-subsampled frames.

  Branch 2 -- CNN features (1024-d): NDNetGaming.
    ``demo_compute_CNN_feats.py`` loads ``subjectiveDemo2_DMOS_Final.model``
    (a Keras 2.2.4 / TensorFlow HDF5 DenseNet, input 299x299x3) via
    ``keras.models.load_model``, drops the final layer
    (``model.layers[-2].output``) to expose the 1024-d penultimate DenseNet
    embedding, and averages it over 299x299 patches of two frames per
    second after ``keras.applications.densenet.preprocess_input``.

  Combine + regress: ``combineFeature.m`` concatenates the branches into a
    2180-d vector; inf/nan are mean-imputed; a grid-searched RBF ``SVR``
    (C, gamma) is fit on the training MOS and its output is passed through a
    4-parameter logistic mapping (``subjectiveDemo2_DMOS_Final.npy`` holds
    the four logistic coefficients).

Why this module reports ``unavailable`` in this environment (no-heuristic
policy -- real metric or None, never an approximation under the GAMIVAL
name):

  * The 1156-d NSS branch is MATLAB-only; upstream ships no Python port,
    so there is no faithful Python equivalent to import.
  * The 1024-d CNN branch needs TensorFlow/Keras to load the NDNetGaming
    DenseNet HDF5 model -- neither is installed here.
  * Upstream ships no ready-to-apply SVR predictor: the mirrored
    ``.model`` is the NDNetGaming CNN (not an SVR), the ``.npy`` is the
    4-parameter logistic mapping, and the SVR itself is re-fit per split
    from grid-search ``best_pamtr/*.mat`` files that are not distributed.

gamival_score -- higher = better quality (0-1).
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class GAMIVALModule(PipelineModule):
    name = "gamival"
    description = "GAMIVAL cloud gaming NR-VQA: 1156 NSS + 1024 NDNetGaming CNN -> SVR (2023)"
    default_config = {
        "subsample": 8,
    }
    metric_groups = {
        "gamival_score": "nr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self._backend = "unavailable"
        self._ndnetgaming = None  # NDNetGaming DenseNet feature extractor (Keras)
        self._logistic = None     # 4-parameter logistic mapping coefficients
        self._svr = None          # trained SVR predictor (not shipped upstream)

    def setup(self) -> None:
        if self.test_mode:
            return

        # Real GAMIVAL needs three genuine components, none reproducible here:
        #   1. NSS branch  -> MATLAB only (no Python port ships upstream)
        #   2. CNN branch  -> TensorFlow/Keras NDNetGaming DenseNet model
        #   3. SVR         -> no pretrained predictor is distributed
        # We probe the one prerequisite that could conceivably be present
        # (TensorFlow) so the log names the concrete blocker, then decline.
        tf_ok = False
        try:
            import tensorflow  # noqa: F401
            tf_ok = True
        except Exception:
            tf_ok = False

        missing = []
        if not tf_ok:
            missing.append(
                "TensorFlow/Keras to run the NDNetGaming DenseNet CNN "
                "(subjectiveDemo2_DMOS_Final.model, Keras 2.2.4 HDF5)"
            )
        # These two are unconditional in this environment.
        missing.append("a Python port of the MATLAB NSS branch (1156-d)")
        missing.append("a trained GAMIVAL SVR predictor (none shipped upstream)")

        self._backend = "unavailable"
        logger.warning(
            "GAMIVAL unavailable: cannot reproduce the published two-branch "
            "NSS+NDNetGaming->SVR pipeline; missing %s. Metric left unset.",
            "; ".join(missing),
        )

    def process(self, sample: Sample) -> Sample:
        if self._backend != "real":
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample
            score = self._compute_gamival(frames)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.gamival_score = score
        except Exception as e:
            logger.warning("GAMIVAL failed for %s: %s", sample.path, e)

        return sample

    def _compute_gamival(self, frames: List[np.ndarray]) -> Optional[float]:
        """Real GAMIVAL prediction: NSS(1156) + NDNetGaming CNN(1024) -> SVR.

        Only reachable when ``self._backend == "real"``. That state is never
        entered in this environment (see ``setup``); the method is retained
        so the module is code-complete for a future environment that supplies
        a Python NSS port, TensorFlow/Keras, and a trained SVR.
        """
        if self._svr is None or self._ndnetgaming is None:
            return None
        # Intentionally unimplemented body: entering it requires the three
        # missing components above. We refuse to fabricate an approximate
        # score under the GAMIVAL name (no-heuristic policy).
        return None

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        import cv2

        frames: List[np.ndarray] = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

    def on_dispose(self) -> None:
        self._ndnetgaming = None
        self._svr = None
        import gc
        gc.collect()

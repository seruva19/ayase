"""FD and Variation on 3DMM expression/pose coefficients.

Learning to Listen (Ng et al., CVPR 2022) evaluates generated motion with a
Fréchet distance on the distribution of per-frame motion coefficients against
a reference set; talking-head evals (EMO, AniTalker, FLOAT) report the same on
expression and pose coefficient streams, together with the temporal Variation
(std) of each coefficient set.

Per-sample outputs are the coefficient-stream variations; dataset-level FDs
(Frechet on expression and on pose coefficients) are stored via
``pipeline.add_dataset_metric`` when every candidate carries a reference video.

expr_var_3dmm / pose_var_3dmm -- higher = more varied expression/pose.
fd_3dmm_expression / fd_3dmm_pose (dataset) -- lower = closer to reference.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from ayase.base_modules import BatchMetricModule
from ayase.models import QualityMetrics, Sample
from ._tddfa_coeffs import EXPR_SLICE, POSE_SLICE, ensure_tddfa_weights

logger = logging.getLogger(__name__)


def _frechet_distance(mu1, sigma1, mu2, sigma2) -> Optional[float]:
    """Standard FID-style Frechet distance between two Gaussians."""
    from scipy.linalg import sqrtm

    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            return None
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def _gaussian(seqs: List[np.ndarray], col_slice: slice):
    """Pool frames across sequences and fit mean/covariance on a slice."""
    feats = np.concatenate([s[:, col_slice] for s in seqs], axis=0).astype(np.float64)
    if len(feats) < 2:
        return None
    return feats.mean(axis=0), np.cov(feats, rowvar=False)


class FD3DMMModule(BatchMetricModule):
    name = "fd_3dmm"
    description = "Frechet distance and variation on 3DMM expression/pose coefficients"
    provenance = "adapted"
    sources = {
        "fd_3dmm_expression": "Expr-FD, Learning to Listen (Ng et al., arXiv:2204.08451); EMO/AniTalker evals — https://github.com/evanscratch/learning-to-listen",
        "fd_3dmm_pose": "Pose-FD, Learning to Listen (Ng et al., arXiv:2204.08451); EMO/AniTalker evals — https://github.com/evanscratch/learning-to-listen",
        "expr_var_3dmm": "Expr-Var, AniTalker (https://arxiv.org/abs/2408.01121) / Learning to Listen eval protocol",
        "pose_var_3dmm": "Pose-Var, AniTalker (https://arxiv.org/abs/2408.01121) / Learning to Listen eval protocol",
    }
    deviations = {
        "fd_3dmm_expression": "the source evals extract coefficients with their own face model; here TDDFA/3DDFA_V2 10-dim expression coefficients are used, so absolute values are not comparable",
        "fd_3dmm_pose": "the source evals extract coefficients with their own face model; here TDDFA/3DDFA_V2 12-dim pose coefficients are used, so absolute values are not comparable",
        "expr_var_3dmm": "coefficients come from TDDFA/3DDFA_V2 rather than the source model, so absolute values are not comparable",
        "pose_var_3dmm": "coefficients come from TDDFA/3DDFA_V2 rather than the source model, so absolute values are not comparable",
    }
    requires_reference = True
    default_config = {
        "fps": 25,
        "read_stride": 96,
        "rec_stride": 32,
        "det_size_threshold": 75,
        "det_score_threshold": 0.7,
        "det_target_size": 1280,
        "device": "auto",
    }
    models = [
        {
            "id": "akhaliq/RetinaFace-R50",
            "type": "huggingface",
            "url": "https://huggingface.co/akhaliq/RetinaFace-R50/resolve/main/RetinaFace-R50.pth",
            "task": "RetinaFace ResNet50 face detector (shared)",
        },
        {
            "id": "Stable-Human/3ddfa_v2",
            "type": "huggingface",
            "url": "https://huggingface.co/Stable-Human/3ddfa_v2/resolve/main/mb1_120x120.pth",
            "task": "TDDFA/3DDFA_V2 MobileNet-1 3DMM regressor (shared)",
        },
    ]
    metric_info = {
        "expr_var_3dmm": "Temporal std of expression coefficients (higher=more varied)",
        "pose_var_3dmm": "Temporal std of pose coefficients (higher=more varied)",
        "fd_3dmm_expression": "Frechet distance on expression-coefficient distributions vs reference set (lower=closer)",
        "fd_3dmm_pose": "Frechet distance on pose-coefficient distributions vs reference set (lower=closer)",
    }
    metric_groups = {
        "expr_var_3dmm": "face",
        "pose_var_3dmm": "face",
        "fd_3dmm_expression": "face",
        "fd_3dmm_pose": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._extractor = None
        self._backend = None

    def setup(self) -> None:
        if self.config.get("test_mode"):
            self._backend = "unavailable"
            return
        resources = ensure_tddfa_weights(self.config.get("models_dir", "models"),
                                         ["Resnet50_Final.pth", "mb1_120x120.pth"])
        if resources is None:
            self._backend = "unavailable"
            return
        device = self.config.get("device", "auto")
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        try:
            from ._tddfa_coeffs import TDDFACoeffExtractor

            self._extractor = TDDFACoeffExtractor(
                resources,
                device=device,
                fps=int(self.config.get("fps", 25)),
                read_stride=int(self.config.get("read_stride", 96)),
                rec_stride=int(self.config.get("rec_stride", 32)),
                det_size_threshold=int(self.config.get("det_size_threshold", 75)),
                det_score_threshold=float(self.config.get("det_score_threshold", 0.7)),
                det_target_size=int(self.config.get("det_target_size", 1280)),
            )
            self._backend = "tddfa"
        except Exception as e:
            logger.warning("fd_3dmm: TDDFA init failed: %s", e)
            self._backend = "unavailable"

    def extract_features(self, sample: Sample):
        if self._backend != "tddfa" or self._extractor is None:
            return None
        if not sample.is_video:
            return None
        res = self._extractor.extract(Path(sample.path))
        if res is None or len(res[0]) < 2:
            return None
        return res[0]

    def process(self, sample: Sample) -> Sample:
        feats = self.extract_features(sample)
        if feats is not None:
            self._feature_cache.append(feats)
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.expr_var_3dmm = float(
                np.std(feats[:, EXPR_SLICE], axis=0).mean())
            sample.quality_metrics.pose_var_3dmm = float(
                np.std(feats[:, POSE_SLICE], axis=0).mean())

        reference_path = getattr(sample, "reference_path", None)
        if reference_path is not None:
            try:
                reference_path = Path(reference_path)
                if reference_path.is_file():
                    ref = Sample(path=reference_path, is_video=sample.is_video)
                    ref_feats = self.extract_features(ref)
                    if ref_feats is not None:
                        self._reference_cache.append(ref_feats)
            except Exception:
                pass
        return sample

    def compute_distribution_metric(
        self,
        features: List[np.ndarray],
        reference_features: Optional[List[np.ndarray]] = None,
    ) -> Optional[float]:
        if not reference_features:
            return None
        for name, col in (("fd_3dmm_expression", EXPR_SLICE),
                          ("fd_3dmm_pose", POSE_SLICE)):
            g = _gaussian(features, col)
            r = _gaussian(reference_features, col)
            if g is None or r is None:
                continue
            fd = _frechet_distance(g[0], g[1], r[0], r[1])
            if fd is None or not np.isfinite(fd):
                continue
            if hasattr(self, "pipeline") and self.pipeline and hasattr(
                    self.pipeline, "add_dataset_metric"):
                self.pipeline.add_dataset_metric(name, fd)
        return None

    def on_dispose(self) -> None:
        if len(self._feature_cache) < 1 or not self._reference_cache:
            self._feature_cache = []
            self._reference_cache = []
            return
        try:
            self.compute_distribution_metric(
                self._feature_cache,
                self._reference_cache,
            )
        except Exception as e:
            logger.error("fd_3dmm: dataset FD failed: %s", e)
        finally:
            self._feature_cache = []
            self._reference_cache = []

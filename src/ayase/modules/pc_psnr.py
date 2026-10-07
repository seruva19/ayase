"""Reference-based D1/D2 geometry PSNR for PLY or PCD point clouds.

D1 uses symmetric point-to-point distances (max of both directions, per
``mpeg-pcc-dmetric``); D2 projects those errors onto the target cloud's
normals (point-to-plane) and is reported when normals are available. The peak
is the reference bounding-box diagonal. Scores are in dB; higher means less
geometric error.

Basis: MPEG point-cloud distortion metrics,
https://github.com/MPEGGroup/mpeg-pcc-tmc13/tree/master/mpeg-pcc-dmetric
"""

import logging
import numpy as np
from pathlib import Path
from typing import Optional
from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule

logger = logging.getLogger(__name__)


class PCPSNRModule(ReferenceBasedModule):
    name = "pc_psnr"
    provenance = "published"
    sources = {
        "pc_d1_psnr": "MPEG D1/D2 PSNR (mpeg-pcc-dmetric), symmetric max — https://github.com/MPEGGroup/mpeg-pcc-tmc13/tree/master/mpeg-pcc-dmetric",
        "pc_d2_psnr": "MPEG D1/D2 PSNR (mpeg-pcc-dmetric), symmetric max — https://github.com/MPEGGroup/mpeg-pcc-tmc13/tree/master/mpeg-pcc-dmetric",
    }
    deviations = {
        "pc_d2_psnr": "at least one of the clouds needs normals; without normals the metric is not emitted",
    }
    description = "D1/D2 MPEG point cloud PSNR"
    metric_field = None
    default_config = {}
    metric_groups = {
        "pc_d1_psnr": "fr_quality",
        "pc_d2_psnr": "fr_quality",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._backend = "numpy"

    def process(self, sample):
        ref = getattr(sample, "reference_path", None)
        if ref is None:
            return sample
        if not isinstance(ref, Path):
            ref = Path(ref)
        if not ref.exists():
            return sample
        ext = sample.path.suffix.lower()
        if ext not in (".ply", ".pcd"):
            return sample
        try:
            d1, d2 = self._compute(sample.path, ref)
            if d1 is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.pc_d1_psnr = d1
                sample.quality_metrics.pc_d2_psnr = d2
        except Exception as e:
            logger.warning(f"PC-PSNR failed: {e}")
        return sample

    def compute_reference_score(self, sample_path, reference_path):
        d1, d2 = self._compute(sample_path, reference_path)
        return d1

    def _compute(self, sample_path, ref_path):
        try:
            import open3d as o3d

            pc1 = o3d.io.read_point_cloud(str(sample_path))
            pc2 = o3d.io.read_point_cloud(str(ref_path))
            p1 = np.asarray(pc1.points)
            p2 = np.asarray(pc2.points)
            if len(p1) == 0 or len(p2) == 0:
                return None, None
            # D1: point-to-point
            from scipy.spatial import cKDTree

            # MPEG pc_error convention: both directions, take the max MSE.
            tree1 = cKDTree(p1)
            tree2 = cKDTree(p2)
            d_fwd, _ = tree2.query(p1)
            d_bwd, _ = tree1.query(p2)
            mse_d1 = max(float(np.mean(d_fwd**2)), float(np.mean(d_bwd**2)))
            peak = float(np.linalg.norm(p2.max(axis=0) - p2.min(axis=0)))
            d1 = 10 * np.log10(peak**2 / max(mse_d1, 1e-10))

            # D2 (point-to-plane): project onto the *target* cloud's normals in
            # each direction and take the symmetric max, like the reference.
            mse_terms = []
            if pc2.has_normals():
                _, idx = tree2.query(p1)
                n2 = np.asarray(pc2.normals)[idx]
                mse_terms.append(float(np.mean(np.sum((p1 - p2[idx]) * n2, axis=1) ** 2)))
            if pc1.has_normals():
                _, idx = tree1.query(p2)
                n1 = np.asarray(pc1.normals)[idx]
                mse_terms.append(float(np.mean(np.sum((p2 - p1[idx]) * n1, axis=1) ** 2)))
            d2 = None
            if mse_terms:
                mse_d2 = max(mse_terms)
                d2 = float(10 * np.log10(peak**2 / max(mse_d2, 1e-10)))
            return float(d1), d2
        except ImportError:
            logger.debug("open3d/scipy not installed for PC-PSNR")
            return None, None

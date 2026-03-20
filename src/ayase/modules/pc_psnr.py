"""D1/D2 Point Cloud PSNR (MPEG Standard). pc_d1_psnr, pc_d2_psnr — dB, higher = better"""
import logging, numpy as np
from pathlib import Path
from typing import Optional
from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule
logger = logging.getLogger(__name__)

class PCPSNRModule(ReferenceBasedModule):
    name = "pc_psnr"; description = "D1/D2 MPEG point cloud PSNR"; metric_field = None; default_config = {}
    def process(self, sample):
        ref = getattr(sample, "reference_path", None)
        if ref is None: return sample
        if not isinstance(ref, Path): ref = Path(ref)
        if not ref.exists(): return sample
        ext = sample.path.suffix.lower()
        if ext not in (".ply", ".pcd"): return sample
        try:
            d1, d2 = self._compute(sample.path, ref)
            if d1 is not None:
                if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
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
            p1 = np.asarray(pc1.points); p2 = np.asarray(pc2.points)
            if len(p1) == 0 or len(p2) == 0: return None, None
            # D1: point-to-point
            from scipy.spatial import cKDTree
            tree2 = cKDTree(p2); d_p2p, _ = tree2.query(p1)
            mse_d1 = np.mean(d_p2p**2)
            peak = np.max(np.linalg.norm(p2.max(axis=0)-p2.min(axis=0)))
            d1 = 10*np.log10(peak**2/max(mse_d1, 1e-10))
            # D2: point-to-plane (approximation using normals)
            d2 = None  # Requires normals for point-to-plane
            if pc2.has_normals():
                normals = np.asarray(pc2.normals)
                _, idx = tree2.query(p1)
                proj = np.sum((p1 - p2[idx]) * normals[idx], axis=1)**2
                mse_d2 = np.mean(proj)
                d2 = 10*np.log10(peak**2/max(mse_d2, 1e-10))
            return float(d1), float(d2)
        except ImportError:
            logger.debug("open3d/scipy not installed for PC-PSNR")
            return None, None

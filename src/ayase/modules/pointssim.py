"""PointSSIM — Structural Similarity for Point Clouds (2020). pointssim_score — 0-1, higher = better"""
import logging, numpy as np
from pathlib import Path
from typing import Optional
from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule
logger = logging.getLogger(__name__)
class PointSSIMModule(ReferenceBasedModule):
    name = "pointssim"; description = "PointSSIM structural similarity for point clouds (2020)"; metric_field = "pointssim_score"; default_config = {}
    def compute_reference_score(self, sample_path: Path, reference_path: Path) -> Optional[float]:
        ext = sample_path.suffix.lower()
        if ext not in (".ply", ".pcd"): return None
        try:
            import open3d as o3d
            from scipy.spatial import cKDTree
            pc1 = o3d.io.read_point_cloud(str(sample_path))
            pc2 = o3d.io.read_point_cloud(str(reference_path))
            p1, p2 = np.asarray(pc1.points), np.asarray(pc2.points)
            if len(p1)==0 or len(p2)==0: return None
            tree2 = cKDTree(p2)
            k = min(12, len(p2))
            n_samples = min(2000, len(p1))
            sample_idx = np.random.choice(len(p1), n_samples, replace=False)
            ssim_vals = []
            for i in sample_idx:
                _, nn = tree2.query(p1[i], k=k)
                local_ref = p2[nn]
                mu1, mu2 = p1[i].mean(), local_ref.mean(axis=0).mean()
                s1, s2 = p1[i].var(), local_ref.var()
                cov = np.cov(np.concatenate([p1[i:i+1], local_ref[:1]]).T)[0,1] if k > 1 else 0
                C1, C2 = 0.01, 0.03
                ssim = ((2*mu1*mu2+C1)*(2*cov+C2))/((mu1**2+mu2**2+C1)*(s1+s2+C2))
                ssim_vals.append(max(min(ssim, 1), 0))
            return float(np.mean(ssim_vals))
        except ImportError:
            logger.debug("open3d/scipy not installed"); return None

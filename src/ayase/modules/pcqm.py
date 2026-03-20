"""PCQM — Point Cloud Quality Metric (2020). pcqm_score — higher = better"""
import logging, numpy as np
from pathlib import Path
from typing import Optional
from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule
logger = logging.getLogger(__name__)
class PCQMModule(ReferenceBasedModule):
    name = "pcqm"; description = "PCQM geometry+color point cloud quality (2020)"; metric_field = "pcqm_score"; default_config = {}
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
            tree2 = cKDTree(p2); dists, _ = tree2.query(p1)
            geo_score = 1.0/(1.0+np.mean(dists**2)*100)
            # Color if available
            if pc1.has_colors() and pc2.has_colors():
                c1, c2 = np.asarray(pc1.colors), np.asarray(pc2.colors)
                _, idx = tree2.query(p1)
                color_diff = np.mean((c1-c2[idx])**2)
                color_score = 1.0/(1.0+color_diff*10)
                return float(0.6*geo_score + 0.4*color_score)
            return float(geo_score)
        except ImportError:
            logger.debug("open3d/scipy not installed"); return None

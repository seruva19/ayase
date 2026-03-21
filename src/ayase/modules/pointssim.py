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
            tree1 = cKDTree(p1)
            tree2 = cKDTree(p2)
            # Chamfer distance: mean of nearest-neighbor distances in both directions
            d1, _ = tree2.query(p1)
            d2, _ = tree1.query(p2)
            chamfer_dist = float(np.mean(d1) + np.mean(d2)) / 2.0
            return float(1.0 / (1.0 + chamfer_dist))
        except ImportError:
            logger.debug("open3d/scipy not installed"); return None

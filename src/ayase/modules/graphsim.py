"""GraphSIM — Graph Signal Gradient Quality for Point Clouds (2020). graphsim_score — higher = better"""
import logging, numpy as np
from pathlib import Path
from typing import Optional
from ayase.models import Sample, QualityMetrics
from ayase.base_modules import ReferenceBasedModule
logger = logging.getLogger(__name__)
class GraphSIMModule(ReferenceBasedModule):
    name = "graphsim"; description = "GraphSIM graph gradient point cloud quality (2020)"; metric_field = "graphsim_score"; default_config = {}
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
            # Graph gradient: local neighborhood color gradient moments
            tree1, tree2 = cKDTree(p1), cKDTree(p2)
            k = min(10, len(p1), len(p2))
            _, nn1 = tree1.query(p1, k=k); _, nn2 = tree2.query(p2, k=k)
            if pc1.has_colors() and pc2.has_colors():
                c1, c2 = np.asarray(pc1.colors), np.asarray(pc2.colors)
                grad1 = np.std([c1[nn1[i]].std(axis=0) for i in range(min(1000,len(p1)))], axis=0).mean()
                grad2 = np.std([c2[nn2[i]].std(axis=0) for i in range(min(1000,len(p2)))], axis=0).mean()
                return float(1.0/(1.0+abs(grad1-grad2)*10))
            # Geometry-only gradient
            grad1 = np.std([p1[nn1[i]].std(axis=0) for i in range(min(1000,len(p1)))], axis=0).mean()
            grad2 = np.std([p2[nn2[i]].std(axis=0) for i in range(min(1000,len(p2)))], axis=0).mean()
            return float(1.0/(1.0+abs(grad1-grad2)*10))
        except ImportError:
            logger.debug("open3d/scipy not installed"); return None

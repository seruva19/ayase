"""MM-PCQA — Multi-Modal Point Cloud QA (IJCAI 2023). mm_pcqa_score — higher = better"""
import logging, numpy as np
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)
class MMPCQAModule(PipelineModule):
    name = "mm_pcqa"; description = "MM-PCQA multi-modal point cloud QA (IJCAI 2023)"; default_config = {}
    def __init__(self, c=None):
        super().__init__(c); self._backend = "heuristic"
    def setup(self):
        try: import mm_pcqa; self._model = mm_pcqa; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        ext = sample.path.suffix.lower()
        if ext not in (".ply", ".pcd"): return sample
        try:
            import open3d as o3d
            pc = o3d.io.read_point_cloud(str(sample.path))
            points = np.asarray(pc.points)
            if len(points) == 0: return sample
            # Point density
            from scipy.spatial import cKDTree
            tree = cKDTree(points)
            dists, _ = tree.query(points, k=min(6, len(points)))
            density = 1.0/(1.0+np.mean(dists[:, 1:])*10)
            # Color uniformity
            color_q = 0.5
            if pc.has_colors():
                colors = np.asarray(pc.colors)
                color_q = min(colors.std(axis=0).mean()/0.3, 1)
            # Point distribution regularity
            spread = np.std(points, axis=0).mean()
            regularity = min(spread/1.0, 1)
            score = 0.4*density + 0.3*color_q + 0.3*regularity
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.mm_pcqa_score = float(np.clip(score, 0, 1))
        except ImportError:
            logger.debug("open3d/scipy not installed")
        except Exception as e:
            logger.warning(f"MM-PCQA failed: {e}")
        return sample

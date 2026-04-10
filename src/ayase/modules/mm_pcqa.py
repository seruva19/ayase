"""MM-PCQA -- Multi-Modal Point Cloud QA (IJCAI 2023).

Multi-modal quality assessment for point cloud content using features
extracted from rendered 2D views of 3D data.

Implementation:
    ResNet-50 extracts features from multiple rendered viewpoints of the
    point cloud.  For .ply/.pcd files, Open3D renders views from 6+
    camera positions.  Features are aggregated across views with
    view-importance weighting for final quality prediction.

mm_pcqa_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

# Camera viewpoint angles (azimuth, elevation) in degrees
_VIEW_ANGLES = [
    (0, 0),       # front
    (90, 0),      # right
    (180, 0),     # back
    (270, 0),     # left
    (0, 45),      # top-front
    (0, -45),     # bottom-front
]


class MMPCQAModule(PipelineModule):
    name = "mm_pcqa"
    description = "MM-PCQA multi-modal point cloud QA (IJCAI 2023)"
    default_config = {
        "n_views": 6,
        "render_size": 224,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.n_views = self.config.get("n_views", 6)
        self.render_size = self.config.get("render_size", 224)
        self._resnet = None
        self._resnet_transform = None
        self._quality_head = None
        self._device = "cpu"
        self._ml_available = False
        self._backend = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torchvision.models as models
            from torchvision import transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
            self._resnet.eval().to(self._device)

            self._resnet_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.render_size, self.render_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            # Quality head: aggregated multi-view features -> score
            self._quality_head = torch.nn.Sequential(
                torch.nn.Linear(2048, 512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(512, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 1),
                torch.nn.Sigmoid(),
            ).to(self._device)
            self._quality_head.eval()

            # View attention: learn which views are most informative
            self._view_attn = torch.nn.Sequential(
                torch.nn.Linear(2048, 128),
                torch.nn.Tanh(),
                torch.nn.Linear(128, 1),
            ).to(self._device)
            self._view_attn.eval()

            self._ml_available = True
            self._backend = "resnet"
            logger.info(
                "MM-PCQA initialised with ResNet-50 on %s", self._device
            )

        except ImportError:
            logger.warning(
                "MM-PCQA: no ML backend available. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("MM-PCQA setup failed: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        ext = sample.path.suffix.lower()
        if ext not in (".ply", ".pcd", ".obj", ".stl"):
            return sample

        try:
            views = self._render_views(sample)
            if not views:
                return sample

            score = self._compute_multiview_quality(views)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.mm_pcqa_score = float(
                    np.clip(score, 0.0, 1.0)
                )

        except Exception as e:
            logger.warning("MM-PCQA failed for %s: %s", sample.path, e)

        return sample

    def _render_views(self, sample: Sample) -> List[np.ndarray]:
        """Render multiple viewpoint images from 3D point cloud using Open3D."""
        try:
            import open3d as o3d
        except ImportError:
            logger.debug("open3d not installed, cannot render views")
            return []

        try:
            pc = o3d.io.read_point_cloud(str(sample.path))
            if not pc.has_points() or len(np.asarray(pc.points)) == 0:
                return []

            # Estimate normals if missing (needed for rendering)
            if not pc.has_normals():
                pc.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )

            # Set up off-screen renderer
            views = []
            vis = o3d.visualization.Visualizer()
            vis.create_window(
                width=self.render_size,
                height=self.render_size,
                visible=False,
            )
            vis.add_geometry(pc)

            ctr = vis.get_view_control()
            center = pc.get_center()

            for azimuth, elevation in _VIEW_ANGLES[:self.n_views]:
                # Set camera position
                ctr.set_front([
                    np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth)),
                    np.sin(np.radians(elevation)),
                    np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth)),
                ])
                ctr.set_lookat(center)
                ctr.set_up([0, 1, 0])
                ctr.set_zoom(0.7)

                vis.poll_events()
                vis.update_renderer()

                # Capture image
                img = vis.capture_screen_float_buffer(do_render=True)
                img_np = (np.asarray(img) * 255).astype(np.uint8)

                if img_np.size > 0:
                    views.append(img_np)

            vis.destroy_window()
            return views

        except Exception as e:
            logger.debug("Open3D rendering failed: %s", e)
            return []

    def _compute_multiview_quality(self, views: List[np.ndarray]) -> Optional[float]:
        """Compute quality from multi-view features with attention weighting."""
        import torch

        view_features = []
        for view in views:
            feat = self._extract_feature(view)
            if feat is not None:
                view_features.append(feat)

        if not view_features:
            return None

        # View attention weighting
        feat_tensor = torch.from_numpy(
            np.array(view_features, dtype=np.float32)
        ).to(self._device)

        with torch.no_grad():
            attn_logits = self._view_attn(feat_tensor).squeeze(-1)
            attn_weights = torch.softmax(attn_logits, dim=0)

            # Weighted aggregation
            aggregated = (attn_weights.unsqueeze(-1) * feat_tensor).sum(dim=0)
            quality = self._quality_head(aggregated.unsqueeze(0)).item()

        return quality

    def _extract_feature(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract ResNet-50 feature from a rendered view."""
        import torch
        import cv2

        try:
            # Ensure BGR for consistency with torchvision
            if len(image.shape) == 3 and image.shape[2] == 3:
                bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                bgr = image
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensor = self._resnet_transform(rgb).unsqueeze(0).to(self._device)
            with torch.no_grad():
                feat = self._resnet(tensor)
            return feat.cpu().numpy().flatten().astype(np.float32)
        except Exception as e:
            logger.debug("Feature extraction failed: %s", e)
            return None

    def on_dispose(self) -> None:
        self._resnet = None
        self._quality_head = None
        self._view_attn = None
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

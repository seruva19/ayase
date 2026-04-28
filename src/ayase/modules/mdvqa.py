"""MD-VQA — Multi-Dimensional Quality Assessment for UGC Live Videos.

CVPR 2023 — evaluates semantic, distortion, and motion aspects
separately for UGC live streaming videos. Uses CLIP-ViT for semantic
features, ResNet-50 for distortion features, and optical flow for
motion quality assessment.

GitHub: https://github.com/zzc-1998/MD-VQA

mdvqa_semantic — semantic content quality (higher = better, 0-1)
mdvqa_distortion — distortion quality (higher = better, 0-1)
mdvqa_motion — motion quality (higher = better, 0-1)
"""

import logging
from typing import Optional, Tuple

import numpy as np

from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class MDVQAModule(PipelineModule):
    name = "mdvqa"
    description = "MD-VQA multi-dimensional UGC live VQA (CVPR 2023)"
    default_config = {
        "subsample": 8,
        "frame_size": 224,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 8)
        self.frame_size = self.config.get("frame_size", 224)
        self._ml_available = False
        self._clip_model = None
        self._clip_processor = None
        self._distortion_backbone = None
        self._semantic_head = None
        self._distortion_head = None
        self._motion_head = None
        self._device = "cpu"
        self._transform = None

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            import torch
            import torch.nn as nn
            import torchvision.models as models
            import torchvision.transforms as transforms

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Try CLIP for semantic features, fall back to ResNet-50
            clip_loaded = self._try_load_clip()

            if not clip_loaded:
                # Use ResNet-50 as semantic backbone
                resnet_semantic = models.resnet50(
                    weights=models.ResNet50_Weights.DEFAULT
                )
                self._clip_model = nn.Sequential(
                    *list(resnet_semantic.children())[:-1]
                )
                self._clip_model.eval()
                self._clip_model.to(self._device)
                self._clip_feat_dim = 2048
            else:
                self._clip_feat_dim = 512  # CLIP ViT-B/32 output dim

            # ResNet-50 backbone for distortion features
            resnet_dist = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._distortion_backbone = nn.Sequential(
                *list(resnet_dist.children())[:-1]
            )
            self._distortion_backbone.eval()
            self._distortion_backbone.to(self._device)

            # Semantic quality head: CLIP/ResNet features -> semantic score
            self._semantic_head = nn.Sequential(
                nn.Linear(self._clip_feat_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            ).to(self._device)
            self._semantic_head.eval()

            # Distortion quality head: ResNet features -> distortion score
            self._distortion_head = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            ).to(self._device)
            self._distortion_head.eval()

            # Motion quality head: flow features (6) -> motion score
            self._motion_head = nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            ).to(self._device)
            self._motion_head.eval()

            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.frame_size + 32),
                transforms.CenterCrop(self.frame_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self._ml_available = True
            semantic_backend = "CLIP" if clip_loaded else "ResNet-50"
            logger.info(
                "MD-VQA initialised on %s (semantic=%s, distortion=ResNet-50, motion=flow)",
                self._device, semantic_backend,
            )

        except ImportError:
            logger.warning(
                "MD-VQA requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            )
        except Exception as e:
            logger.warning("MD-VQA setup failed: %s", e)

    def _try_load_clip(self) -> bool:
        """Try loading CLIP ViT-B/32 for semantic features."""
        try:
            import clip

            model, preprocess = clip.load("ViT-B/32", device=self._device)
            self._clip_model = model.visual
            self._clip_model.eval()
            self._clip_processor = preprocess
            logger.debug("MD-VQA: CLIP ViT-B/32 loaded for semantic features")
            return True
        except ImportError:
            logger.debug("CLIP not available, using ResNet-50 for semantic features")
            return False
        except Exception as e:
            logger.debug("CLIP loading failed: %s", e)
            return False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            semantic, distortion, motion = self._compute_quality(sample)

            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.mdvqa_semantic = semantic
            sample.quality_metrics.mdvqa_distortion = distortion
            sample.quality_metrics.mdvqa_motion = motion
            logger.debug(
                "MD-VQA for %s: sem=%.4f dist=%.4f mot=%.4f",
                sample.path.name, semantic, distortion, motion,
            )

        except Exception as e:
            logger.warning("MD-VQA failed for %s: %s", sample.path, e)

        return sample

    def _compute_quality(
        self, sample: Sample
    ) -> Tuple[float, float, float]:
        """Multi-dimensional quality: semantic, distortion, motion."""
        import torch
        import cv2

        frames_bgr = self._load_frames(sample)
        if not frames_bgr:
            return 0.5, 0.5, 0.5

        semantic_features = []
        distortion_features = []

        with torch.no_grad():
            for frame in frames_bgr:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Semantic features (CLIP or ResNet)
                if self._clip_processor is not None:
                    # CLIP preprocessing
                    from PIL import Image
                    pil_img = Image.fromarray(rgb)
                    clip_input = self._clip_processor(pil_img).unsqueeze(0).to(
                        self._device
                    )
                    sem_feat = self._clip_model(clip_input)  # (1, 512)
                    if sem_feat.dim() == 1:
                        sem_feat = sem_feat.unsqueeze(0)
                    sem_feat = sem_feat.float()
                else:
                    tensor = self._transform(rgb).unsqueeze(0).to(self._device)
                    sem_feat = self._clip_model(tensor).squeeze(-1).squeeze(-1)
                semantic_features.append(sem_feat)

                # Distortion features (ResNet-50)
                tensor = self._transform(rgb).unsqueeze(0).to(self._device)
                dist_feat = self._distortion_backbone(tensor).squeeze(-1).squeeze(-1)
                distortion_features.append(dist_feat)

        # Aggregate features
        sem_stack = torch.cat(semantic_features, dim=0)
        sem_mean = sem_stack.mean(dim=0, keepdim=True)

        dist_stack = torch.cat(distortion_features, dim=0)
        dist_mean = dist_stack.mean(dim=0, keepdim=True)

        with torch.no_grad():
            semantic_score = self._semantic_head(sem_mean).item()
            distortion_score = self._distortion_head(dist_mean).item()

        # Motion quality from optical flow
        motion_score = self._compute_motion_quality(frames_bgr)

        return (
            float(np.clip(semantic_score, 0.0, 1.0)),
            float(np.clip(distortion_score, 0.0, 1.0)),
            float(np.clip(motion_score, 0.0, 1.0)),
        )

    def _compute_motion_quality(self, frames_bgr: list) -> float:
        """Compute motion quality from optical flow features."""
        import torch
        import cv2

        if len(frames_bgr) < 2:
            return 0.5

        flow_mags = []
        flow_stds = []

        for i in range(len(frames_bgr) - 1):
            g1 = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames_bgr[i + 1], cv2.COLOR_BGR2GRAY)
            g1 = cv2.resize(g1, (320, 240))
            g2 = cv2.resize(g2, (320, 240))

            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            flow_mags.append(float(np.mean(mag)))
            flow_stds.append(float(np.std(mag)))

        motion_features = np.array([
            np.mean(flow_mags),
            np.std(flow_mags),
            np.mean(flow_stds),
            np.std(flow_stds),
            float(np.var(np.diff(flow_mags))) if len(flow_mags) > 1 else 0.0,
            float(np.mean(np.abs(np.diff(flow_mags)))) if len(flow_mags) > 1 else 0.0,
        ], dtype=np.float32)

        motion_tensor = (
            torch.from_numpy(motion_features).float().unsqueeze(0).to(self._device)
        )
        with torch.no_grad():
            score = self._motion_head(motion_tensor).item()

        return float(score)

    def _load_frames(self, sample: Sample) -> list:
        """Load frames as BGR numpy arrays."""
        import cv2

        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                cap.release()
                return []
            n_frames = min(self.subsample, total)
            indices = np.linspace(0, total - 1, n_frames, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

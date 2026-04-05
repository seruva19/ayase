"""Spectral complexity analysis via DINOv2 feature SVD effective rank.

Computes effective rank and spectral entropy of frame-level DINOv2 features.
Low rank indicates static/redundant content; high entropy indicates chaos.
Returns spectral_entropy and spectral_rank."""

import logging
import numpy as np
import cv2
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class SpectralComplexityModule(PipelineModule):
    name = "spectral_complexity"
    description = "Analyzes spectral complexity (Effective Rank) of video features (DINOv2)"
    default_config = {
        "model_type": "dinov2_vits14", # 'dinov2_vits14', 'dinov2_vitb14'
        "sample_rate": 8, # Process every 8th frame
        "min_rank_ratio": 0.05, # Minimum ratio of rank/frames (avoid static)
        "max_entropy_threshold": 6.0, # Arbitrary complexity ceiling
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_type = self.config.get("model_type", "dinov2_vits14")
        self.sample_rate = self.config.get("sample_rate", 8)
        
        self.min_rank_ratio = self.config.get("min_rank_ratio", 0.05)
        self.max_entropy_threshold = self.config.get("max_entropy_threshold", 6.0)
        
        self._model = None
        self._device = "cpu"
        self._ml_available = False
        self._transform = None

    def setup(self):
        try:
            import torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Setting up Spectral Complexity (DINOv2) on {self._device}...")
            
            # Use Torch Hub for DINOv2
            self._model = torch.hub.load('facebookresearch/dinov2', self.model_type).to(self._device)
            self._model.eval()
            
            # DINOv2 transforms
            from torchvision import transforms
            self._transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)), # DINOv2 patch size fit
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            self._ml_available = True
            
        except Exception as e:
            logger.warning(f"Failed to load DINOv2 for spectral analysis: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample
        
        # Only meaningful for Videos
        if not sample.is_video:
            return sample

        try:
            import torch
            frames = self._load_frames(sample, step=self.sample_rate)
            if len(frames) < 4:
                # Need enough frames for spectral analysis
                return sample
            
            features_list = []
            
            with torch.no_grad():
                for frame in frames:
                    input_t = self._transform(frame).unsqueeze(0).to(self._device)
                    # DINOv2 extraction
                    feats = self._model(input_t) # (1, embed_dim)
                    features_list.append(feats)
                    
            # Stack: (T, D)
            features = torch.cat(features_list, dim=0)
            
            # Center the features
            features = features - features.mean(dim=0, keepdim=True)
            
            # SVD
            # U, S, V = torch.svd(features)
            # Use PCA / covariance-based singular values
            # S are singular values of Data Matrix X
            
            _, S, _ = torch.linalg.svd(features)
            
            # Singular Values to Probabilities
            # Normalize sum(S) or sum(S^2)? 
            # Effective Rank is typically exp(Entropy of normalized singular values)
            
            # Normalize S such that sum(p) = 1
            s_sum = torch.sum(S)
            if s_sum == 0:
                p = S * 0
            else:
                p = S / s_sum
            
            # Entropy
            # Mask zeros
            p = p[p > 0]
            entropy = -torch.sum(p * torch.log(p)).item()
            
            effective_rank = np.exp(entropy)
            
            # Analyze
            # High rank = High complexity / Randomness?
            # Low rank = Low complexity / Correlation / Static
            
            # Rank Ratio (Effective Rank / Number of Frames)
            # If Rank is 1 regardless of frames -> Static.
            rank_ratio = effective_rank / len(frames)
            
            from ayase.models import QualityMetrics
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()

            sample.quality_metrics.spectral_entropy = float(entropy)
            sample.quality_metrics.spectral_rank = float(rank_ratio)
            
            if rank_ratio < self.min_rank_ratio:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Low Spectral Complexity (Rank Ratio: {rank_ratio:.3f})",
                        details={"effective_rank": effective_rank, "rank_ratio": rank_ratio},
                        recommendation="Video acts like a slideshow or static image. Low training value."
                    )
                )
            
            if entropy > self.max_entropy_threshold:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High Spectral Complexity (Entropy: {entropy:.2f})",
                        details={"spectral_entropy": entropy},
                        recommendation="Video content is extremely chaotic or noisy (High Intruder Dimensions)."
                    )
                )

        except Exception as e:
            logger.warning(f"Spectral analysis failed: {e}")

        return sample

    def _load_frames(self, sample: Sample, step: int = 8) -> list:
        max_frames = self.config.get("max_frames", 300)
        frames = []
        try:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Simple uniform sampling
            for i in range(0, total, step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if len(frames) >= max_frames:
                    break
            cap.release()
        except Exception as e:
            logger.debug(f"Failed to load frames for spectral analysis: {e}")
        return frames


class SpectralCompatModule(SpectralComplexityModule):
    """Compatibility alias matching filename-based discovery."""

    name = "spectral"

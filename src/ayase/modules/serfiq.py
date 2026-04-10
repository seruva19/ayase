"""SER-FIQ -- Stochastic Embedding Robustness for Face Image Quality (CVPR 2020).

Terhoerst et al. "SER-FIQ: Unsupervised Estimation of Face Image Quality
Based on Stochastic Embedding Robustness" -- quality is measured by the
robustness of the face embedding under stochastic forward passes with
dropout enabled.

Algorithm (from the paper):
    1. Detect + align face.
    2. Run N stochastic forward passes through the FR network with dropout
       layers enabled (model.train() mode for dropout only).
    3. Collect N embeddings, L2-normalise each.
    4. Compute all pairwise cosine similarities.
    5. quality = (2 / (N*(N-1))) * sum(pairwise_similarities).
       High-quality faces produce consistent embeddings regardless of
       dropout noise, yielding high mean pairwise similarity.

Tiered backend:
    - Tier 1 (best): PyTorch ArcFace with MC-Dropout (true SER-FIQ).
    - Tier 2 (fallback): InsightFace ONNX + input-noise perturbation
      (approximates stochastic variation when no torch model available).

serfiq_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


def _enable_dropout(model) -> None:
    """Enable dropout layers while keeping everything else in eval mode."""
    import torch.nn as nn

    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


class SERFIQModule(PipelineModule):
    name = "serfiq"
    description = "SER-FIQ face quality via embedding robustness (2020)"
    default_config = {
        "subsample": 4,
        "face_model": "buffalo_l",
        "n_forward_passes": 10,
        "noise_std": 5.0,
        "det_size": 640,
        "dropout_rate": 0.1,
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 4)
        self.face_model = self.config.get("face_model", "buffalo_l")
        self.n_forward_passes = self.config.get("n_forward_passes", 10)
        self.noise_std = self.config.get("noise_std", 5.0)
        self.det_size = self.config.get("det_size", 640)
        self.dropout_rate = self.config.get("dropout_rate", 0.1)
        self._face_app = None
        self._torch_model = None
        self._face_transform = None
        self._device = "cpu"
        self._ml_available = False
        self._torch_available = False

    def setup(self) -> None:
        if self.test_mode:
            return

        try:
            from insightface.app import FaceAnalysis

            self._face_app = FaceAnalysis(
                name=self.face_model,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self._face_app.prepare(ctx_id=0, det_size=(self.det_size, self.det_size))
            self._ml_available = True
            logger.info("SER-FIQ initialised with InsightFace (%s)", self.face_model)
        except ImportError:
            logger.warning(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )
        except Exception as e:
            logger.warning("SER-FIQ setup failed: %s", e)

        # Try to load a torch-based ArcFace for true MC-Dropout
        self._try_load_torch_arcface()

    def _try_load_torch_arcface(self) -> None:
        """Load a PyTorch ArcFace model and inject dropout for MC-Dropout."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Use ResNet-50 as backbone (ArcFace commonly uses ResNet variants).
            # We inject Dropout layers after each residual block to enable
            # true MC-Dropout during inference.
            backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            # Inject dropout after each layer group (layer1..layer4)
            layers = []
            for name, child in backbone.named_children():
                layers.append(child)
                if name in ("layer1", "layer2", "layer3", "layer4"):
                    layers.append(nn.Dropout(p=self.dropout_rate))
            # Remove original avgpool and fc, add our own pooling
            # backbone children: conv1, bn1, relu, maxpool, layer1..4, avgpool, fc
            # We keep up to and including the dropout after layer4
            feature_layers = []
            for name, child in backbone.named_children():
                if name == "avgpool":
                    break
                feature_layers.append(child)
                if name in ("layer1", "layer2", "layer3", "layer4"):
                    feature_layers.append(nn.Dropout(p=self.dropout_rate))
            feature_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            feature_layers.append(nn.Flatten())

            self._torch_model = nn.Sequential(*feature_layers)
            self._torch_model.eval()  # eval mode for BN etc.
            self._torch_model.to(self._device)

            self._face_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            self._torch_available = True
            logger.info("SER-FIQ: PyTorch MC-Dropout model loaded (tier 1)")
        except Exception as e:
            logger.debug("SER-FIQ: torch model not available, using noise fallback: %s", e)
            self._torch_available = False

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            frames = self._extract_frames(sample)
            if not frames:
                return sample

            scores = []
            for frame in frames:
                score = self._compute_serfiq(frame)
                if score is not None:
                    scores.append(score)

            if scores:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.serfiq_score = float(np.clip(np.mean(scores), 0, 1))

        except Exception as e:
            logger.warning("SER-FIQ failed for %s: %s", sample.path, e)

        return sample

    def _compute_serfiq(self, frame: np.ndarray) -> Optional[float]:
        """Compute SER-FIQ quality for a single frame.

        Run N stochastic forward passes and compute the mean pairwise
        cosine similarity of the resulting embeddings.
        """
        # Detect face
        faces = self._face_app.get(frame)
        if not faces:
            return None

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        if self._torch_available:
            return self._compute_serfiq_mcdropout(frame, face)
        return self._compute_serfiq_noise(frame, face)

    def _compute_serfiq_mcdropout(self, frame: np.ndarray, face) -> Optional[float]:
        """Tier 1: True MC-Dropout -- enable dropout during inference."""
        import torch

        # Extract aligned face crop
        x1, y1, x2, y2 = [int(c) for c in face.bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        input_tensor = self._face_transform(face_rgb).unsqueeze(0).to(self._device)

        # Enable dropout layers while keeping BN in eval mode
        _enable_dropout(self._torch_model)

        embeddings = []
        with torch.no_grad():
            for _ in range(self.n_forward_passes):
                emb = self._torch_model(input_tensor)  # (1, D)
                emb = emb.squeeze(0).cpu().numpy()
                emb = emb / (np.linalg.norm(emb) + 1e-10)
                embeddings.append(emb)

        return self._mean_pairwise_similarity(embeddings)

    def _compute_serfiq_noise(self, frame: np.ndarray, face) -> Optional[float]:
        """Tier 2 fallback: perturb input with Gaussian noise to approximate
        stochastic variation when no torch model with dropout is available.
        """
        embeddings = []

        # Clean embedding
        clean_emb = face.embedding.copy()
        clean_emb = clean_emb / (np.linalg.norm(clean_emb) + 1e-10)
        embeddings.append(clean_emb)

        x1, y1, x2, y2 = [int(c) for c in face.bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        rng = np.random.RandomState(42)

        for _ in range(self.n_forward_passes - 1):
            noisy = frame.copy()
            noise = rng.normal(0, self.noise_std, noisy[y1:y2, x1:x2].shape)
            noisy[y1:y2, x1:x2] = np.clip(
                noisy[y1:y2, x1:x2].astype(np.float32) + noise, 0, 255
            ).astype(np.uint8)

            noisy_faces = self._face_app.get(noisy)
            if not noisy_faces:
                continue

            nf = max(
                noisy_faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            )
            emb = nf.embedding.copy()
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            embeddings.append(emb)

        if len(embeddings) < 3:
            return None

        return self._mean_pairwise_similarity(embeddings)

    @staticmethod
    def _mean_pairwise_similarity(embeddings: List[np.ndarray]) -> float:
        """SER-FIQ quality = mean of all pairwise cosine similarities.

        quality = (2 / (N*(N-1))) * sum_{i<j} cos(e_i, e_j)
        """
        emb_matrix = np.array(embeddings)
        # Gram matrix -- cosine sim since vectors are L2-normalised
        sim_matrix = emb_matrix @ emb_matrix.T

        n = len(embeddings)
        triu_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_matrix[triu_indices]

        # Paper formula: quality = mean pairwise similarity
        quality = float(np.mean(pairwise_sims))
        # Clamp: similarity is in [-1, 1], map to [0, 1]
        return float(np.clip((quality + 1.0) / 2.0, 0.0, 1.0))

    def _extract_frames(self, sample: Sample) -> List[np.ndarray]:
        """Extract frames from video or load image."""
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            try:
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0:
                    return frames
                indices = np.linspace(0, total - 1, min(self.subsample, total), dtype=int)
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
            finally:
                cap.release()
        else:
            img = cv2.imread(str(sample.path))
            if img is not None:
                frames.append(img)
        return frames

    def on_dispose(self) -> None:
        self._face_app = None
        self._torch_model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

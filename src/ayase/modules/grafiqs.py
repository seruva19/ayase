"""GraFIQs -- Gradient-Based Face Image Quality (CVPRW 2024).

Babnik et al. "GraFIQs: Face Image Quality Assessment Using Gradient
Magnitudes" -- quality is derived from the gradient magnitude of a
BatchNorm statistics loss w.r.t. the input image.

Algorithm (from the paper):
    1. Detect + align face with InsightFace.
    2. Feed aligned face crop through a pretrained FR network (e.g. ArcFace).
    3. Compute the BN-statistics loss: MSE between each BatchNorm layer's
       running mean/variance and the batch statistics from the test sample.
    4. Backpropagate this loss to the input image.
    5. quality = 1 / (1 + sum_of_abs_gradients).
       Well-recognised faces have low BN-statistics loss, producing small
       gradients; poorly-recognised faces deviate from training statistics,
       producing large gradients.

Tiered backend:
    - Tier 1 (best): PyTorch model with BN-statistics gradient (true GraFIQs).
    - Tier 2 (fallback): InsightFace embedding norm + det_score proxy.

grafiqs_score -- higher = better quality (0-1)
"""

import logging
from typing import List, Optional

import cv2
import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class GraFIQsModule(PipelineModule):
    name = "grafiqs"
    description = "GraFIQs gradient face quality (CVPRW 2024)"
    default_config = {
        "subsample": 4,
        "face_model": "buffalo_l",
        "det_size": 640,
        "gradient_scale": 1e4,  # Scaling for gradient -> quality mapping
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 4)
        self.face_model = self.config.get("face_model", "buffalo_l")
        self.det_size = self.config.get("det_size", 640)
        self.gradient_scale = self.config.get("gradient_scale", 1e4)
        self._face_app = None
        self._torch_model = None
        self._bn_layers = []
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
            logger.info("GraFIQs initialised with InsightFace (%s)", self.face_model)
        except ImportError:
            logger.warning(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )
        except Exception as e:
            logger.warning("GraFIQs setup failed: %s", e)

        # Try to load a torch model for true gradient computation
        self._try_load_torch_model()

    def _try_load_torch_model(self) -> None:
        """Load a PyTorch model with BatchNorm layers for GraFIQs gradient computation."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms

            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            # Use ResNet-50 pretrained on ImageNet as differentiable FR proxy.
            # The key requirement is BatchNorm layers whose running statistics
            # represent "well-recognised" inputs from training.
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # Remove final FC -- we only need the feature extractor
            self._torch_model = nn.Sequential(*list(model.children())[:-1])
            self._torch_model.eval()
            self._torch_model.to(self._device)

            # Collect all BatchNorm layers for loss computation
            self._bn_layers = []
            for module in self._torch_model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    self._bn_layers.append(module)

            self._face_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((112, 112)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])

            self._torch_available = True
            logger.info(
                "GraFIQs: torch gradient model loaded (%d BN layers)",
                len(self._bn_layers),
            )
        except Exception as e:
            logger.debug("GraFIQs: torch model not available, using fallback: %s", e)
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
                score = self._compute_grafiqs(frame)
                if score is not None:
                    scores.append(score)

            if scores:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.grafiqs_score = float(np.clip(np.mean(scores), 0, 1))

        except Exception as e:
            logger.warning("GraFIQs failed for %s: %s", sample.path, e)

        return sample

    def _compute_grafiqs(self, frame: np.ndarray) -> Optional[float]:
        """Compute GraFIQs quality for a single frame."""
        faces = self._face_app.get(frame)
        if not faces:
            return None

        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )

        if self._torch_available:
            return self._compute_bn_gradient_quality(frame, face)

        # Fallback: use embedding properties if torch model unavailable
        return self._compute_embedding_quality(face)

    def _compute_bn_gradient_quality(self, frame: np.ndarray, face) -> Optional[float]:
        """GraFIQs core: BN-statistics loss gradient w.r.t. input.

        1. Forward pass through the model (eval mode keeps running stats frozen).
        2. For each BN layer, compute MSE between the running mean/var
           (learned during training) and the actual batch statistics of the
           test sample.  This measures how much the test input deviates
           from the training distribution.
        3. Backpropagate the total BN loss to the input.
        4. quality = 1 / (1 + sum(|grad|)).
        """
        import torch

        x1, y1, x2, y2 = [int(c) for c in face.bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        try:
            input_tensor = self._face_transform(face_rgb).unsqueeze(0).to(self._device)
            input_tensor = input_tensor.detach().requires_grad_(True)

            # Hook to capture intermediate BN inputs
            bn_inputs = {}
            hooks = []

            def make_hook(layer_id):
                def hook_fn(module, inp, out):
                    bn_inputs[layer_id] = inp[0]
                return hook_fn

            for i, bn_layer in enumerate(self._bn_layers):
                hooks.append(bn_layer.register_forward_hook(make_hook(i)))

            # Forward pass
            _ = self._torch_model(input_tensor)

            # Compute BN statistics loss: MSE between running stats and
            # the test-sample batch statistics
            bn_loss = torch.tensor(0.0, device=self._device, requires_grad=True)
            for i, bn_layer in enumerate(self._bn_layers):
                if i not in bn_inputs:
                    continue
                feat = bn_inputs[i]  # (1, C, H, W)
                # Compute sample statistics across spatial dims
                sample_mean = feat.mean(dim=(0, 2, 3))  # (C,)
                sample_var = feat.var(dim=(0, 2, 3))  # (C,)

                running_mean = bn_layer.running_mean.detach()
                running_var = bn_layer.running_var.detach()

                # MSE between running and sample statistics
                mean_loss = torch.nn.functional.mse_loss(sample_mean, running_mean)
                var_loss = torch.nn.functional.mse_loss(sample_var, running_var)
                bn_loss = bn_loss + mean_loss + var_loss

            # Remove hooks
            for hook in hooks:
                hook.remove()

            if bn_loss.item() == 0.0:
                return self._compute_embedding_quality(face)

            # Backward pass to get gradient w.r.t. input
            bn_loss.backward()

            grad = input_tensor.grad
            if grad is None:
                return self._compute_embedding_quality(face)

            # GraFIQs: quality inversely proportional to sum of |grad|
            grad_sum = float(torch.sum(torch.abs(grad)).item())

            # Invert: small gradient = high quality
            quality = 1.0 / (1.0 + grad_sum / self.gradient_scale)
            return float(np.clip(quality, 0.0, 1.0))

        except Exception as e:
            logger.debug("GraFIQs gradient computation failed: %s", e)
            return self._compute_embedding_quality(face)

    def _compute_embedding_quality(self, face) -> Optional[float]:
        """Tier 2 fallback: quality from embedding properties.

        Use embedding norm + detection confidence as quality proxy.
        """
        embedding = face.embedding
        norm = float(np.linalg.norm(embedding))
        det_score = float(getattr(face, "det_score", 0.5))

        # Combine norm (MagFace-like) with detection confidence
        norm_quality = np.clip((norm - 10.0) / 20.0, 0.0, 1.0)
        quality = 0.6 * norm_quality + 0.4 * det_score
        return float(np.clip(quality, 0.0, 1.0))

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
        self._bn_layers = []
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

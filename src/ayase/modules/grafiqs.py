"""GraFIQs -- Gradient-Based Face Image Quality (CVPRW 2024).

Kolf, Damer, Boutros "GraFIQs: Face Image Quality Assessment Using Gradient
Magnitudes" -- official protocol (extract_grafiqs.py + backbones/bn.py in
github.com/jankolf/GraFIQs):

    1. Detect + align the face to the FR input convention (112x112).
    2. Feed it through a pretrained face-recognition network with BatchNorm
       layers (facexlib ArcFace IR-SE50, FR-trained).
    3. BN-statistics loss: for each BatchNorm2d layer, MSE between the batch
       mean/variance of the test sample and the stored running statistics,
       summed over layers and divided by their count.
    4. quality signal = sum |d(BNS_loss)/d(image)| — the raw GraFIQs
       magnitude. Lower = closer to the FR training distribution = better
       face quality.

grafiqs_score -- raw |grad| sum; lower = better quality.
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
    provenance = "adapted"
    sources = {
        "grafiqs_score": "GraFIQs (Kolf, Damer, Boutros, CVPRW 2024) — https://github.com/jankolf/GraFIQs",
    }
    deviations = {
        "grafiqs_score": "backbone is facexlib ArcFace IR-SE50 instead of upstream iresnet50/100 "
        "(the same MS1M-class weights are unavailable); alignment uses InsightFace norm_crop "
        "112x112; emits raw Σ|∇| from the upstream image branch, while videos report the "
        "mean over up to four uniformly sampled frames",
    }
    description = "GraFIQs gradient face quality (CVPRW 2024; raw |grad|, lower=better)"
    default_config = {
        "subsample": 4,
        "face_model": "buffalo_l",
        "det_size": 640,
    }
    metric_groups = {
        "grafiqs_score": "face",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.subsample = self.config.get("subsample", 4)
        self.face_model = self.config.get("face_model", "buffalo_l")
        self.det_size = self.config.get("det_size", 640)
        self._face_app = None
        self._torch_model = None
        self._bn_layers = []
        self._device = "cpu"
        self._ml_available = False
        self._torch_available = False
        self._backend = "unavailable"

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
            logger.info("GraFIQs initialised with InsightFace (%s)", self.face_model)
        except ImportError:
            logger.warning(
                "insightface not installed. Install with: pip install insightface onnxruntime"
            )
        except Exception as e:
            logger.warning("GraFIQs setup failed: %s", e)

        # Load the torch model for the BN-statistics gradient (the GraFIQs core).
        self._try_load_torch_model()

        # The real GraFIQs computation needs both the detector and the gradient
        # model; without either there is no honest metric to report.
        if self._face_app is not None and self._torch_available:
            self._ml_available = True
            self._backend = "grafiqs_bn_gradient"
        else:
            self._ml_available = False
            self._backend = "unavailable"
            logger.warning(
                "GraFIQs unavailable: requires both InsightFace and a torch BN-gradient model."
            )

    def _try_load_torch_model(self) -> None:
        """Load the pretrained FR network (facexlib ArcFace IR-SE50) whose
        BatchNorm running statistics carry the FR training distribution —
        the backbone family the official GraFIQs protocol is built on."""
        try:
            import torch  # noqa: F401
            import torch.nn as nn
            from ayase.runtime import resolve_torch_device

            self._device = resolve_torch_device(self.config.get("device", "auto"))

            from facexlib.recognition import init_recognition_model

            self._torch_model = init_recognition_model(
                "arcface", device=self._device
            )
            self._torch_model.eval()

            # All BatchNorm layers participate in the BNS loss.
            self._bn_layers = [
                m for m in self._torch_model.modules()
                if isinstance(m, nn.BatchNorm2d)
            ]
            if not self._bn_layers:
                raise RuntimeError("FR backbone has no BatchNorm2d layers")

            self._torch_available = True
            logger.info(
                "GraFIQs: FR gradient model loaded (%d BN layers)",
                len(self._bn_layers),
            )
        except Exception as e:
            logger.debug("GraFIQs: FR gradient model not available: %s", e)
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
                sample.quality_metrics.grafiqs_score = float(np.mean(scores))

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

        return self._compute_bn_gradient_quality(frame, face)

    def _compute_bn_gradient_quality(self, frame: np.ndarray, face) -> Optional[float]:
        """GraFIQs core: BN-statistics loss gradient w.r.t. input.

        1. Forward pass through the FR backbone (eval mode keeps running
           stats frozen).
        2. For each BN layer, MSE between the running mean/var and the actual
           batch statistics of the test sample, summed over layers and
           divided by their count (verbatim ``bn.py`` get_BN).
        3. Backpropagate the BNS loss to the aligned input image.
        4. GraFIQs signal = sum(|grad|) — raw magnitude, lower = better.
        """
        import torch

        # Aligned 112x112 face in the FR input convention (norm_crop is the
        # standard ArcFace alignment used for FR training/eval).
        try:
            from insightface.utils import face_align

            face_rgb = face_align.norm_crop(
                frame, landmark=face.kps, image_size=112
            )
            face_rgb = cv2.cvtColor(face_rgb, cv2.COLOR_BGR2RGB)
        except Exception:
            return None

        try:
            t = (
                torch.from_numpy(face_rgb)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .to(self._device)
                / 255.0
            )
            input_tensor = ((t - 0.5) / 0.5).detach().requires_grad_(True)

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
                # Compute sample statistics across spatial dims — upstream uses
                # the biased variance (unbiased=False).
                sample_mean = feat.mean(dim=(0, 2, 3))  # (C,)
                sample_var = feat.var(dim=(0, 2, 3), unbiased=False)  # (C,)

                running_mean = bn_layer.running_mean.detach()
                running_var = bn_layer.running_var.detach()

                # MSE between running and sample statistics
                mean_loss = torch.nn.functional.mse_loss(sample_mean, running_mean)
                var_loss = torch.nn.functional.mse_loss(sample_var, running_var)
                bn_loss = bn_loss + mean_loss + var_loss

            # Upstream divides the accumulated BNS loss by the layer count.
            if self._bn_layers:
                bn_loss = bn_loss / len(self._bn_layers)

            # Remove hooks
            for hook in hooks:
                hook.remove()

            if bn_loss.item() == 0.0:
                return None

            # Backward pass to get gradient w.r.t. input
            bn_loss.backward()

            grad = input_tensor.grad
            if grad is None:
                return None

            # GraFIQs signal: raw sum |grad| w.r.t. the aligned image.
            return float(torch.sum(torch.abs(grad)).item())

        except Exception as e:
            logger.debug("GraFIQs gradient computation failed: %s", e)
            return None

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

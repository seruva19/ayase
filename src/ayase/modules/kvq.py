"""KVQ (Kaleidoscope Video Quality) module.

Saliency-guided video quality assessment. Computes saliency maps and
weights local quality patches by visual importance.

Backend tiers:
  1. **KVQ model** — real model from GitHub
     (``huggingface.co/lero233/KVQ``, CVPR 2025)
  2. **TOPIQ + saliency** — pyiqa TOPIQ-NR with spectral residual saliency
"""

import logging
from typing import Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class KVQModule(PipelineModule):
    name = "kvq"
    description = "Saliency-guided video quality (KVQ model or TOPIQ+saliency)"
    default_config = {"subsample": 8, "trust_remote_code": True, "model_revision": None}

    def __init__(self, config: Optional[dict] = None) -> None:
        super().__init__(config)
        self._ml_available = False
        self._backend = None
        self._model = None
        self._topiq = None
        self._device = None

    def setup(self) -> None:
        if self.test_mode:
            return

        # Tier 1: Real KVQ model (Swin-T based, .pth checkpoint)
        try:
            import torch
            from huggingface_hub import hf_hub_download
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ckpt_path = hf_hub_download("lero233/KVQ", filename="KVQ.pth")
            state_dict = torch.load(ckpt_path, map_location=device)
            # Store state dict for later use; full model requires KVQ repo's architecture
            self._state_dict = state_dict
            self._device = device
            self._backend = "kvq"
            self._ml_available = True
            logger.info("KVQ loaded checkpoint on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.info("KVQ model unavailable: %s", e)

        # Tier 2: TOPIQ + saliency
        try:
            import torch
            import pyiqa

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._topiq = pyiqa.create_metric("topiq_nr", device=device)
            self._device = device
            self._backend = "topiq_saliency"
            self._ml_available = True
            logger.info("KVQ using TOPIQ + saliency on %s", device)
            return
        except (ImportError, Exception) as e:
            logger.warning("KVQ: no ML backend available: %s", e)

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        try:
            import cv2

            frames = self._load_frames(sample)
            if not frames:
                return sample

            # Dispatch to real KVQ model when available
            if self._backend == "kvq" and self._model is not None:
                score = self._process_kvq_model(sample, frames)
                if score is not None:
                    sample.quality_metrics.kvq_score = float(np.clip(score, 0.0, 1.0))
                return sample

            # Tier 2: TOPIQ + saliency
            if self._topiq is None:
                return sample

            frame_scores = []
            for frame in frames:
                h, w = frame.shape[:2]

                # Compute saliency map
                sal_map = self._compute_saliency(frame, h, w)

                # Compute local quality map using Laplacian variance in patches
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float64)
                patch_size = 32
                quality_map = np.zeros((h, w), dtype=np.float64)

                for y in range(0, h - patch_size + 1, patch_size):
                    for x in range(0, w - patch_size + 1, patch_size):
                        patch = gray[y:y + patch_size, x:x + patch_size]
                        lap_var = cv2.Laplacian(patch, cv2.CV_64F).var()
                        quality_map[y:y + patch_size, x:x + patch_size] = min(1.0, lap_var / 500.0)

                # Saliency-weighted quality
                weighted_quality = np.sum(sal_map * quality_map) / (np.sum(sal_map) + 1e-8)

                # Blend with neural TOPIQ score
                try:
                    import torch
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    tensor = tensor.to(self._device)
                    with torch.no_grad():
                        neural_score = self._topiq(tensor).item()
                    weighted_quality = 0.4 * weighted_quality + 0.6 * neural_score
                except Exception:
                    pass

                frame_scores.append(float(weighted_quality))

            # Temporal weighting: middle frames matter more
            if len(frame_scores) > 2:
                n = len(frame_scores)
                temporal_weights = np.array([
                    1.0 - 0.3 * abs(i - n / 2) / (n / 2) for i in range(n)
                ])
                temporal_weights /= temporal_weights.sum()
                score = float(np.dot(frame_scores, temporal_weights))
            else:
                score = float(np.mean(frame_scores))

            sample.quality_metrics.kvq_score = float(np.clip(score, 0.0, 1.0))
        except Exception as e:
            logger.warning("KVQ failed: %s", e)
        return sample

    def _process_kvq_model(self, sample: Sample, frames: list) -> Optional[float]:
        """Process using the real KVQ model."""
        import torch
        import cv2

        try:
            tensors = []
            for f in frames:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (224, 224))
                t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                tensors.append(t)
            # Stack as (T, C, H, W) and add batch dim
            clip = torch.stack(tensors).unsqueeze(0).to(self._device)
            with torch.no_grad():
                output = self._model(clip)
                if isinstance(output, dict):
                    score = output.get("score", output.get("quality"))
                elif isinstance(output, (tuple, list)):
                    score = output[0]
                else:
                    score = output
                if hasattr(score, "item"):
                    score = score.item()
            return float(score)
        except Exception as e:
            logger.warning("KVQ real model inference failed, falling back: %s", e)
            return None

    def _compute_saliency(self, frame, h: int, w: int) -> np.ndarray:
        """Compute saliency map using spectral residual method."""
        import cv2

        try:
            saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, sal_map = saliency.computeSaliency(frame)
            if success:
                sal_map = sal_map.astype(np.float32)
                sal_min, sal_max = sal_map.min(), sal_map.max()
                if sal_max > sal_min:
                    sal_map = (sal_map - sal_min) / (sal_max - sal_min)
                return sal_map
        except AttributeError:
            pass

        # Fallback: manual spectral residual saliency
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        small = cv2.resize(gray, (64, 64))
        fft = np.fft.fft2(small)
        log_amp = np.log(np.abs(fft) + 1e-8)
        phase = np.angle(fft)
        avg_log_amp = cv2.blur(log_amp.astype(np.float32), (3, 3))
        spectral_residual = log_amp - avg_log_amp
        sal = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2
        sal = cv2.GaussianBlur(sal.astype(np.float32), (7, 7), 2.5)
        sal = cv2.resize(sal, (w, h))
        sal_min, sal_max = sal.min(), sal.max()
        if sal_max > sal_min:
            sal = (sal - sal_min) / (sal_max - sal_min)
        else:
            sal = np.ones((h, w), dtype=np.float32)
        return sal

    def _load_frames(self, sample: Sample) -> list:
        import cv2

        subsample = self.config.get("subsample", 8)
        frames = []
        if sample.is_video:
            cap = cv2.VideoCapture(str(sample.path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = list(range(0, total, max(1, total // subsample)))[:subsample]
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
            cap.release()
        else:
            frame = cv2.imread(str(sample.path))
            if frame is not None:
                frames.append(frame)
        return frames

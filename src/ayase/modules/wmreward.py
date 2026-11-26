"""WMReward latent prediction surprise using the V-JEPA 2 ViT-g predictor."""

import hashlib
import json
import logging
import os
from pathlib import Path

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

WEIGHTS_URL = "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt"
WEIGHTS_SHA256 = "67129f011434e605d894e69f2c8e13d9db118deabe59d54bf6e0fa62c2c5cb8e"
IMAGE_SIZE = 256
WINDOW_SIZE = 16
CONTEXT_FRAMES = 8
WINDOW_STRIDE = 2
MAX_FRAMES = 49


def _load_vitg(path, device):
    import torch

    from ayase.third_party.vjepa2.hub.backbones import vjepa2_vit_giant

    encoder, predictor = vjepa2_vit_giant(pretrained=False)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    for model, name in ((encoder, "encoder"), (predictor, "predictor")):
        tensors = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in checkpoint.pop(name).items()
        }
        status = model.load_state_dict(tensors, strict=False)
        if status.missing_keys or set(status.unexpected_keys) - {
            "pos_embed",
            "predictor_pos_embed",
        }:
            raise ValueError(f"Incompatible V-JEPA 2 {name} checkpoint: {status}")
        model.to(device).eval()
    return encoder, predictor


class WMRewardModule(PipelineModule):
    name = "wmreward"
    description = "WMReward latent prediction surprise (V-JEPA 2 ViT-g)"
    input_type = "video"
    implementation_status = "implemented"
    provenance = "published"
    default_config = {"checkpoint_path": None, "device": "auto"}
    metric_groups = {"latent_prediction_reward": "temporal"}
    metric_info = {
        "latent_prediction_reward": "Mean 1 − cosine similarity of predicted and observed future tokens"
    }
    sources = {
        "latent_prediction_reward": "WMReward (CVPR 2026) — https://facebookresearch.github.io/WMReward/"
    }
    models = [
        {
            "id": "wmreward/vitg.pt",
            "type": "local",
            "url": WEIGHTS_URL,
            "task": "Official V-JEPA 2 ViT-g encoder and predictor",
            "size": "16.46 GB",
            "auto_download": True,
        }
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self._encoder = self._predictor = self._device = self._backend = None

    @staticmethod
    def _verify_checkpoint(checkpoint: Path, sidecar: Path) -> None:
        st = checkpoint.stat()
        if sidecar.is_file():
            try:
                cached = json.loads(sidecar.read_text(encoding="utf-8"))
                if (
                    isinstance(cached, dict)
                    and cached.get("sha256") == WEIGHTS_SHA256
                    and cached.get("size") == st.st_size
                    and cached.get("mtime_ns") == st.st_mtime_ns
                ):
                    return
            except Exception:
                pass

        digest = hashlib.sha256()
        with checkpoint.open("rb") as file:
            for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != WEIGHTS_SHA256:
            sidecar.unlink(missing_ok=True)
            raise ValueError("WMReward ViT-g checkpoint has an unexpected SHA-256 digest")

        tmp = sidecar.with_suffix(".tmp")
        payload = json.dumps(
            {
                "sha256": WEIGHTS_SHA256,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, sidecar)

    def setup(self):
        if self.test_mode:
            return
        try:
            import torch

            from ayase.config import download_model_file
            from ayase.managed_runtime import _SetupLock

            device = self.config["device"]
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            override = self.config["checkpoint_path"]
            if override:
                checkpoint = Path(override)
            else:
                models_dir = Path(self.config.get("models_dir", "models"))
                directory = models_dir / "wmreward"
                directory.mkdir(parents=True, exist_ok=True)
                with _SetupLock(directory / "vitg.lock"):
                    checkpoint = download_model_file(
                        "wmreward/vitg.pt", WEIGHTS_URL, str(models_dir)
                    )
                    self._verify_checkpoint(checkpoint, directory / "vitg.pt.verified.json")
            self._encoder, self._predictor = _load_vitg(checkpoint, device)
            self._device = device
            self._backend = "vjepa2_vitg"
        except Exception as exc:
            logger.warning("WMReward setup failed: %s", exc)

    def _score(self, path):
        import numpy as np
        import torch
        from decord import VideoReader
        from torch.nn import functional as F
        from torchvision.transforms.functional import resize

        reader = VideoReader(str(path))
        count = min(len(reader), MAX_FRAMES)
        if count < WINDOW_SIZE:
            return None
        indices = np.linspace(0, len(reader) - 1, count, dtype=int)
        frames = torch.from_numpy(reader.get_batch(indices).asnumpy()).permute(0, 3, 1, 2)
        frames = resize(frames.float(), [IMAGE_SIZE, IMAGE_SIZE]).to(self._device)
        frames = ((frames / 127.5 - 1.0) + 1.0) * 127.5
        mean = frames.new_tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
        std = frames.new_tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
        clip = ((frames.permute(1, 0, 2, 3) / 255.0 - mean) / std).unsqueeze(0)

        encoder = self._encoder
        tokens_per_frame = (IMAGE_SIZE // encoder.patch_size) ** 2
        split = tokens_per_frame * (CONTEXT_FRAMES // encoder.tubelet_size)
        total = tokens_per_frame * (WINDOW_SIZE // encoder.tubelet_size)
        positions = torch.arange(total, device=self._device).unsqueeze(0)
        context, future = positions[:, :split].contiguous(), positions[:, split:].contiguous()
        scores = []
        with torch.inference_mode():
            for start in range(0, count - WINDOW_SIZE + 1, WINDOW_STRIDE):
                window = clip[:, :, start : start + WINDOW_SIZE].contiguous()
                observed = encoder(window)[:, split:total]
                predicted = self._predictor(encoder(window, context), context, future)
                observed = F.layer_norm(observed, (observed.size(-1),))
                predicted = F.layer_norm(predicted, (predicted.size(-1),))
                scores.append(1 - F.cosine_similarity(predicted, observed, dim=-1).mean())
        return float(torch.stack(scores).mean())

    def process(self, sample: Sample) -> Sample:
        if self._encoder is None or not sample.is_video:
            return sample
        try:
            score = self._score(sample.path)
            if score is not None:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.latent_prediction_reward = score
        except Exception as exc:
            logger.warning("WMReward failed for %s: %s", sample.path, exc)
        return sample

    def teardown(self):
        self._encoder = self._predictor = self._device = self._backend = None

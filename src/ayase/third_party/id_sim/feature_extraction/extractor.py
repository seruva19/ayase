"""ViT feature extraction helpers.

Portions of this module are adapted from Shir Amir's dino-vit-features
project (https://github.com/ShirAmir/dino-vit-features), which is released
under the MIT license.
"""

import math
import types
from pathlib import Path

import torch
import torch.nn.modules.utils as nn_utils
from torch import nn

from ..config import DINO_CHECKPOINTS, DINOV3_HUB_REPO, DINOV3_WEIGHTS_ACCESS_URL


class ViTExtractor(nn.Module):
    """Feature extractor wrapper around supported ViT backbones."""

    def __init__(
        self,
        model_type: str = "dinov3_vitb16",
        stride: int = 4,
        load_dir: str | Path = "./models",
        device: str = "cuda",
    ):
        super().__init__()
        self.model_type = model_type
        self.device = device
        self.model = ViTExtractor.create_model(model_type, load_dir)
        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride).eval().to(self.device)
        self.p = self.model.patch_embed.patch_size
        if isinstance(self.p, tuple):
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride
        self._feats = []
        self.hook_handlers = []

    @staticmethod
    def _resolve_dinov3_checkpoint(model_type: str, load_dir: str | Path) -> str:
        """Resolve the local path to Meta's gated DINOv3 backbone weights.

        The weights are not redistributed by ID-Sim; the user downloads them from
        Meta (see DINOV3_WEIGHTS_ACCESS_URL) and places them under the cache dir.
        """
        checkpoint_spec = DINO_CHECKPOINTS.get(model_type)
        if checkpoint_spec is None:
            raise ValueError(
                f"Model {model_type} not supported. Available DINOv3 models: {sorted(DINO_CHECKPOINTS)}"
            )

        load_path = Path(load_dir)
        checkpoint_name = checkpoint_spec["filename"]
        checkpoint_path = load_path / "checkpoints" / checkpoint_name
        if checkpoint_path.exists():
            return str(checkpoint_path)

        legacy_checkpoint_path = load_path / checkpoint_name
        if legacy_checkpoint_path.exists():
            return str(legacy_checkpoint_path)

        raise FileNotFoundError(
            f"DINOv3 backbone weights for {model_type!r} were not found locally.\n"
            f"Expected one of:\n"
            f"  - {checkpoint_path}\n"
            f"  - {legacy_checkpoint_path}\n\n"
            "ID-Sim does not redistribute the gated DINOv3 weights. Accept the DINOv3 "
            f"license and download {checkpoint_name!r} from:\n"
            f"  {DINOV3_WEIGHTS_ACCESS_URL}\n"
            f"then place it at {checkpoint_path}."
        )

    @staticmethod
    def create_model(model_type: str, load_dir: str | Path = "./models") -> nn.Module:
        if "dinov2" in model_type:
            # Ayase resolves the exact backbone checkpoint from Hugging Face into
            # ``load_dir/checkpoints``. torch.hub is used for architecture code only.
            torch.hub.set_dir(str(load_dir))
            checkpoint = Path(load_dir) / "checkpoints" / DINO_CHECKPOINTS[model_type]["filename"]
            if not checkpoint.is_file():
                raise FileNotFoundError(f"DINOv2 checkpoint not found: {checkpoint}")
            model = torch.hub.load("facebookresearch/dinov2", model_type, pretrained=False)
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            return model
        if "dinov3" in model_type:
            checkpoint_path = ViTExtractor._resolve_dinov3_checkpoint(model_type, load_dir)
            # Fetch the DINOv3 model code from Meta's public GitHub repo via torch.hub
            # (downloaded and cached automatically; no manual clone required) and load
            # it with the locally-provided gated weights.
            return torch.hub.load(
                DINOV3_HUB_REPO,
                model_type,
                weights=checkpoint_path,
                trust_repo=True,
            )
        else:
            raise ValueError(
                f"Model {model_type} not supported. Supported backbones are "
                "dinov2_vitb14, dinov2_vitl14, dinov3_vitb16, and dinov3_vitl16."
            )

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: tuple[int, int]):
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            n = self.pos_embed.shape[1] - 1
            if npatch == n and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert w0 * h0 == npatch, (
                f"got wrong grid size for {h}x{w} with patch_size {patch_size} and "
                f"stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"
            )
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(n)), int(math.sqrt(n)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(n), h0 / math.sqrt(n)),
                mode="bicubic",
                align_corners=False,
                recompute_scale_factor=False,
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        patch_size = model.patch_embed.patch_size
        if stride == patch_size or (stride, stride) == patch_size:
            return model

        stride = nn_utils._pair(stride)
        assert all((patch_size // s_) * s_ == patch_size for s_ in stride), (
            f"stride {stride} should divide patch_size {patch_size}"
        )

        model.patch_embed.proj.stride = stride
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def _get_hook(self):
        def _hook(model, input, output):
            del model, input
            self._feats.append(output)

        return _hook

    def _register_hooks(self, layers: list[int]) -> None:
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                self.hook_handlers.append(block.register_forward_hook(self._get_hook()))

    def _unregister_hooks(self) -> None:
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(
        self,
        batch: torch.Tensor,
        layers: int | list[int] = 11,
    ) -> list[torch.Tensor]:
        if isinstance(layers, int):
            layers = [layers]
        self._feats = []
        self._register_hooks(layers)
        try:
            _ = self.model(batch)
        finally:
            self._unregister_hooks()
        return self._feats

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11) -> torch.Tensor:
        """Return one transformer block's token descriptors as [B, num_tokens, dim].

        The token axis follows the backbone's native layout. DINOv2 returns
        [CLS, <patch tokens>]; DINOv3 returns [CLS, <register tokens>, <patch tokens>].
        """
        feats = self._extract_features(batch, layer)
        descriptor = feats[0]
        # Some ViT blocks return a (tokens, ...) tuple; keep only the token tensor.
        if isinstance(descriptor, (list, tuple)):
            descriptor = descriptor[0]
        return descriptor

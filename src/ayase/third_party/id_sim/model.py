"""Public API for ID-Sim.

Example:
    model, preprocess = id_sim(device="cuda")
    img_a = preprocess(pil_a).to("cuda")
    img_b = preprocess(pil_b).to("cuda")

    embeddings = model.embed(img_a)
    distance = model(img_a, img_b)
"""

from dataclasses import dataclass
from pathlib import Path

from geomloss import SamplesLoss
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms

from .config import ALL_ID_SIM_CONFIGS, DEFAULT_CACHE_DIR, ID_SIM_CONFIG
from .constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
)
from .feature_extraction.extractor import ViTExtractor
from .loading import ensure_weights, load_packaged_model_state

BACKBONE_CONFIG = {
    # feature_layer is the index of the transformer block whose output is used
    # as the descriptor (the last block: 12-block ViT-B -> 11, 24-block ViT-L -> 23).
    "dinov2_vitb14": {"feat_dims": {"cls": 768, "patch": 768}, "feature_layer": 11, "num_register_tokens": 0},
    "dinov2_vitl14": {"feat_dims": {"cls": 1024, "patch": 1024}, "feature_layer": 23, "num_register_tokens": 0},
    "dinov3_vitb16": {"feat_dims": {"cls": 768, "patch": 768}, "feature_layer": 11, "num_register_tokens": 4},
    "dinov3_vitl16": {"feat_dims": {"cls": 1024, "patch": 1024}, "feature_layer": 23, "num_register_tokens": 4},
}


@dataclass(frozen=True)
class ScoringPolicy:
    feat_type: str | None = None
    patch_mode: str = "pooled_cosine"


class IDSimModel(torch.nn.Module):
    """Thin public inference wrapper around the packaged ID-Sim model."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.eval()

    def _policy(self, mode: str | None):
        if mode is None:
            return None
        if mode == "cls":
            return ScoringPolicy(feat_type="cls")
        if mode == "patch":
            return ScoringPolicy(feat_type="patch")
        if mode == "joint":
            return ScoringPolicy(feat_type="cls_patch")
        raise ValueError(f"Unsupported mode: {mode}. Expected one of: cls, patch, joint")

    def _public_embeddings(self, embed_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        public = {}
        if "cls_embed" in embed_dict:
            public["cls"] = embed_dict["cls_embed"]
        if "patch_embed" in embed_dict:
            public["patch"] = embed_dict["patch_embed"]
        return public

    @torch.inference_mode()
    def forward(self, img_a, img_b, mode: str | None = None):
        policy = self._policy(mode)
        embed_a = self.model.embed(img_a, policy=policy)
        embed_b = self.model.embed(img_b, policy=policy)
        dist_dict = self.model.compute_distance_from_embeddings(embed_a, embed_b, policy=policy)

        if mode == "cls":
            return dist_dict["cls"]
        if mode == "patch":
            return dist_dict["patch"]
        if "cls" in dist_dict and "patch" in dist_dict:
            return dist_dict["cls"] + dist_dict["patch"]
        if "cls" in dist_dict:
            return dist_dict["cls"]
        return dist_dict["patch"]

    @torch.inference_mode()
    def embed(self, img, mode: str | None = None):
        policy = self._policy(mode)
        return self._public_embeddings(self.model.embed(img, policy=policy))


def id_sim(
    pretrained: bool = True,
    device="cuda",
    cache_dir=None,
    normalize_embeds: bool = True,
    id_sim_type: str = "dinov3_vitl16_cls_patch",
    hf_token=None,
):
    """Load a packaged ID-Sim checkpoint and preprocessing function.

    Args:
        id_sim_type: Which checkpoint to load. One of:
            "dinov3_vitl16_cls_patch" (default), "dinov3_vitb16_cls_patch",
            "dinov2_vitl14_cls_patch", "dinov2_vitb14_cls_patch".
        cache_dir: Root directory for cached weights. Adapter weights are
            stored under <cache_dir>/<id_sim_type>/. DINOv3 backbone weights
            are looked up directly in <cache_dir>/checkpoints/.
    """
    if id_sim_type not in ALL_ID_SIM_CONFIGS:
        raise ValueError(
            f"Unsupported id_sim_type: {id_sim_type!r}. "
            f"Supported types: {sorted(ALL_ID_SIM_CONFIGS)}"
        )
    config = ALL_ID_SIM_CONFIGS[id_sim_type]

    if cache_dir is None:
        cache_dir = str(DEFAULT_CACHE_DIR)

    # Adapter weights live in a per-type subdir; backbone weights stay at the root.
    adapter_cache_dir = str(Path(cache_dir) / id_sim_type)

    ensure_weights(config, cache_dir=adapter_cache_dir, hf_token=hf_token)
    model = PerceptualModel(
        model_type=config["model_type"],
        feat_type=config["feat_type"],
        stride=config["stride"],
        hidden_size=config.get("hidden_size", 512),
        hidden_size_patch=config.get("hidden_size_patch", 512),
        device=device,
        load_dir=cache_dir,
        normalize_embeds=normalize_embeds,
        patch_metric=config.get("patch_metric", "sinkhorn"),
        sinkhorn_reg=config.get("sinkhorn_reg", 0.05),
        sinkhorn_blur=config.get("sinkhorn_blur", 0.05),
    )

    model = load_packaged_model_state(
        model,
        config,
        cache_dir=adapter_cache_dir,
        pretrained=pretrained,
        device=device,
    )
    public_model = IDSimModel(model).to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(
                (config.get("img_size", 224), config.get("img_size", 224)),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
        ]
    )

    def preprocess(pil_img):
        pil_img = pil_img.convert("RGB")
        return transform(pil_img).unsqueeze(0)

    return public_model, preprocess

class PerceptualModel(torch.nn.Module):
    """Single-backbone ID-Sim wrapper."""

    def __init__(
        self,
        model_type: str = "dinov3_vitb16",
        feat_type: str = "cls",
        stride: str | int = "16",
        hidden_size: int = 512,
        load_dir: str = "./models",
        normalize_embeds: bool = False,
        device: str = "cuda",
        patch_metric: str = "sinkhorn",
        sinkhorn_reg: float = 0.05,
        sinkhorn_blur: float = 0.05,
        hidden_size_patch: int = 512,
    ):
        super().__init__()
        self.model_type = str(model_type)
        self.feat_type = str(feat_type)
        self.stride = int(stride)
        self.hidden_size = hidden_size
        self.hidden_size_patch = hidden_size_patch
        self.normalize_embeds = normalize_embeds
        self.patch_metric = patch_metric
        self.is_patch = self.feat_type in {"patch", "cls_patch"}
        self.scoring_policy = ScoringPolicy(feat_type=self.feat_type, patch_mode="pooled_cosine")

        self._validate_args()
        self.extractor = ViTExtractor(self.model_type, self.stride, load_dir, device=device)
        if self.feat_type == "patch":
            dim_key = "patch"
        else:
            dim_key = "cls"
        self.embed_size = BACKBONE_CONFIG[self.model_type]["feat_dims"][dim_key]

        self.mlp_cls = torch.nn.Identity()
        self.mlp_patch = torch.nn.Identity()
        if self.feat_type in {"cls", "cls_patch"}:
            self.mlp_cls = MLP(in_features=self.embed_size, hidden_size=self.hidden_size)
        if self.feat_type in {"patch", "cls_patch"}:
            patch_hidden_size = self.hidden_size_patch if self.feat_type == "cls_patch" else self.hidden_size
            self.mlp_patch = MLP(in_features=self.embed_size, hidden_size=patch_hidden_size)

        if self.is_patch:
            self.patch_dist = PatchDistance(
                kind=patch_metric,
                sinkhorn_reg=sinkhorn_reg,
                sinkhorn_blur=sinkhorn_blur,
            )
        else:
            self.patch_dist = None

    def _resolve_scoring_policy(self, policy=None):
        if policy is not None:
            resolved_feat_type = policy.feat_type or self.feat_type
            resolved_patch_mode = policy.patch_mode
        else:
            base_policy = getattr(self, "scoring_policy", None)
            resolved_feat_type = self.feat_type
            resolved_patch_mode = "pooled_cosine"
            if base_policy is not None:
                resolved_feat_type = base_policy.feat_type or resolved_feat_type
                resolved_patch_mode = base_policy.patch_mode

        if resolved_patch_mode not in {"pooled_cosine", "sinkhorn"}:
            raise ValueError(f"Unsupported patch scoring mode: {resolved_patch_mode}")
        return ScoringPolicy(feat_type=resolved_feat_type, patch_mode=resolved_patch_mode)

    def _uses_pooled_patch_scoring(self, policy: ScoringPolicy):
        return policy.patch_mode == "pooled_cosine"

    def _validate_patch_scoring_mode(self, policy: ScoringPolicy):
        if self._uses_pooled_patch_scoring(policy):
            return
        if policy.patch_mode != self.patch_metric:
            raise ValueError(
                f"Patch scoring mode {policy.patch_mode} is not available for this model; "
                f"configured patch_metric is {self.patch_metric}."
            )

    def _validate_args(self):
        if "," in self.model_type or "," in self.feat_type or "," in str(self.stride):
            raise ValueError("PerceptualModel supports a single backbone, feat_type, and stride only")
        if self.model_type not in BACKBONE_CONFIG:
            raise ValueError(f"Unsupported model_type: {self.model_type}")
        if self.feat_type not in {"cls", "patch", "cls_patch"}:
            raise ValueError(f"Feature type {self.feat_type} is not supported.")
        if self.patch_metric != "sinkhorn":
            raise ValueError(f"Unsupported patch_metric: {self.patch_metric}. Released training supports sinkhorn only.")

    def _pool_patch_tokens(self, patch_tokens):
        batch_size, num_tokens, _ = patch_tokens.shape
        side = int(num_tokens ** 0.5)
        pooled = F.adaptive_avg_pool2d(
            patch_tokens.reshape(batch_size, side, side, -1).permute(0, 3, 1, 2),
            (1, 1),
        )
        return pooled.squeeze(-1).squeeze(-1)

    def _pairwise_cosine_distance(self, embed_a, embed_b):
        embed_a_norm = F.normalize(embed_a, p=2, dim=-1)
        embed_b_norm = F.normalize(embed_b, p=2, dim=-1)
        return 1.0 - torch.mm(embed_a_norm, embed_b_norm.T)

    def _pairwise_patch_distance(self, patch_a, patch_b):
        batch_a = patch_a.shape[0]
        batch_b = patch_b.shape[0]
        patch_a_expanded = patch_a.unsqueeze(1).expand(-1, batch_b, -1, -1).reshape(
            batch_a * batch_b,
            patch_a.shape[1],
            patch_a.shape[2],
        )
        patch_b_expanded = patch_b.unsqueeze(0).expand(batch_a, -1, -1, -1).reshape(
            batch_a * batch_b,
            patch_b.shape[1],
            patch_b.shape[2],
        )
        distances = self.patch_dist(patch_a_expanded, patch_b_expanded)
        return distances.reshape(batch_a, batch_b)

    def _extract_cls_and_patch(self, img):
        layer = BACKBONE_CONFIG[self.model_type]["feature_layer"]
        return self.extractor.extract_descriptors(img, layer)

    def _extract_patch(self, img):
        # Drop the CLS token; DINOv3 leaves register tokens before patch tokens.
        return self._extract_cls_and_patch(img)[:, 1:, :]

    def _project_embeddings(self, full_feats, feat_type):
        if feat_type == "cls_patch":
            return {
                "cls_embed": self.mlp_cls(full_feats[:, 0:1]).squeeze(1),
                "patch_embed": self.mlp_patch(full_feats[:, 1:]),
            }
        if feat_type == "cls":
            return {"cls_embed": self.mlp_cls(full_feats[:, 0:1]).squeeze(1)}
        if feat_type == "patch":
            return {"patch_embed": self.mlp_patch(full_feats)}
        raise ValueError(f"Feature type {feat_type} is not supported.")

    def _normalize_embedding_dict(self, embed_dict):
        if not self.normalize_embeds:
            return embed_dict
        normalized = {}
        for key, value in embed_dict.items():
            normalized[key] = normalize_embedding(value)
        return normalized

    def _patch_distance(self, a, b):
        if self.patch_dist is None:
            raise RuntimeError("Patch distance not initialized. Set patch_metric in __init__.")
        return self.patch_dist(a, b)

    def _preprocess(self, img):
        return transforms.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )(img)

    def embed(self, img, policy=None):
        resolved_policy = self._resolve_scoring_policy(policy=policy)
        feat_type = resolved_policy.feat_type
        preprocessed = self._preprocess(img)
        if feat_type == "cls" or feat_type == "cls_patch":
            full_feats = self._extract_cls_and_patch(preprocessed)
        elif feat_type == "patch":
            full_feats = self._extract_patch(preprocessed)
        else:
            raise ValueError(f"Feature type {feat_type} is not supported.")

        # DINOv3 inserts register tokens between CLS and patch tokens. Strip
        # them before projecting so DINOv2 and DINOv3 expose the same logical
        # feature layouts: [patches] or [CLS, patches].
        num_register_tokens = BACKBONE_CONFIG[self.model_type].get("num_register_tokens", 0)
        if num_register_tokens:
            if feat_type == "patch":
                full_feats = full_feats[:, num_register_tokens:]
            elif feat_type == "cls_patch":
                full_feats = torch.cat(
                    [full_feats[:, 0:1], full_feats[:, 1 + num_register_tokens:]],
                    dim=1,
                )

        embed_dict = self._project_embeddings(full_feats, feat_type)
        return self._normalize_embedding_dict(embed_dict)

    def compute_distance_from_embeddings(self, embed_dict_a, embed_dict_b, policy=None):
        resolved_policy = self._resolve_scoring_policy(policy=policy)
        distance_dict = {}
        if "cls_embed" in embed_dict_a:
            distance_dict["cls"] = 1.0 - F.cosine_similarity(
                embed_dict_a["cls_embed"],
                embed_dict_b["cls_embed"],
                dim=-1,
            )
        if "patch_embed" in embed_dict_a:
            patch_a = embed_dict_a["patch_embed"]
            patch_b = embed_dict_b["patch_embed"]
            self._validate_patch_scoring_mode(resolved_policy)
            if self.patch_dist is not None and not self._uses_pooled_patch_scoring(resolved_policy):
                distance_dict["patch"] = self._patch_distance(patch_a, patch_b)
            else:
                patch_a_pooled = self._pool_patch_tokens(patch_a)
                patch_b_pooled = self._pool_patch_tokens(patch_b)
                distance_dict["patch"] = 1.0 - F.cosine_similarity(patch_a_pooled, patch_b_pooled, dim=-1)
        return distance_dict

    def compute_pairwise_distances(self, embed_set_a, embed_set_b, embed_type="feature_set", policy=None):
        if embed_type not in {"feature_set", "cls", "patch"}:
            raise ValueError("embed_type must be one of: feature_set, cls, patch")
        resolved_policy = self._resolve_scoring_policy(policy=policy)
        feat_type = resolved_policy.feat_type
        use_pooled_patch_scoring = self._uses_pooled_patch_scoring(resolved_policy)
        self._validate_patch_scoring_mode(resolved_policy)

        if embed_type == "cls":
            if embed_set_a.dim() != 2 or embed_set_b.dim() != 2:
                raise ValueError(
                    f"embed_type='{embed_type}' requires 2D tensors [B, D], got {embed_set_a.shape} and {embed_set_b.shape}"
                )
            return self._pairwise_cosine_distance(embed_set_a, embed_set_b)

        if embed_type == "patch":
            if embed_set_a.dim() != 3 or embed_set_b.dim() != 3:
                raise ValueError(
                    f"embed_type='patch' requires 3D tensors [B, N, D], got {embed_set_a.shape} and {embed_set_b.shape}"
                )
            if self.patch_dist is not None and not use_pooled_patch_scoring:
                return self._pairwise_patch_distance(embed_set_a, embed_set_b)
            return self._pairwise_cosine_distance(
                self._pool_patch_tokens(embed_set_a),
                self._pool_patch_tokens(embed_set_b),
            )

        if embed_set_a.dim() != 3 or embed_set_b.dim() != 3:
            raise ValueError(
                f"embed_type='feature_set' requires 3D tensors [B, N, D], got {embed_set_a.shape} and {embed_set_b.shape}"
            )
        if feat_type == "cls_patch":
            cls_a, patch_a = embed_set_a[:, 0], embed_set_a[:, 1:]
            cls_b, patch_b = embed_set_b[:, 0], embed_set_b[:, 1:]
            cls_dist = self._pairwise_cosine_distance(cls_a, cls_b)
            if self.patch_dist is not None and not use_pooled_patch_scoring:
                return cls_dist + self._pairwise_patch_distance(patch_a, patch_b)
            if use_pooled_patch_scoring:
                return cls_dist + self._pairwise_cosine_distance(
                    self._pool_patch_tokens(patch_a),
                    self._pool_patch_tokens(patch_b),
                )
            pooled_a = self._pool_patch_tokens(patch_a)
            pooled_b = self._pool_patch_tokens(patch_b)
            return self._pairwise_cosine_distance(
                torch.cat((cls_a, pooled_a), dim=-1),
                torch.cat((cls_b, pooled_b), dim=-1),
            )
        if feat_type == "patch":
            patch_a, patch_b = embed_set_a, embed_set_b
            if self.patch_dist is not None and not use_pooled_patch_scoring:
                return self._pairwise_patch_distance(patch_a, patch_b)
            return self._pairwise_cosine_distance(self._pool_patch_tokens(patch_a), self._pool_patch_tokens(patch_b))
        return self._pairwise_cosine_distance(embed_set_a, embed_set_b)

    def forward(self, img_a, img_b, policy=None):
        resolved_policy = self._resolve_scoring_policy(policy=policy)
        embed_dict_a = self.embed(img_a, policy=resolved_policy)
        embed_dict_b = self.embed(img_b, policy=resolved_policy)
        return self.compute_distance_from_embeddings(
            embed_dict_a,
            embed_dict_b,
            policy=resolved_policy,
        )

class PatchDistance(nn.Module):
    """Distance between two sets of patch embeddings for the released metrics."""

    def __init__(
        self,
        kind: str = "sinkhorn",
        sinkhorn_reg: float = 0.05,
        sinkhorn_blur: float = 0.05,
        sinkhorn_reach: float = None,
        sinkhorn_debias: bool = False,
    ):
        super().__init__()
        if kind != "sinkhorn":
            raise ValueError(
                f"Unsupported patch metric: {kind}. Released patch metric is sinkhorn."
            )
        self.kind = kind
        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_blur = sinkhorn_blur
        self.sinkhorn_reach = sinkhorn_reach
        self.sinkhorn_debias = sinkhorn_debias

        if kind == "sinkhorn":
            self.sinkhorn = SamplesLoss(
                "sinkhorn",
                p=2,
                blur=sinkhorn_blur,
                scaling=sinkhorn_reg,
                reach=sinkhorn_reach,
                debias=sinkhorn_debias,
            )

    def _pack_tokens(self, a, b):
        batch_size = a.shape[0]
        tokens_a = a.shape[1]
        tokens_b = b.shape[1]
        a_packed = a.reshape(-1, a.shape[-1])
        b_packed = b.reshape(-1, b.shape[-1])
        ptr_a = (torch.arange(1, batch_size + 1, device=a.device) * tokens_a).tolist()
        ptr_b = (torch.arange(1, batch_size + 1, device=b.device) * tokens_b).tolist()
        return a_packed, b_packed, ptr_a, ptr_b

    def _sinkhorn_patch_distance(self, a_tokens, b_tokens):
        return self.sinkhorn(a_tokens.squeeze(0), b_tokens.squeeze(0))

    def forward(self, a, b):
        a_packed, b_packed, ptr_a, ptr_b = self._pack_tokens(a, b)
        a_packed = F.normalize(a_packed, p=2, dim=-1)
        b_packed = F.normalize(b_packed, p=2, dim=-1)

        out = []
        start_a = start_b = 0
        for end_a, end_b in zip(ptr_a, ptr_b):
            a_tokens = a_packed[start_a:end_a][None]
            b_tokens = b_packed[start_b:end_b][None]
            start_a, start_b = end_a, end_b

            if a_tokens.shape[1] == 0 or b_tokens.shape[1] == 0:
                out.append(a_tokens.new_ones(1)[0])
                continue
            out.append(self._sinkhorn_patch_distance(a_tokens, b_tokens))
        return torch.stack(out, dim=0)

class MLP(torch.nn.Module):
    def __init__(self, in_features: int, hidden_size: int = 512):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_size, bias=True)
        self.fc2 = nn.Linear(hidden_size, in_features, bias=True)
        self.beta_logit = nn.Parameter(torch.tensor(-4.0))

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, img):
        z = self.fc2(F.silu(self.fc1(self.ln(img))))
        beta = torch.sigmoid(self.beta_logit)
        return z + beta * img

def normalize_embedding(embed):
    """Mean-center and L2-normalize along the last dimension.

    Works for both CLS embeddings ([B, D]) and patch embeddings ([B, N, D]).
    """
    centered = embed - embed.mean(dim=-1, keepdim=True)
    return centered / torch.norm(centered, dim=-1, keepdim=True)


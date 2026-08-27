from pathlib import Path


DEFAULT_CACHE_DIR = Path("./models/id_sim_checkpoint")

# ID-Sim loads the DINOv3 backbone *code* from Meta's public GitHub repo via
# torch.hub (no manual clone needed). Override with a pinned ref such as
# "facebookresearch/dinov3:<commit>" for stricter reproducibility.
DINOV3_HUB_REPO = "facebookresearch/dinov3"

# Meta gates the DINOv3 backbone *weights*. Accept the license and download the
# checkpoint from the page below, then place it under
# <cache_dir>/checkpoints/<filename>. ID-Sim does not redistribute these weights.
DINOV3_WEIGHTS_ACCESS_URL = "https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/"

DINO_CHECKPOINTS = {
    "dinov2_vitb14": {"filename": "dinov2_vitb14_pretrain.pth"},
    "dinov2_vitl14": {"filename": "dinov2_vitl14_pretrain.pth"},
    "dinov3_vitb16": {"filename": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"},
    "dinov3_vitl16": {"filename": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"},
}

_ADAPTER_FILES = [
    "adapter_config.json",
    "adapter_model.safetensors",
    "mlp_cls.pth",
    "mlp_patch.pth",
]

ALL_ID_SIM_CONFIGS = {
    "dinov3_vitl16_cls_patch": {
        "id_sim_type": "dinov3_vitl16_cls_patch",
        "model_type": "dinov3_vitl16",
        "feat_type": "cls_patch",
        "stride": 16,
        "img_size": 448,
        "hidden_size": 512,
        "hidden_size_patch": 512,
        "patch_metric": "sinkhorn",
        "sinkhorn_reg": 0.05,
        "sinkhorn_blur": 0.05,
        "root_files": [],
        "adapter_dir": ".",
        "adapter_files": _ADAPTER_FILES,
    },
    "dinov3_vitb16_cls_patch": {
        "id_sim_type": "dinov3_vitb16_cls_patch",
        "model_type": "dinov3_vitb16",
        "feat_type": "cls_patch",
        "stride": 16,
        "img_size": 448,
        "hidden_size": 512,
        "hidden_size_patch": 512,
        "patch_metric": "sinkhorn",
        "sinkhorn_reg": 0.05,
        "sinkhorn_blur": 0.05,
        "root_files": [],
        "adapter_dir": ".",
        "adapter_files": _ADAPTER_FILES,
    },
    "dinov2_vitl14_cls_patch": {
        "id_sim_type": "dinov2_vitl14_cls_patch",
        "model_type": "dinov2_vitl14",
        "feat_type": "cls_patch",
        "stride": 14,
        "img_size": 448,
        "hidden_size": 512,
        "hidden_size_patch": 512,
        "patch_metric": "sinkhorn",
        "sinkhorn_reg": 0.05,
        "sinkhorn_blur": 0.05,
        "root_files": [],
        "adapter_dir": ".",
        "adapter_files": _ADAPTER_FILES,
    },
    "dinov2_vitb14_cls_patch": {
        "id_sim_type": "dinov2_vitb14_cls_patch",
        "model_type": "dinov2_vitb14",
        "feat_type": "cls_patch",
        "stride": 14,
        "img_size": 448,
        "hidden_size": 512,
        "hidden_size_patch": 512,
        "patch_metric": "sinkhorn",
        "sinkhorn_reg": 0.05,
        "sinkhorn_blur": 0.05,
        "root_files": [],
        "adapter_dir": ".",
        "adapter_files": _ADAPTER_FILES,
    },
}

# Backward-compat alias — default checkpoint is DINOv3 ViT-L/16.
ID_SIM_CONFIG = ALL_ID_SIM_CONFIGS["dinov3_vitl16_cls_patch"]

ID_SIM_WEIGHTS = {
    "dinov3_vitl16_cls_patch": {
        "source": "huggingface",
        "repo_id": "chaenayo/id-sim_dinov3_vitl16_cls_patch",
        "allow_patterns": _ADAPTER_FILES,
    },
    "dinov3_vitb16_cls_patch": {
        "source": "huggingface",
        "repo_id": "chaenayo/id-sim_dinov3_vitb16_cls_patch",
        "allow_patterns": _ADAPTER_FILES,
    },
    "dinov2_vitl14_cls_patch": {
        "source": "huggingface",
        "repo_id": "chaenayo/id-sim_dinov2_vitl14_cls_patch",
        "allow_patterns": _ADAPTER_FILES,
    },
    "dinov2_vitb14_cls_patch": {
        "source": "huggingface",
        "repo_id": "chaenayo/id-sim_dinov2_vitb14_cls_patch",
        "revision": "bcd3f388bec7db42e75b165e235196833111c4ae",
        "allow_patterns": _ADAPTER_FILES,
    },
}

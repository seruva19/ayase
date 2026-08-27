"""Checkpoint and adapter loading helpers for ID-Sim."""

import json
import logging
from pathlib import Path

import peft
import torch
from packaging import version
from peft import LoraConfig, PeftModel, get_peft_model

from .config import DEFAULT_CACHE_DIR, ID_SIM_WEIGHTS


LOGGER = logging.getLogger(__name__)

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


peft_version = version.parse(peft.__version__)
min_version = version.parse("0.2.0")

if peft_version < min_version:
    raise RuntimeError(
        f"ID-Sim requires peft version {min_version} or greater. "
        "Please update peft with 'pip install --upgrade peft'."
    )


def missing_files(cache_dir: str | Path, config: dict) -> list[Path]:
    cache_path = Path(cache_dir)
    missing = []
    for required_file in config["root_files"]:
        path = cache_path / required_file
        if not path.exists():
            missing.append(path)

    adapter_dir = cache_path / config["adapter_dir"]
    if not adapter_dir.is_dir():
        missing.append(adapter_dir)
    else:
        for required_file in config["adapter_files"]:
            path = adapter_dir / required_file
            if not path.exists():
                missing.append(path)
    return missing


def _format_missing_error(id_sim_type: str, cache_dir: str | Path, missing: list[Path]) -> str:
    return (
        f"Downloaded files for {id_sim_type!r}, but the checkpoint cache is incomplete: {cache_dir}\n"
        "Missing files:\n"
        + "\n".join(f"  - {path}" for path in missing)
    )


def ensure_weights(config: dict, cache_dir: str | Path, hf_token=None) -> None:
    cache_path = Path(cache_dir)
    id_sim_type = config["id_sim_type"]
    cache_path.mkdir(parents=True, exist_ok=True)

    missing = missing_files(cache_path, config)
    if not missing:
        LOGGER.info("Using cached weights in %s", cache_path)
        return

    weight_spec = ID_SIM_WEIGHTS.get(id_sim_type)
    if not weight_spec:
        raise RuntimeError(
            f"Packaged weights for {id_sim_type!r} are not configured. "
            f"Populate cache_dir with the expected files or configure a download source."
        )

    if isinstance(weight_spec, dict) and weight_spec.get("source") == "huggingface":
        if snapshot_download is None:
            raise RuntimeError(
                "Loading ID-Sim weights from Hugging Face requires huggingface_hub. "
                "Install it with: pip install -U huggingface_hub"
            )

        repo_id = weight_spec["repo_id"]
        LOGGER.info("Downloading checkpoint from Hugging Face repo %s", repo_id)
        try:
            snapshot_kwargs = {
                "repo_id": repo_id,
                "local_dir": str(cache_path),
                "token": hf_token,
            }
            if "revision" in weight_spec:
                snapshot_kwargs["revision"] = weight_spec["revision"]
            if "allow_patterns" in weight_spec:
                snapshot_kwargs["allow_patterns"] = weight_spec["allow_patterns"]
            snapshot_download(**snapshot_kwargs)
        except Exception as exc:
            raise RuntimeError(
                "Could not download ID-Sim weights from Hugging Face.\n\n"
                f"Repository: {repo_id}\n\n"
                "If the repository requires authentication, run:\n\n"
                "  huggingface-cli login\n\n"
                "You can also pass hf_token=... to id_sim(...)."
            ) from exc
    else:
        raise RuntimeError(f"Unsupported weight specification for {id_sim_type!r}: {weight_spec!r}")

    missing = missing_files(cache_path, config)
    if missing:
        raise RuntimeError(_format_missing_error(id_sim_type, cache_path, missing))


def load_adapter_config(adapter_dir: str | Path) -> LoraConfig:
    adapter_path = Path(adapter_dir)
    with open(adapter_path / "adapter_config.json", "r") as handle:
        adapter_config = json.load(handle)
    lora_keys = ["r", "lora_alpha", "lora_dropout", "bias", "target_modules"]
    return LoraConfig(**{key: adapter_config[key] for key in lora_keys})


def attach_peft_adapter(model, adapter_dir: str | Path, pretrained: bool = True, device: str = "cuda"):
    adapter_path = Path(adapter_dir)
    if pretrained:
        return PeftModel.from_pretrained(model, str(adapter_path)).to(device)
    lora_config = load_adapter_config(adapter_path)
    return get_peft_model(model, lora_config).to(device)


def load_heads(model, adapter_dir: str | Path, device: str = "cuda"):
    adapter_path = Path(adapter_dir)
    head_model = model.base_model.model if hasattr(model, "base_model") else model
    cls_head_path = adapter_path / "mlp_cls.pth"
    patch_head_path = adapter_path / "mlp_patch.pth"
    if cls_head_path.exists() and hasattr(head_model, "mlp_cls"):
        head_model.mlp_cls.load_state_dict(torch.load(cls_head_path, map_location=device, weights_only=True))
    if patch_head_path.exists() and hasattr(head_model, "mlp_patch"):
        head_model.mlp_patch.load_state_dict(torch.load(patch_head_path, map_location=device, weights_only=True))
    return model


def load_packaged_model_state(
    model,
    config: dict,
    cache_dir: str | Path,
    pretrained: bool = True,
    device: str = "cuda",
):
    adapter_dir = Path(cache_dir) / config["adapter_dir"]
    model = attach_peft_adapter(model, adapter_dir, pretrained=pretrained, device=device)
    model = load_heads(model, adapter_dir, device=device)
    return model.eval().requires_grad_(False)

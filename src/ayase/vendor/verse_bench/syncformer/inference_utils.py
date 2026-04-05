import torch
import torchvision

from syncformer.utils.utils import instantiate_from_config


def get_transforms(cfg, which_transforms=None):
    transforms = {}
    for mode in which_transforms or ["train", "test"]:
        ts_cfg = cfg.get(f"transform_sequence_{mode}", None)
        ts = [lambda x: x] if ts_cfg is None else [instantiate_from_config(c) for c in ts_cfg]
        transforms[mode] = torchvision.transforms.Compose(ts)
    return transforms


def get_model(cfg, device):
    model = instantiate_from_config(cfg.model)

    if cfg.model.params.vfeat_extractor.is_trainable is False:
        for params in model.vfeat_extractor.parameters():
            params.requires_grad = False
    if cfg.model.params.afeat_extractor.is_trainable is False:
        for params in model.afeat_extractor.parameters():
            params.requires_grad = False

    model = model.to(device)
    return model, model


def _apply_fn_recursive(obj, fn):
    if isinstance(obj, torch.Tensor):
        return fn(obj)
    if isinstance(obj, dict):
        return {k: _apply_fn_recursive(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_apply_fn_recursive(v, fn) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_apply_fn_recursive(v, fn) for v in obj)
    raise NotImplementedError(f"obj type: {type(obj)}")


def prepare_inputs(batch, device, get_targets=True):
    targets = None
    if get_targets:
        targets = batch["targets"]
        for key, value in targets.items():
            if "target" in key:
                targets[key] = _apply_fn_recursive(value, lambda x: x.to(device))

    aud = batch["audio"].to(device)
    vid = batch["video"].to(device)
    return aud, vid, targets

# Lazy imports — models are loaded only when requested via get_itmscore_model().

from ...constants import HF_CACHE_DIR

_MODEL_REGISTRY = {
    "blip2_itm": (".blip2_itm_model", "BLIP2_ITM_MODELS", "BLIP2ITMScoreModel"),
    "umt_itm": (".umt_itm_model", "UMT_ITM_MODELS", "UMTITMScoreModel"),
    "internvideo2_itm": (".internvideo2_itm_model", "INTERNVIDEO2_ITM_MODELS", "InternVideo2ITMScoreModel"),
}


def _load_entry(key):
    import importlib
    mod_path, models_attr, cls_attr = _MODEL_REGISTRY[key]
    mod = importlib.import_module(mod_path, package=__name__)
    return getattr(mod, models_attr), getattr(mod, cls_attr)


def list_all_itmscore_models():
    all_models = []
    for key in _MODEL_REGISTRY:
        models_dict, _ = _load_entry(key)
        all_models.extend(models_dict)
    return all_models


def get_itmscore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR):
    for key in _MODEL_REGISTRY:
        models_dict, model_cls = _load_entry(key)
        if model_name in models_dict:
            return model_cls(model_name, device=device, cache_dir=cache_dir)
    raise NotImplementedError(f"Unknown ITMScore model: {model_name}")

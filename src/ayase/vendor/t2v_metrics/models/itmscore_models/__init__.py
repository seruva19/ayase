# Lazy imports — model modules are loaded only when instantiated.

from ...constants import HF_CACHE_DIR

_MODEL_NAME_TO_MODULE = {}
_names_loaded = False

_MODULE_KEY_TO_IMPORT = {
    "blip2_itm": (".blip2_itm_model", "BLIP2_ITM_MODELS", "BLIP2ITMScoreModel"),
    "umt_itm": (".umt_itm_model", "UMT_ITM_MODELS", "UMTITMScoreModel"),
    "internvideo2_itm": (".internvideo2_itm_model", "INTERNVIDEO2_ITM_MODELS", "InternVideo2ITMScoreModel"),
}


def _ensure_names():
    global _names_loaded
    if _names_loaded:
        return
    import importlib
    import logging
    logger = logging.getLogger(__name__)
    for key, (mod_path, models_attr, _) in _MODULE_KEY_TO_IMPORT.items():
        try:
            mod = importlib.import_module(mod_path, package=__name__)
            for name in getattr(mod, models_attr):
                _MODEL_NAME_TO_MODULE[name] = key
        except ImportError as exc:
            logger.debug("Skipped %s: %s", key, exc)
    _names_loaded = True


def list_all_itmscore_models():
    _ensure_names()
    return list(_MODEL_NAME_TO_MODULE.keys())


def get_itmscore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR):
    _ensure_names()
    if model_name not in _MODEL_NAME_TO_MODULE:
        raise NotImplementedError(f"Unknown ITMScore model: {model_name}")
    key = _MODEL_NAME_TO_MODULE[model_name]
    mod_path, _, cls_attr = _MODULE_KEY_TO_IMPORT[key]
    import importlib
    mod = importlib.import_module(mod_path, package=__name__)
    model_cls = getattr(mod, cls_attr)
    return model_cls(model_name, device=device, cache_dir=cache_dir)

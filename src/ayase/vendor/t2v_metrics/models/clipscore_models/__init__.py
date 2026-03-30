# Lazy imports — model modules are loaded only when instantiated.

from ...constants import HF_CACHE_DIR

_MODEL_NAME_TO_MODULE = {}  # populated by _ensure_names()
_names_loaded = False

_MODULE_KEY_TO_IMPORT = {
    "clip": (".clip_model", "CLIP_MODELS", "CLIPScoreModel"),
    "blip2_itc": (".blip2_itc_model", "BLIP2_ITC_MODELS", "BLIP2ITCScoreModel"),
    "hpsv2": (".hpsv2_model", "HPSV2_MODELS", "HPSV2ScoreModel"),
    "pickscore": (".pickscore_model", "PICKSCORE_MODELS", "PickScoreModel"),
    "umt_clip": (".umt_clip_model", "UMT_CLIP_MODELS", "UMTCLIPScoreModel"),
    "internvideo2_clip": (".internvideo2_clip_model", "INTERNVIDEO2_CLIP_MODELS", "InternVideo2CLIPScoreModel"),
    "languagebind": (".languagebind_video_clip_model", "LANGUAGEBIND_VIDEO_CLIP_MODELS", "LanguageBindVideoCLIPScoreModel"),
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


def list_all_clipscore_models():
    _ensure_names()
    return list(_MODEL_NAME_TO_MODULE.keys())


def get_clipscore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    _ensure_names()
    if model_name not in _MODEL_NAME_TO_MODULE:
        raise NotImplementedError(f"Unknown CLIPScore model: {model_name}")
    key = _MODEL_NAME_TO_MODULE[model_name]
    mod_path, _, cls_attr = _MODULE_KEY_TO_IMPORT[key]
    import importlib
    mod = importlib.import_module(mod_path, package=__name__)
    model_cls = getattr(mod, cls_attr)
    return model_cls(model_name, device=device, cache_dir=cache_dir, **kwargs)

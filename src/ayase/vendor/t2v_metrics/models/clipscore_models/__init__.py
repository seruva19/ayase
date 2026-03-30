# Lazy imports — models are loaded only when requested via get_clipscore_model().

from ...constants import HF_CACHE_DIR

_MODEL_REGISTRY = {
    "clip": (".clip_model", "CLIP_MODELS", "CLIPScoreModel"),
    "blip2_itc": (".blip2_itc_model", "BLIP2_ITC_MODELS", "BLIP2ITCScoreModel"),
    "hpsv2": (".hpsv2_model", "HPSV2_MODELS", "HPSV2ScoreModel"),
    "pickscore": (".pickscore_model", "PICKSCORE_MODELS", "PickScoreModel"),
    "umt_clip": (".umt_clip_model", "UMT_CLIP_MODELS", "UMTCLIPScoreModel"),
    "internvideo2_clip": (".internvideo2_clip_model", "INTERNVIDEO2_CLIP_MODELS", "InternVideo2CLIPScoreModel"),
    "languagebind": (".languagebind_video_clip_model", "LANGUAGEBIND_VIDEO_CLIP_MODELS", "LanguageBindVideoCLIPScoreModel"),
}


def _load_entry(key):
    import importlib
    mod_path, models_attr, cls_attr = _MODEL_REGISTRY[key]
    mod = importlib.import_module(mod_path, package=__name__)
    return getattr(mod, models_attr), getattr(mod, cls_attr)


def list_all_clipscore_models():
    all_models = []
    for key in _MODEL_REGISTRY:
        models_dict, _ = _load_entry(key)
        all_models.extend(models_dict)
    return all_models


def get_clipscore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    for key in _MODEL_REGISTRY:
        models_dict, model_cls = _load_entry(key)
        if model_name in models_dict:
            return model_cls(model_name, device=device, cache_dir=cache_dir, **kwargs)
    raise NotImplementedError(f"Unknown CLIPScore model: {model_name}")

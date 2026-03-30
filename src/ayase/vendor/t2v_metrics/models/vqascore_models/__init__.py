# Lazy imports — models are loaded only when requested via get_vqascore_model().

from ...constants import HF_CACHE_DIR

_MODEL_REGISTRY = {
    "clip_t5": (".clip_t5_model", "CLIP_T5_MODELS", "CLIPT5Model"),
    "llava": (".llava_model", "LLAVA_MODELS", "LLaVAModel"),
    "llava16": (".llava16_model", "LLAVA16_MODELS", "LLaVA16Model"),
    "instructblip": (".instructblip_model", "InstructBLIP_MODELS", "InstructBLIPModel"),
    "gpt4v": (".gpt4v_model", "GPT4V_MODELS", "GPT4VModel"),
    "llavaov": (".llavaov_model", "LLAVA_OV_MODELS", "LLaVAOneVisionModel"),
    "mplug": (".mplug_model", "MPLUG_OWL3_MODELS", "mPLUGOwl3Model"),
    "paligemma": (".paligemma_model", "PALIGEMMA_MODELS", "PaliGemmaModel"),
    "internvl": (".internvl_model", "INTERNVL2_MODELS", "InternVL2Model"),
    "internvideo": (".internvideo_model", "INTERNVIDEO2_MODELS", "InternVideo2Model"),
    "internlm": (".internlm_model", "INTERNLMXCOMPOSER25_MODELS", "InternLMXComposer25Model"),
    "llama32": (".llama32_model", "LLAMA_32_VISION_MODELS", "LLaMA32VisionModel"),
    "molmo": (".molmo_model", "MOLMO_MODELS", "MOLMOVisionModel"),
    "gemini": (".gemini_model", "GEMINI_MODELS", "GeminiModel"),
    "qwen2vl": (".qwen2vl_model", "QWEN2_VL_MODELS", "Qwen2VLModel"),
    "llavavideo": (".llavavideo_model", "LLAVA_VIDEO_MODELS", "LLaVAVideoModel"),
    "tarsier": (".tarsier_model", "TARSIER_MODELS", "TarsierModel"),
    "perceptionlm": (".perceptionlm_model", "PERCEPTION_LM_MODELS", "PerceptionLMModel"),
}


def _load_entry(key):
    """Import module and return (MODELS_DICT, ModelClass)."""
    import importlib
    mod_path, models_attr, cls_attr = _MODEL_REGISTRY[key]
    mod = importlib.import_module(mod_path, package=__name__)
    return getattr(mod, models_attr), getattr(mod, cls_attr)


def list_all_vqascore_models():
    all_models = []
    for key in _MODEL_REGISTRY:
        models_dict, _ = _load_entry(key)
        all_models.extend(models_dict)
    return all_models


def get_vqascore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    for key in _MODEL_REGISTRY:
        models_dict, model_cls = _load_entry(key)
        if model_name in models_dict:
            return model_cls(model_name, device=device, cache_dir=cache_dir, **kwargs)
    raise NotImplementedError(f"Unknown VQAScore model: {model_name}")

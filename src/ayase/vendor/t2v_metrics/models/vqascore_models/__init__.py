# Lazy imports — model modules are loaded only when instantiated via get_vqascore_model().
# Model name lists are hardcoded to avoid importing heavy dependencies at module level.

from ...constants import HF_CACHE_DIR

_MODEL_NAME_TO_MODULE = {
    "clip-flant5-xxl": "clip_t5",
    "clip-flant5-xl": "clip_t5",
    "clip-flant5-xxl-no-system": "clip_t5",
    "clip-flant5-xxl-no-system-no-user": "clip_t5",
    "llava-v1.5-13b": "llava",
    "llava-v1.5-7b": "llava",
    "sharegpt4v-7b": "llava",
    "sharegpt4v-13b": "llava",
    "llava-v1.6-13b": "llava16",
    "instructblip-flant5-xxl": "instructblip",
    "instructblip-flant5-xl": "instructblip",
    "gpt-4-turbo": "gpt4v",
    "gpt-4o": "gpt4v",
    "gpt-4.1": "gpt4v",
    "llava-onevision-qwen2-7b-si": "llavaov",
    "llava-onevision-qwen2-7b-ov": "llavaov",
    "mplug-owl3-7b": "mplug",
    "paligemma-3b-mix-224": "paligemma",
    "paligemma-3b-mix-448": "paligemma",
    "paligemma-3b-mix-896": "paligemma",
    "internvl2-1b": "internvl",
    "internvl2-2b": "internvl",
    "internvl2-4b": "internvl",
    "internvl2-8b": "internvl",
    "internvl2-26b": "internvl",
    "internvl2-40b": "internvl",
    "internvl2-llama3-76b": "internvl",
    "internvl2.5-1b": "internvl",
    "internvl2.5-2b": "internvl",
    "internvl2.5-4b": "internvl",
    "internvl2.5-8b": "internvl",
    "internvl2.5-26b": "internvl",
    "internvl2.5-38b": "internvl",
    "internvl2.5-78b": "internvl",
    "internvl3-8b": "internvl",
    "internvl3-14b": "internvl",
    "internvl3-78b": "internvl",
    "internvideo2-chat-8b": "internvideo",
    "internvideo2-chat-8b-hd": "internvideo",
    "internvideo2-chat-8b-internlm": "internvideo",
    "internlmxcomposer25-7b": "internlm",
    "llama-3.2-1b": "llama32",
    "llama-3.2-3b": "llama32",
    "llama-3.2-1b-instruct": "llama32",
    "llama-3.2-3b-instruct": "llama32",
    "llama-guard-3-1b": "llama32",
    "llama-3.2-11b-vision": "llama32",
    "llama-3.2-11b-vision-instruct": "llama32",
    "llama-3.2-90b-vision": "llama32",
    "llama-3.2-90b-vision-instruct": "llama32",
    "llama-guard-3-11b-vision": "llama32",
    "molmo-72b-0924": "molmo",
    "molmo-7b-d-0924": "molmo",
    "molmo-7b-o-0924": "molmo",
    "molmoe-1b-0924": "molmo",
    "gemini-1.5-pro": "gemini",
    "gemini-1.5-flash": "gemini",
    "gemini-2.5-flash": "gemini",
    "gemini-2.5-pro": "gemini",
    "qwen2-vl-2b": "qwen2vl",
    "qwen2-vl-7b": "qwen2vl",
    "qwen2-vl-72b": "qwen2vl",
    "qwen2.5-vl-3b": "qwen2vl",
    "qwen2.5-vl-7b": "qwen2vl",
    "qwen2.5-vl-32b": "qwen2vl",
    "qwen2.5-vl-72b": "qwen2vl",
    "llava-video-7b": "llavavideo",
    "llava-video-72B": "llavavideo",
    "tarsier-recap-7b": "tarsier",
    "tarsier2-7b": "tarsier",
    "perception-lm-1b": "perceptionlm",
    "perception-lm-3b": "perceptionlm",
    "perception-lm-8b": "perceptionlm",
}

_MODULE_KEY_TO_IMPORT = {
    "clip_t5": (".clip_t5_model", "CLIPT5Model"),
    "llava": (".llava_model", "LLaVAModel"),
    "llava16": (".llava16_model", "LLaVA16Model"),
    "instructblip": (".instructblip_model", "InstructBLIPModel"),
    "gpt4v": (".gpt4v_model", "GPT4VModel"),
    "llavaov": (".llavaov_model", "LLaVAOneVisionModel"),
    "mplug": (".mplug_model", "mPLUGOwl3Model"),
    "paligemma": (".paligemma_model", "PaliGemmaModel"),
    "internvl": (".internvl_model", "InternVL2Model"),
    "internvideo": (".internvideo_model", "InternVideo2Model"),
    "internlm": (".internlm_model", "InternLMXComposer25Model"),
    "llama32": (".llama32_model", "LLaMA32VisionModel"),
    "molmo": (".molmo_model", "MOLMOVisionModel"),
    "gemini": (".gemini_model", "GeminiModel"),
    "qwen2vl": (".qwen2vl_model", "Qwen2VLModel"),
    "llavavideo": (".llavavideo_model", "LLaVAVideoModel"),
    "tarsier": (".tarsier_model", "TarsierModel"),
    "perceptionlm": (".perceptionlm_model", "PerceptionLMModel"),
}


def list_all_vqascore_models():
    return list(_MODEL_NAME_TO_MODULE.keys())


def get_vqascore_model(model_name, device='cuda', cache_dir=HF_CACHE_DIR, **kwargs):
    if model_name not in _MODEL_NAME_TO_MODULE:
        raise NotImplementedError(f"Unknown VQAScore model: {model_name}")
    module_key = _MODEL_NAME_TO_MODULE[model_name]
    mod_path, cls_attr = _MODULE_KEY_TO_IMPORT[module_key]
    import importlib
    mod = importlib.import_module(mod_path, package=__name__)
    model_cls = getattr(mod, cls_attr)
    return model_cls(model_name, device=device, cache_dir=cache_dir, **kwargs)

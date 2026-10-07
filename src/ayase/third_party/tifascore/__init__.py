"""Vendored tifascore package (github.com/Yushi-Hu/tifa, official release).

Lazy attribute access keeps optional heavy dependencies (modelscope, lavis,
promptcap, openai, word2number) out of the import path; each symbol imports its
defining module on first use.
"""

_LAZY = {
    "VQAModel": ".vqa_models",
    "tifa_score_benchmark": ".tifa_score",
    "tifa_score_single": ".tifa_score",
    "get_llama2_pipeline": ".question_gen_llama2",
    "get_llama2_question_and_answers": ".question_gen_llama2",
    "filter_question_and_answers": ".question_filter",
    "UnifiedQAModel": ".unifiedqa",
    "SBERTModel": ".mc_sbert",
    "get_question_and_answers": ".question_gen",
}


def __getattr__(name):
    import importlib

    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)


__all__ = list(_LAZY)

"""Tests for the GenEval2 / Soft-TIFA module."""

import json
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ..conftest import _test_module_basics


class _Inputs(dict):
    def __init__(self):
        super().__init__(pixels="value")
        self.destination = None

    def to(self, destination):
        self.destination = destination
        return self


class _Tokenizer:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.seen = []

    def encode(self, text):
        self.seen.append(text)
        return [self.token_ids[text], 999]


class _Processor:
    def __init__(self, token_ids):
        self.tokenizer = _Tokenizer(token_ids)
        self.inputs = _Inputs()
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.inputs

    def batch_decode(self, values):
        self.decoded = values
        return ["Yes"]


class _Model:
    device = "cpu"

    def __init__(self, logits):
        self.logits = torch.tensor([logits], dtype=torch.float64)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(scores=[self.logits, torch.tensor([[999.0]])])


def _write_benchmark(path: Path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _sample(path, prompt=None):
    from ayase.models import CaptionMetadata, Sample

    caption = CaptionMetadata(text=prompt, length=len(prompt)) if prompt is not None else None
    return Sample(path=path, is_video=False, caption=caption)


def test_geneval2_basics():
    from ayase.modules.geneval2 import GenEval2Module

    _test_module_basics(GenEval2Module, "geneval2")
    assert GenEval2Module.input_type == "image"


def test_bundled_benchmark_has_canonical_identity():
    from ayase.modules.geneval2 import GenEval2Module

    module = GenEval2Module({"test_mode": True})
    assert module._benchmark_valid
    assert module._canonical_benchmark
    assert len(module._benchmark) == 800


def test_protocol_full_vocab_duplicates_and_canonical_kwargs():
    from ayase.modules._geneval2_protocol import score_question

    processor = _Processor({"Yes": 0, "yes": 0, " yes": 1, " Yes": 2})
    model = _Model([math.log(0.1), math.log(0.1), math.log(0.1), math.log(0.1), math.log(0.6)])
    score = score_question(processor, model, torch, "image.png", "Is there a cat?", "No")

    assert score == pytest.approx(0.4)
    messages, template_kwargs = processor.calls[0]
    assert messages[0]["content"] == [
        {"type": "image", "image": "image.png"},
        {"type": "text", "text": "Is there a cat? Answer in one word."},
    ]
    assert template_kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert processor.inputs.destination == "cpu"
    assert model.calls == [{
        "pixels": "value",
        "max_new_tokens": 1,
        "do_sample": False,
        "output_scores": True,
        "return_dict_in_generate": True,
    }]


def test_protocol_count_variants_and_unknown_numeric():
    from ayase.modules._geneval2_protocol import answer_variants

    assert answer_variants("How many cats?", "two") == ["two", "Two", " two", " Two", "2", " 2"]
    assert answer_variants("How many cats?", "eleven")[-2:] == ["other", " other"]
    assert answer_variants("how many cats?", "No") == ["Yes", "yes", " yes", " Yes"]


def test_aggregate_atom_scores_am_gm_and_zero():
    from ayase.modules._geneval2_protocol import aggregate_atom_scores

    assert aggregate_atom_scores([0.25, 1.0]) == pytest.approx((0.625, 0.5))
    assert aggregate_atom_scores([0.0, 1.0]) == (0.5, 0.0)


def test_bundled_author_soft_tifa_oracle_preserves_answer_protocol():
    """Execute only the Python-3.9-safe author function, not its eager model imports."""
    from ayase.modules._geneval2_protocol import answer_variants
    import ayase.modules.geneval2 as geneval2

    source_path = (
        Path(geneval2.__file__).resolve().parent.parent
        / "third_party" / "geneval2" / "evaluation.py.source"
    )
    source = source_path.read_text(encoding="utf-8")
    function_source = "def soft_tifa" + source.split("def soft_tifa", 1)[1].split("\ndef main", 1)[0]
    calls = []

    def author_send(prompt, image_path, answer_list=None):
        calls.append((prompt, image_path, answer_list))
        return "ignored", 0.25 if len(calls) == 1 else 1.0

    namespace = {
        "send_message_with_image": author_send,
        "return_numeric_string": lambda value: {"two": "2"}.get(value, "other"),
    }
    exec(compile(function_source, str(source_path), "exec"), namespace)
    score, atoms = namespace["soft_tifa"](
        [["How many cats?", "two"], ["Is it blue?", "No"]], "image.png"
    )

    assert (score, atoms) == (0.625, [0.25, 1.0])
    assert calls == [
        ("How many cats? Answer in one word.", "image.png", answer_variants("How many cats?", "two")),
        ("Is it blue? Answer in one word.", "image.png", answer_variants("Is it blue?", "No")),
    ]


def test_protocol_matches_bundled_author_probability_extraction():
    from ayase.modules._geneval2_protocol import score_question
    import ayase.modules.geneval2 as geneval2

    source_path = (
        Path(geneval2.__file__).resolve().parent.parent
        / "third_party" / "geneval2" / "evaluation.py.source"
    )
    source = source_path.read_text(encoding="utf-8")
    construct = "def construct_message_with_image" + source.split(
        "def construct_message_with_image", 1
    )[1].split("\ndef send_message_with_image", 1)[0]
    send = "def send_message_with_image" + source.split(
        "def send_message_with_image", 1
    )[1].split("\ndef vqa_score", 1)[0]
    token_ids = {"Yes": 0, "yes": 0, " yes": 1, " Yes": 2}
    logits = [math.log(0.1), math.log(0.1), math.log(0.1), math.log(0.1), math.log(0.6)]
    author_processor = _Processor(token_ids)
    author_model = _Model(logits)
    namespace = {
        "torch": torch,
        "qwen_processor": author_processor,
        "qwen_model": author_model,
    }
    exec(compile(construct + "\n" + send, str(source_path), "exec"), namespace)
    _, author_probability = namespace["send_message_with_image"](
        "Is there a cat? Answer in one word.",
        "image.png",
        answer_list=["Yes", "yes", " yes", " Yes"],
    )

    wrapper_processor = _Processor(token_ids)
    wrapper_model = _Model(logits)
    wrapper_probability = score_question(
        wrapper_processor, wrapper_model, torch, "image.png", "Is there a cat?", "No"
    )

    assert author_probability == pytest.approx(0.4)
    assert wrapper_probability == pytest.approx(author_probability)
    assert author_processor.tokenizer.seen == wrapper_processor.tokenizer.seen
    assert author_processor.calls == wrapper_processor.calls
    assert author_model.calls == wrapper_model.calls


def test_process_noncanonical_prompt_records_custom_trace(monkeypatch, tmp_path):
    from ayase.modules.geneval2 import GenEval2Module
    import ayase.modules.geneval2 as geneval2

    benchmark = tmp_path / "benchmark.jsonl"
    record = {"prompt": "a cat", "atom_count": 1, "skills": ["object"], "vqa_list": [["A cat?", "Yes"]]}
    _write_benchmark(benchmark, [record])
    module = GenEval2Module({"benchmark_path": str(benchmark)})
    module._backend = "qwen3_vl"
    module._processor = module._model = module._torch = object()
    monkeypatch.setattr(geneval2, "score_question", lambda *args: 0.25)
    sample = _sample(tmp_path / "image.png", "a cat")

    assert module.process(sample) is sample
    assert sample.quality_metrics.soft_tifa_am_score == 0.25
    assert sample.quality_metrics.soft_tifa_gm_score == 0.25
    assert sample.metadata["geneval2"] == {
        "prompt": "a cat", "identity": "custom", "matched_benchmark": False,
        "atom_count": 1, "skills": ["object"], "vqa_list": [["A cat?", "Yes"]],
        "atom_probabilities": [0.25],
    }


def test_unknown_caption_and_unsupported_custom_no_are_unscored(tmp_path):
    from ayase.modules.geneval2 import GenEval2Module

    benchmark = tmp_path / "benchmark.jsonl"
    _write_benchmark(benchmark, [{"prompt": "known", "atom_count": 1, "skills": ["object"], "vqa_list": [["Known?", "Yes"]]}])
    module = GenEval2Module({"benchmark_path": str(benchmark)})
    module._backend = "qwen3_vl"
    for metadata in ({}, {"geneval2_vqa_list": [["Is absent?", "No"]]}):
        sample = _sample(tmp_path / "image.png", "unknown")
        sample.metadata.update(metadata)
        assert module.process(sample) is sample
        assert sample.quality_metrics is None


def test_explicit_custom_questions_override_official_prompt(monkeypatch, tmp_path):
    from ayase.modules.geneval2 import GenEval2Module
    import ayase.modules.geneval2 as geneval2

    module = GenEval2Module()
    prompt = module._benchmark_order[0]
    module._backend = "qwen3_vl"
    module._processor = module._model = module._torch = object()
    monkeypatch.setattr(geneval2, "score_question", lambda *args: 0.75)
    sample = _sample(tmp_path / "image.png", prompt)
    sample.metadata["geneval2_vqa_list"] = [["Is this custom?", "Yes"]]

    module.process(sample)
    assert sample.metadata["geneval2"]["identity"] == "custom"
    assert sample.metadata["geneval2"]["matched_benchmark"] is False
    assert sample.metadata["geneval2"]["vqa_list"] == [["Is this custom?", "Yes"]]

    module.pipeline = _Pipeline()
    module.post_process([sample])
    assert module.pipeline.values == {
        "geneval2_am_score": None,
        "geneval2_gm_score": None,
        "geneval2_coverage": 0.0,
    }


class _Pipeline:
    def __init__(self):
        self.values = {}

    def add_dataset_metric(self, name, value):
        self.values[name] = value


def _canonical_scored_samples(module, tmp_path, am=0.5, gm=0.25):
    from ayase.models import QualityMetrics

    samples = []
    for index, prompt in enumerate(module._benchmark_order):
        sample = _sample(tmp_path / f"official-{index}.png", prompt)
        sample.quality_metrics = QualityMetrics(
            soft_tifa_am_score=am, soft_tifa_gm_score=gm
        )
        sample.metadata["geneval2"] = {"identity": "benchmark", "prompt": prompt}
        samples.append(sample)
    return samples


def test_dataset_equal_prompt_weighting_complete_and_duplicate_rules(tmp_path):
    from ayase.modules.geneval2 import GenEval2Module
    from ayase.models import QualityMetrics

    benchmark = tmp_path / "benchmark.jsonl"
    records = [
        {"prompt": "p1", "atom_count": 2, "skills": ["object", "attribute"], "vqa_list": [["a", "Yes"], ["b", "Yes"]]},
        {"prompt": "p2", "atom_count": 1, "skills": ["object"], "vqa_list": [["c", "Yes"]]},
    ]
    _write_benchmark(benchmark, records)
    module = GenEval2Module({"benchmark_path": str(benchmark)})
    module._canonical_benchmark = True  # isolate official coverage mathematics
    module.pipeline = _Pipeline()
    samples = []
    for prompt, am, gm in [("p1", 0.625, 0.5), ("p2", 1.0, 1.0)]:
        sample = _sample(tmp_path / (prompt + ".png"), prompt)
        sample.quality_metrics = QualityMetrics(soft_tifa_am_score=am, soft_tifa_gm_score=gm)
        sample.metadata["geneval2"] = {"identity": "benchmark", "prompt": prompt}
        samples.append(sample)
    module.post_process(samples)
    assert module.pipeline.values == {
        "geneval2_coverage": 1.0, "geneval2_am_score": 81.25, "geneval2_gm_score": 75.0
    }

    module.pipeline = _Pipeline()
    module.post_process(samples + [samples[0].model_copy(deep=True)])
    assert module.pipeline.values == {
        "geneval2_am_score": None,
        "geneval2_gm_score": None,
        "geneval2_coverage": 1.0,
    }


def test_dataset_partial_reports_coverage_only(tmp_path):
    from ayase.modules.geneval2 import GenEval2Module
    from ayase.models import QualityMetrics

    benchmark = tmp_path / "benchmark.jsonl"
    _write_benchmark(benchmark, [
        {"prompt": "p1", "atom_count": 1, "skills": ["object"], "vqa_list": [["a", "Yes"]]},
        {"prompt": "p2", "atom_count": 1, "skills": ["object"], "vqa_list": [["b", "Yes"]]},
    ])
    module = GenEval2Module({"benchmark_path": str(benchmark)})
    module._canonical_benchmark = True  # isolate official coverage mathematics
    module.pipeline = _Pipeline()
    sample = _sample(tmp_path / "p1.png", "p1")
    sample.quality_metrics = QualityMetrics(soft_tifa_am_score=0.5, soft_tifa_gm_score=0.5)
    sample.metadata["geneval2"] = {"identity": "benchmark", "prompt": "p1"}
    module.post_process([sample])
    assert module.pipeline.values == {
        "geneval2_am_score": None,
        "geneval2_gm_score": None,
        "geneval2_coverage": 0.5,
    }


def test_noncanonical_benchmark_never_emits_official_dataset_metrics(tmp_path):
    from ayase.modules.geneval2 import GenEval2Module

    benchmark = tmp_path / "benchmark.jsonl"
    _write_benchmark(benchmark, [
        {"prompt": "p1", "atom_count": 1, "skills": ["object"], "vqa_list": [["a", "Yes"]]}
    ])
    module = GenEval2Module({"benchmark_path": str(benchmark)})
    module.pipeline = _Pipeline()
    module.post_process([])
    assert not module._canonical_benchmark
    assert module.pipeline.values == {}


def test_failed_explicit_custom_input_blocks_complete_official_aggregate(tmp_path):
    from ayase.modules.geneval2 import GenEval2Module

    module = GenEval2Module()
    module.pipeline = _Pipeline()
    samples = _canonical_scored_samples(module, tmp_path)
    failed_custom = _sample(tmp_path / "failed-custom.png", "unrelated")
    failed_custom.metadata["geneval2_vqa_list"] = [["Unsupported?", "No"]]

    module.post_process(samples + [failed_custom])

    assert module.pipeline.values == {
        "geneval2_am_score": None,
        "geneval2_gm_score": None,
        "geneval2_coverage": 1.0,
    }


def test_post_process_clears_stale_aggregate_on_partial_rerun(tmp_path):
    from ayase.modules.geneval2 import GenEval2Module

    module = GenEval2Module()
    module.pipeline = _Pipeline()
    samples = _canonical_scored_samples(module, tmp_path)
    module.post_process(samples)
    assert module.pipeline.values["geneval2_am_score"] == 50.0
    assert module.pipeline.values["geneval2_gm_score"] == 25.0

    module.post_process(samples[:-1])
    assert module.pipeline.values == {
        "geneval2_am_score": None,
        "geneval2_gm_score": None,
        "geneval2_coverage": 799 / 800,
    }


def test_unscored_duplicate_official_input_blocks_full_aggregate(tmp_path):
    from ayase.modules.geneval2 import GenEval2Module

    module = GenEval2Module()
    module.pipeline = _Pipeline()
    samples = _canonical_scored_samples(module, tmp_path)
    duplicate = _sample(tmp_path / "duplicate.png", module._benchmark_order[0])
    module.post_process(samples + [duplicate])

    assert module.pipeline.values == {
        "geneval2_am_score": None,
        "geneval2_gm_score": None,
        "geneval2_coverage": 1.0,
    }


def test_video_with_official_caption_or_custom_questions_is_ignored(tmp_path):
    from ayase.models import CaptionMetadata, Sample
    from ayase.modules.geneval2 import GenEval2Module

    module = GenEval2Module()
    module.pipeline = _Pipeline()
    samples = _canonical_scored_samples(module, tmp_path)
    prompt = module._benchmark_order[0]
    video = Sample(
        path=tmp_path / "extra.mp4",
        is_video=True,
        caption=CaptionMetadata(text=prompt, length=len(prompt)),
        metadata={"geneval2_vqa_list": [["Unsupported video override?", "No"]]},
    )

    module.post_process(samples + [video])

    assert module.pipeline.values == {
        "geneval2_am_score": 50.0,
        "geneval2_gm_score": 25.0,
        "geneval2_coverage": 1.0,
    }


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -0.01])
def test_post_process_rejects_invalid_restored_scores(tmp_path, bad_value):
    from ayase.modules.geneval2 import GenEval2Module

    module = GenEval2Module()
    module.pipeline = _Pipeline()
    samples = _canonical_scored_samples(module, tmp_path)
    samples[0].quality_metrics.soft_tifa_am_score = bad_value
    module.post_process(samples)

    assert module.pipeline.values["geneval2_am_score"] is None
    assert module.pipeline.values["geneval2_gm_score"] is None
    assert module.pipeline.values["geneval2_coverage"] == 799 / 800


def test_invalid_benchmark_is_atomic_and_setup_does_not_download(monkeypatch, tmp_path):
    from ayase.modules.geneval2 import GenEval2Module
    import ayase.config

    benchmark = tmp_path / "invalid.jsonl"
    benchmark.write_text(
        json.dumps({"prompt": "valid", "vqa_list": [["a", "Yes"]]}) + "\nnot-json\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(ayase.config, "download_hf_snapshot", lambda *a, **k: pytest.fail("download"))
    monkeypatch.setattr(GenEval2Module, "_global_test_mode", False)
    module = GenEval2Module({"benchmark_path": str(benchmark)})
    assert module._benchmark == {}
    assert module._benchmark_order == []
    assert not module._benchmark_valid
    assert module.config["_geneval2_benchmark_sha256"] is None
    module.setup()
    assert module._backend == "unavailable"


def test_same_path_benchmark_edit_changes_pipeline_fingerprint(tmp_path):
    from ayase.modules.geneval2 import GenEval2Module
    from ayase.pipeline import Pipeline

    benchmark = tmp_path / "benchmark.jsonl"
    first = {
        "prompt": "p",
        "atom_count": 1,
        "skills": ["object"],
        "vqa_list": [["Is the first question true?", "Yes"]],
    }
    second = {
        "prompt": "p",
        "atom_count": 1,
        "skills": ["object"],
        "vqa_list": [["Is the changed question true?", "Yes"]],
    }
    _write_benchmark(benchmark, [first])
    first_module = GenEval2Module({"benchmark_path": str(benchmark), "test_mode": True})
    first_fingerprint = Pipeline([first_module])._pipeline_fingerprint()

    _write_benchmark(benchmark, [second])
    second_module = GenEval2Module({"benchmark_path": str(benchmark), "test_mode": True})
    second_fingerprint = Pipeline([second_module])._pipeline_fingerprint()

    assert first_module.config["benchmark_path"] == second_module.config["benchmark_path"]
    assert (
        first_module.config["_geneval2_benchmark_sha256"]
        != second_module.config["_geneval2_benchmark_sha256"]
    )
    assert first_fingerprint != second_fingerprint


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01])
def test_process_rejects_invalid_probability(monkeypatch, tmp_path, value):
    from ayase.modules.geneval2 import GenEval2Module
    import ayase.modules.geneval2 as geneval2

    benchmark = tmp_path / "benchmark.jsonl"
    _write_benchmark(benchmark, [
        {"prompt": "p", "atom_count": 1, "skills": ["object"], "vqa_list": [["a", "Yes"]]}
    ])
    module = GenEval2Module({"benchmark_path": str(benchmark)})
    module._backend = "qwen3_vl"
    module._processor = module._model = module._torch = object()
    monkeypatch.setattr(geneval2, "score_question", lambda *args: value)
    sample = _sample(tmp_path / "image.png", "p")
    assert module.process(sample) is sample
    assert sample.quality_metrics is None
    assert "geneval2" not in sample.metadata


def test_process_skips_video(monkeypatch, video_sample):
    from ayase.modules.geneval2 import GenEval2Module
    import ayase.modules.geneval2 as geneval2

    module = GenEval2Module()
    module._backend = "qwen3_vl"
    monkeypatch.setattr(geneval2, "score_question", lambda *args: pytest.fail("inference"))
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None


def test_setup_test_mode_skips_download(monkeypatch):
    from ayase.modules.geneval2 import GenEval2Module
    import ayase.config

    monkeypatch.setattr(ayase.config, "download_hf_snapshot", lambda *a, **k: pytest.fail("download"))
    module = GenEval2Module({"test_mode": True})
    module.setup()
    assert module._backend is None


def test_setup_uses_pinned_snapshot(monkeypatch, tmp_path):
    from ayase.modules.geneval2 import GenEval2Module, MODEL_ID, MODEL_REVISION
    import ayase.config

    calls = []
    root = tmp_path / "model"
    class Factory:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((cls.__name__, path, kwargs))
            return cls()
    class AutoProcessor(Factory):
        pass
    class Qwen3VLForConditionalGeneration(Factory):
        def eval(self):
            return self
    fake = types.ModuleType("transformers")
    fake.AutoProcessor = AutoProcessor
    fake.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(ayase.config, "download_hf_snapshot", lambda *a, **k: (calls.append(("download", a, k)), root)[1])
    monkeypatch.setattr(GenEval2Module, "_global_test_mode", False)
    module = GenEval2Module({"models_dir": str(tmp_path / "models")})
    module.setup()
    assert calls[0] == ("download", (MODEL_ID, str(tmp_path / "models")), {"revision": MODEL_REVISION})
    assert calls[1:] == [
        ("AutoProcessor", str(root), {}),
        ("Qwen3VLForConditionalGeneration", str(root), {"dtype": "auto", "device_map": "auto"}),
    ]
    assert module._backend == "qwen3_vl"


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"device": "cpu", "dtype": "float32"}, {"dtype": "float32", "device_map": "cpu"}),
        (
            {"device": "cpu", "dtype": "float32", "device_map": "cuda:1"},
            {"dtype": "float32", "device_map": "cuda:1"},
        ),
    ],
)
def test_setup_forwards_runtime_device_and_dtype(monkeypatch, tmp_path, config, expected):
    from ayase.modules.geneval2 import GenEval2Module
    import ayase.config

    calls = []
    root = tmp_path / "model"

    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            return cls()

    class Qwen3VLForConditionalGeneration:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append(kwargs)
            return cls()

        def eval(self):
            return self

    fake = types.ModuleType("transformers")
    fake.AutoProcessor = AutoProcessor
    fake.Qwen3VLForConditionalGeneration = Qwen3VLForConditionalGeneration
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(ayase.config, "download_hf_snapshot", lambda *a, **k: root)
    monkeypatch.setattr(GenEval2Module, "_global_test_mode", False)
    module = GenEval2Module({"models_dir": str(tmp_path / "models"), **config})

    module.setup()

    assert calls == [expected]
    assert module._backend == "qwen3_vl"


def test_setup_failure_is_visible_unavailable(monkeypatch):
    from ayase.modules.geneval2 import GenEval2Module
    import ayase.config

    monkeypatch.setattr(ayase.config, "download_hf_snapshot", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    monkeypatch.setattr(GenEval2Module, "_global_test_mode", False)
    module = GenEval2Module()
    module.setup()
    assert module._backend == "unavailable"
    assert module._model is None

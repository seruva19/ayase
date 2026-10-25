import hashlib
import sys
import types
from pathlib import Path

from ayase.models import CaptionMetadata, Sample


def test_speechjudge_basics():
    from tests.modules.conftest import _test_module_basics
    from ayase.modules.speechjudge import SpeechJudgeModule

    _test_module_basics(SpeechJudgeModule, "speechjudge")


def test_extract_ratings_uses_last_pair_and_numeric_comparison():
    from ayase.modules.speechjudge import extract_ratings

    result = extract_ratings("Output A: 4, Output B: 8\n**Output A: 10, Output B: 9**")
    assert result == (10.0, 9.0)
    assert result[0] > result[1]


def test_extract_ratings_rejects_out_of_range_and_malformed():
    from ayase.modules.speechjudge import extract_ratings

    assert extract_ratings("Output A: 11, Output B: 9") is None
    assert extract_ratings("Output B: 8, Output A: 7") is None
    assert extract_ratings("Output A: nan, Output B: 7") is None


def test_inputs_require_audio_comparator_and_text(tmp_path):
    from ayase.modules.speechjudge import SpeechJudgeModule

    candidate = tmp_path / "candidate.wav"
    comparator = tmp_path / "reference.wav"
    candidate.write_bytes(b"audio-a")
    comparator.write_bytes(b"audio-b")
    module = SpeechJudgeModule({"test_mode": True})
    sample = Sample(path=candidate, is_video=False, reference_path=comparator)
    assert module._inputs(sample) is None
    sample.caption = CaptionMetadata(text="hello", length=5)
    assert module._inputs(sample) == (candidate, comparator, "hello")
    sample.is_video = True
    assert module._inputs(sample) is None


def test_protocol_ast_reader_preserves_a_b_order_without_execution(tmp_path, monkeypatch):
    import ayase.modules.speechjudge as module

    marker = tmp_path / "executed"
    source = tmp_path / "utils.py"
    source.write_text(
        "def build_cot_conversation(target_text, wav_path_a, wav_path_b):\n"
        "    return [\n"
        "      {'role':'system','content':[{'type':'text','text':'system'}]},\n"
        "      {'role':'user','content':[\n"
        "       {'type':'text','text':'compare'},\n"
        "       {'type':'text','text':f'Target text: {target_text}'},\n"
        "       {'type':'text','text':'Output A:'},{'type':'audio','audio':wav_path_a},\n"
        "       {'type':'text','text':'Output B:'},{'type':'audio','audio':wav_path_b},\n"
        "       {'type':'text','text':'score'},{'type':'text','text':'format'}]}]\n"
        f"open({str(marker)!r}, 'w').write('bad')\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROTOCOL_SHA256", hashlib.sha256(source.read_bytes()).hexdigest())
    conversation = module.load_protocol_conversation(
        source, "spoken words", Path("a.wav"), Path("b.wav")
    )
    assert conversation[1]["content"][1]["text"] == "Target text: spoken words"
    assert conversation[1]["content"][3]["audio"] == "a.wav"
    assert conversation[1]["content"][5]["audio"] == "b.wav"
    assert not marker.exists()


def test_test_mode_never_acquires_assets(tmp_path, monkeypatch):
    import ayase.modules.speechjudge as module

    candidate = tmp_path / "candidate.wav"
    comparator = tmp_path / "reference.wav"
    candidate.write_bytes(b"a")
    comparator.write_bytes(b"b")
    judge = module.SpeechJudgeModule({"test_mode": True, "target_text": "hello"})
    monkeypatch.setattr(judge, "_prepare_backend", lambda: (_ for _ in ()).throw(AssertionError()))
    sample = Sample(path=candidate, is_video=False, reference_path=comparator)
    assert judge.process(sample) is sample
    assert sample.quality_metrics is None


def test_scaling_mean_uses_only_valid_generations(tmp_path, monkeypatch):
    import ayase.modules.speechjudge as module
    from ayase.pipeline import PipelineModule

    monkeypatch.setattr(PipelineModule, "_global_test_mode", False)

    candidate = tmp_path / "candidate.wav"
    comparator = tmp_path / "reference.wav"
    protocol = tmp_path / "protocol.py"
    candidate.write_bytes(b"a")
    comparator.write_bytes(b"b")
    protocol.write_bytes(b"unused")
    judge = module.SpeechJudgeModule({
        "protocol": "scaling_10", "target_text": "hello", "protocol_path": protocol,
    })
    judge._protocol_file = protocol
    judge._backend = "speechjudge-test"
    judge._model = object()
    judge._processor = object()
    responses = iter([
        "Output A: 10, Output B: 9",
        "invalid",
        "Output A: 8, Output B: 4",
    ] + ["bad"] * 7)
    monkeypatch.setattr(module, "load_protocol_conversation", lambda *args: [])
    monkeypatch.setattr(judge, "_generate", lambda conversation: next(responses))
    sample = Sample(path=candidate, is_video=False, reference_path=comparator)
    result = judge.process(sample)
    assert result is sample
    assert sample.quality_metrics.speechjudge_candidate_score == 9.0
    assert sample.quality_metrics.speechjudge_comparator_score == 6.5
    assert sample.quality_metrics.speechjudge_margin == 2.5
    assert sample.quality_metrics.speechjudge_preference == 1.0
    assert sample.metadata["speechjudge"]["valid_generations"] == 2
    assert sample.metadata["speechjudge"]["aggregation"] == "released_code_mean_of_valid_scores"


def test_all_invalid_generations_leave_metrics_unset(tmp_path, monkeypatch):
    import ayase.modules.speechjudge as module
    from ayase.pipeline import PipelineModule

    monkeypatch.setattr(PipelineModule, "_global_test_mode", False)

    candidate = tmp_path / "candidate.wav"
    comparator = tmp_path / "reference.wav"
    protocol = tmp_path / "protocol.py"
    candidate.write_bytes(b"a")
    comparator.write_bytes(b"b")
    protocol.write_bytes(b"unused")
    judge = module.SpeechJudgeModule({"target_text": "hello", "protocol_path": protocol})
    judge._protocol_file = protocol
    judge._backend = "speechjudge-test"
    judge._model = object()
    judge._processor = object()
    monkeypatch.setattr(module, "load_protocol_conversation", lambda *args: [])
    monkeypatch.setattr(judge, "_generate", lambda conversation: "no rating")
    sample = Sample(path=candidate, is_video=False, reference_path=comparator)
    judge.process(sample)
    assert sample.quality_metrics is None
    assert sample.metadata["speechjudge"]["valid_generations"] == 0


def test_generate_accepts_tuple_and_decodes_only_suffix(monkeypatch):
    import torch
    from ayase.modules.speechjudge import SpeechJudgeModule

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            return Inputs(input_ids=torch.tensor([[1, 2]]))

        def batch_decode(self, tokens, **kwargs):
            assert tokens.tolist() == [[8, 9]]
            return ["Output A: 10, Output B: 9"]

    class Model:
        device = torch.device("cpu")

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, **kwargs):
            assert kwargs["max_new_tokens"] == 1024
            assert kwargs["eos_token_id"] == [151645]
            return torch.tensor([[1, 2, 8, 9]]), None

    fake_utils = types.SimpleNamespace(
        process_mm_info=lambda *args, **kwargs: (None, None, None)
    )
    monkeypatch.setitem(sys.modules, "qwen_omni_utils", fake_utils)
    judge = SpeechJudgeModule()
    judge._processor = Processor()
    judge._model = Model()
    assert judge._generate([]) == "Output A: 10, Output B: 9"


def test_setup_marks_lazy_backend_ready_without_acquiring(monkeypatch):
    from ayase.modules.speechjudge import SpeechJudgeModule
    from ayase.pipeline import PipelineModule

    monkeypatch.setattr(PipelineModule, "_global_test_mode", False)
    judge = SpeechJudgeModule()
    monkeypatch.setattr(judge, "_prepare_backend", lambda: (_ for _ in ()).throw(AssertionError()))
    judge.setup()
    assert judge._backend == "deferred"
    assert judge.config["managed_operation_timeout"] == 7200


def test_process_before_mount_does_not_resolve_or_download(tmp_path, monkeypatch):
    from ayase.modules.speechjudge import SpeechJudgeModule

    candidate = tmp_path / "candidate.wav"
    candidate.write_bytes(b"a")
    judge = SpeechJudgeModule({"target_text": "hello"})
    monkeypatch.setattr(judge, "_inputs", lambda sample: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(judge, "_prepare_backend", lambda: (_ for _ in ()).throw(AssertionError()))
    sample = Sample(path=candidate, is_video=False)
    assert judge.process(sample) is sample
    assert judge._backend is None


def test_loading_info_allows_only_unused_speech_generation_prefixes():
    from ayase.modules.speechjudge import _loading_problems

    clean = {
        "unexpected_keys": ["talker.layer.weight", "token2wav.head.weight"],
        "missing_keys": [], "mismatched_keys": [], "error_msgs": [],
    }
    assert _loading_problems(clean) == ([], [], [], [])
    problems = _loading_problems({
        "unexpected_keys": ["thinker.bad"], "missing_keys": ["thinker.missing"],
        "mismatched_keys": [("thinker.shape", (1,), (2,))], "error_msgs": ["broken"],
    })
    assert all(problems)

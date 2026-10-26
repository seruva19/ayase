"""Focused protocol tests for the source-backed AVGen remote adapters."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


def _load_adapter():
    path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "ayase"
        / "third_party"
        / "avgen_bench"
        / "adapters_remote.py"
    )
    spec = importlib.util.spec_from_file_location("avgen_adapters_remote_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def adapter():
    return _load_adapter()


def _request(tmp_path: Path, dimension: str, prompt: str = "Café 猫 says: ‘hello’"):
    source_dir = tmp_path / "source checkout 日本語"
    source_dir.mkdir()
    work_dir = tmp_path / "work area тест"
    output_dir = tmp_path / "output area 漢字"
    staged = tmp_path / "staged videos" / "music tutorial" / "safe name.mp4"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"synthetic-video")
    original = tmp_path / "original videos" / "original имя.mp4"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"original-video")
    det = tmp_path / "models" / "ocr det"
    rec = tmp_path / "models" / "ocr rec"
    whisper = tmp_path / "models" / "whisper large-v3"
    det.mkdir(parents=True)
    rec.mkdir(parents=True)
    whisper.mkdir(parents=True)
    basic_pitch = tmp_path / "models" / "basic pitch" / "nmp.onnx"
    basic_pitch.parent.mkdir(parents=True)
    basic_pitch.write_bytes(b"onnx")
    ffmpeg = tmp_path / "native tools" / "ffmpeg.exe"
    ffmpeg.parent.mkdir(parents=True)
    ffmpeg.write_bytes(b"exe")
    return {
        "schema": 1,
        "dimension": dimension,
        "source_dir": str(source_dir.resolve()),
        "models_dir": str((tmp_path / "models").resolve()),
        "work_dir": str(work_dir.resolve()),
        "videos_dir": str(staged.parents[1].resolve()),
        "prompts_dir": str((tmp_path / "prompts").resolve()),
        "face_prompts_dir": str((tmp_path / "faces").resolve()),
        "output_dir": str(output_dir.resolve()),
        "run_tag": "test-run",
        "assets": {
            "ocr_det_dir": str(det.resolve()),
            "ocr_rec_dir": str(rec.resolve()),
            "basic_pitch_model": str(basic_pitch.resolve()),
            "whisper_model_dir": str(whisper.resolve()),
            "ffmpeg_path": str(ffmpeg.resolve()),
        },
        "config": {"workers": 2, "timeout_s": 17},
        "records": [
            {
                "id": "record-é-猫",
                "video_path": str(original.resolve()),
                "staged_video_path": str(staged.resolve()),
                "safe_content": "safe name",
                "category": "musical_instrument_tutorial",
                "content": "Original title не трогать",
                "prompt_text": prompt,
                "expected_faces": None,
                "mode": "canonical",
            }
        ],
    }


def test_load_source_module_handles_spaces_unicode_and_synthetic_io(adapter, tmp_path):
    source = tmp_path / "source with spaces 日本語"
    module_path = source / "eval" / "tiny.py"
    module_path.parent.mkdir(parents=True)
    module_path.write_text(
        "from pathlib import Path\n"
        "def parse(path):\n"
        "    return Path(path).read_text(encoding='utf-8')\n",
        encoding="utf-8",
    )
    payload = tmp_path / "入力 file.txt"
    payload.write_text("привет 猫", encoding="utf-8")

    loaded = adapter._load_source_module(source, "eval/tiny.py", "Tiny Route")

    assert loaded.parse(payload) == "привет 猫"
    assert not (module_path.parent / "__pycache__").exists()


def test_text_loader_injects_pinned_det_and_rec_before_source_import(
    adapter, tmp_path, monkeypatch
):
    observed = {}

    def original_paddle_ocr(*args, **kwargs):
        observed.update(kwargs)
        return object()

    paddleocr = SimpleNamespace(PaddleOCR=original_paddle_ocr)
    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: paddleocr)

    def load_source(*args, **kwargs):
        paddleocr.PaddleOCR(device="cpu")
        return SimpleNamespace()

    monkeypatch.setattr(adapter, "_load_source_module", load_source)
    det = tmp_path / "det"
    rec = tmp_path / "rec"

    adapter._load_text_module(tmp_path, det, rec)

    assert observed["text_detection_model_dir"] == str(det)
    assert observed["text_recognition_model_dir"] == str(rec)
    assert paddleocr.PaddleOCR is original_paddle_ocr


def _valid_raw_payloads():
    empty_audit = {
        key: []
        for key in (
            "kinematics",
            "gravity",
            "dynamics",
            "contact",
            "occlusion",
            "identity",
            "fluids_fire",
            "imaging",
        )
    }
    return {
        "text.judge": {
            "prompt_requires_visible_text": True,
            "text_presence": "required",
            "missing_required_text": False,
            "incidental_text_is_contextual": None,
            "overall_text_quality_score": 87,
            "subscores": {
                "legibility_accuracy": 81,
                "temporal_stability": 82,
                "spatial_stability": 83,
                "completeness": 84,
                "prompt_text_match": 85,
            },
            "confidence": 0.9,
            "top_issues": [],
            "good_tracks": ["T1"],
            "summary": "readable",
        },
        "music.constraints": {
            "should_midi_eval": True,
            "skip_reason": None,
            "constraints": {"expected_chord": "C"},
        },
        "music.judge": {
            "overall_score": 79,
            "confidence": 0.8,
            "subscores": {"chord": 79, "sequence_or_scale": None},
            "interpretation": {},
            "checks": [],
            "violations": [],
            "summary": "matches",
        },
        "speech.judge": {
            "score": 91,
            "explicit_speech_required": True,
            "match_type": "verbatim",
            "pass": True,
            "required_speech_lines": ["hello"],
            "verbatim_match_details": [],
            "missing_lines": [],
            "extra_or_mismatched_segments": [],
            "normalized_transcript": "hello",
            "score_rationale": "matched",
            "suggested_fix": "none",
        },
        "hi_phy.expectations": {
            "expectations": [{"id": "E1", "expectation": "object falls"}]
        },
        "hi_phy.semantic": {
            "semantic_physics_score": 81,
            "confidence": 0.8,
            "event_log": [],
            "checks": [],
            "major_issues": [],
        },
        "hi_phy.visual": {
            "visual_physics_score": 62,
            "confidence": 0.6,
            "cut_detected": False,
            "audit_by_category": empty_audit,
            "visual_violations": [],
            "major_issues": [],
        },
        "holistic.judge": {
            "plot_alignment_score": 86,
            "confidence": 0.9,
            "subscores": {
                "narrative_alignment": 81,
                "shot_alignment": 82,
                "visual_attribute_alignment": 83,
                "audio_alignment": 84,
            },
            "prompt_constraints": {},
            "observed_event_log": [],
            "checks": [],
            "violations": [],
            "summary": "aligned",
        },
    }


def test_all_raw_stage_schemas_accept_complete_source_prompt_shapes(adapter):
    for stage, payload in _valid_raw_payloads().items():
        adapter._RAW_VALIDATORS[stage](payload, stage)


@pytest.mark.parametrize(
    ("stage", "field"),
    [
        ("text.judge", "overall_text_quality_score"),
        ("music.constraints", "constraints"),
        ("music.judge", "overall_score"),
        ("speech.judge", "score"),
        ("hi_phy.expectations", "expectations"),
        ("hi_phy.semantic", "semantic_physics_score"),
        ("hi_phy.visual", "visual_physics_score"),
        ("holistic.judge", "plot_alignment_score"),
    ],
)
def test_raw_stage_schemas_reject_missing_fields(adapter, stage, field):
    payload = _valid_raw_payloads()[stage]
    payload.pop(field)
    with pytest.raises(adapter.RawJudgeValidationError, match="missing required field"):
        adapter._RAW_VALIDATORS[stage](payload, stage)


@pytest.mark.parametrize("value", [True, "50", 101, -1])
def test_raw_score_policy_rejects_bool_string_and_out_of_range(adapter, value):
    payload = _valid_raw_payloads()["holistic.judge"]
    payload["plot_alignment_score"] = value
    with pytest.raises(adapter.RawJudgeValidationError):
        adapter._validate_holistic_raw(payload, "holistic.judge")


def test_raw_parser_rejects_nonfinite_json_number_and_keeps_raw_text(adapter):
    raw = '{"plot_alignment_score": NaN}'
    with pytest.raises(adapter.RawJudgeValidationError) as caught:
        adapter._raw_object(raw, "holistic.judge")
    assert "finite" in str(caught.value)


def test_source_loader_binds_pinned_gemini_and_restores_generic_module(
    adapter, tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "gemini_client.py").write_text(
        "def generate_content_text(*args, **kwargs): return 'pinned'\n"
        "def resolve_api_key(*args, **kwargs): return 'pinned-key'\n"
        "def inline_file_part(*args, **kwargs): return {'pinned': True}\n",
        encoding="utf-8",
    )
    evaluator = source / "eval" / "judge.py"
    evaluator.parent.mkdir()
    evaluator.write_text(
        "from gemini_client import generate_content_text\n"
        "def call(): return generate_content_text()\n",
        encoding="utf-8",
    )
    poison = ModuleType("gemini_client")
    poison.generate_content_text = lambda *args, **kwargs: "poison"
    monkeypatch.setitem(sys.modules, "gemini_client", poison)

    loaded = adapter._load_source_module(
        source, "eval/judge.py", "isolation", bind_gemini=True
    )

    assert loaded.call() == "pinned"
    assert sys.modules["gemini_client"] is poison
    assert loaded._ayase_pinned_gemini_client_sha256


def test_text_dispatch_preserves_prompt_and_uses_full_ocr_protocol(adapter, tmp_path, monkeypatch):
    request = _request(tmp_path, "Text")
    observed = {}

    def run_ocr(video_path, out_payload_path):
        observed["video"] = video_path
        Path(out_payload_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"tracks": [{"text": "hello"}], "pipeline_meta": {"tracking": "source"}}
        Path(out_payload_path).write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def run_gemini(prompt_text, payload):
        observed["prompt"] = prompt_text
        observed["payload"] = payload
        return {
            "prompt_requires_visible_text": False,
            "text_presence": "incidental",
            "overall_text_quality_score": 91,
            "subscores": {"legibility_accuracy": 90},
            "confidence": 0.8,
        }

    fake = SimpleNamespace(
        run_ocr_only=run_ocr,
        run_gemini_only=run_gemini,
        MODEL_NAME=None,
        TIMEOUT_S=None,
        PROMPT_VARIANT=None,
        generate_content_text=lambda *args, **kwargs: "{}",
    )
    monkeypatch.setattr(adapter, "_load_text_module", lambda *args: fake)

    response = adapter.run_dimension(request)

    assert response["records"][0]["status"] == "scored"
    assert response["records"][0]["raw_score"] == 91.0
    assert response["raw_summary"]["count_eligible_for_summary"] == 1
    assert observed["prompt"] == request["records"][0]["prompt_text"]
    assert observed["video"] == request["records"][0]["staged_video_path"]
    assert observed["payload"]["pipeline_meta"]["tracking"] == "source"
    assert Path(response["records"][0]["evidence_path"]).is_file()


def test_text_non_applicable_keeps_source_score_only_as_evidence(adapter, tmp_path, monkeypatch):
    request = _request(tmp_path, "Text")
    fake = SimpleNamespace(
        run_ocr_only=lambda *_: {"tracks": []},
        run_gemini_only=lambda *_: {
            "prompt_requires_visible_text": False,
            "text_presence": "none",
            "overall_text_quality_score": 100,
            "confidence": 0.9,
        },
        MODEL_NAME=None,
        TIMEOUT_S=None,
        PROMPT_VARIANT=None,
        generate_content_text=lambda *args, **kwargs: "{}",
    )
    monkeypatch.setattr(adapter, "_load_text_module", lambda *args: fake)

    record = adapter.run_dimension(request)["records"][0]

    assert record["status"] == "ineligible"
    assert record["eligible"] is False
    assert record["raw_score"] is None
    assert record["raw_details"]["overall_text_quality_score"] == 100


def test_music_dispatch_pins_basic_pitch_and_preserves_zero_ineligible(adapter, tmp_path, monkeypatch):
    request = _request(tmp_path, "Music")
    observed = {}

    class SourceTranscriber:
        def __init__(self, temp_dir):
            self.temp_dir = temp_dir
            self.model_path = "unversioned-package-default"

    fake = SimpleNamespace(
        BasicPitchTranscriber=SourceTranscriber,
        MODEL_NAME=None,
        DEFAULT_TIMEOUT_S=None,
        PROMPT_VARIANT=None,
        TEMP_DIR=None,
        CHORD_ONSET_WINDOW_S=None,
        MAX_EVENTS_TO_SEND=None,
        generate_content_text=lambda *args, **kwargs: "{}",
    )

    def judge_video_music_raw_midi(**kwargs):
        transcriber = fake.BasicPitchTranscriber(fake.TEMP_DIR)
        observed.update(kwargs)
        observed["model_path"] = transcriber.model_path
        return {
            "overall_score": 0,
            "confidence": 0.0,
            "skipped": True,
            "skip_reason": "no explicit pitch constraints",
            "constraints": {},
            "_artifacts": {
                "raw_events_count": 0,
                "chord_frames_count": 0,
                "chord_onset_window_s": fake.CHORD_ONSET_WINDOW_S,
                "max_events_sent": fake.MAX_EVENTS_TO_SEND,
            },
        }

    fake.judge_video_music_raw_midi = judge_video_music_raw_midi
    monkeypatch.setattr(adapter, "_load_source_module", lambda *args, **kwargs: fake)

    response = adapter.run_dimension(request)
    record = response["records"][0]

    assert record["status"] == "ineligible"
    assert record["raw_score"] is None
    assert record["raw_details"]["overall_score"] == 0
    assert response["raw_summary"]["mean_score"] == 0.0
    assert response["raw_summary"]["num_midi_evaluable"] == 0
    assert observed["prompt_text"] == request["records"][0]["prompt_text"]
    assert observed["model_path"] == request["assets"]["basic_pitch_model"]
    assert fake.CHORD_ONSET_WINDOW_S == 0.08
    assert fake.MAX_EVENTS_TO_SEND == 600


def test_music_canonical_selection_does_not_call_source_for_other_category(
    adapter, tmp_path, monkeypatch
):
    request = _request(tmp_path, "Music")
    request["records"][0]["category"] = "ads"
    monkeypatch.setattr(
        adapter,
        "_load_source_module",
        lambda *args: pytest.fail("non-music canonical records must not load the backend"),
    )

    response = adapter.run_dimension(request)

    assert response["records"][0]["status"] == "ineligible"
    assert response["records"][0]["eligible"] is False
    assert response["raw_summary"]["adapter_source_filtered"] == 1


def test_speech_dispatch_uses_actual_launcher_profile(adapter, tmp_path, monkeypatch):
    request = _request(tmp_path, "Speech")
    request["config"].update({"device": "cpu", "compute_type": "float16"})
    observed = {}

    class WhisperModel:
        def __init__(self, model_path, **kwargs):
            observed["model"] = model_path
            observed["constructor"] = kwargs

    def transcribe(model, video_path, **kwargs):
        observed["video"] = video_path
        observed["transcribe"] = kwargs
        return {
            "text": "hello exactly",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello exactly"}],
            "language": "en",
            "language_probability": 0.99,
            "duration": 1.0,
        }

    def evaluate(**kwargs):
        observed["judge"] = kwargs
        return {
            "score": 84,
            "pass": True,
            "explicit_speech_required": True,
            "match_type": "verbatim",
        }

    fake = SimpleNamespace(
        WhisperModel=WhisperModel,
        transcribe_with_faster_whisper=transcribe,
        evaluate_speech_with_gemini=evaluate,
        resolve_api_key=lambda required=False: "test-key",
        torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)),
        PROMPT_VARIANT=None,
        generate_content_text=lambda *args, **kwargs: "{}",
    )
    monkeypatch.setattr(adapter, "_load_source_module", lambda *args, **kwargs: fake)

    response = adapter.run_dimension(request)

    assert response["records"][0]["raw_score"] == 84.0
    assert observed["constructor"] == {"device": "cpu", "compute_type": "int8"}
    assert observed["transcribe"] == {
        "language": "en",
        "beam_size": 1,
        "temperature": 0.0,
        "vad_filter": True,
        "vad_min_silence_ms": 300,
        "vad_speech_pad_ms": 500,
    }
    assert observed["judge"]["generation_prompt"] == request["records"][0]["prompt_text"]
    assert observed["judge"]["model_name"] == "gemini-3-flash-preview"
    assert response["records"][0]["raw_details"]["whisper_meta"]["engine"] == "faster-whisper"


def test_hi_phy_dispatch_calls_all_three_source_stages(adapter, tmp_path, monkeypatch):
    request = _request(tmp_path, "Hi-Phy")
    observed = {}

    def run_two_stage(**kwargs):
        observed.update(kwargs)
        return {
            "overall_score": 75,
            "semantic_physics_score": 80,
            "visual_physics_score": 70,
            "confidence": 0.7,
            "expectations": [{"id": "E1"}],
            "semantic": {"checks": [{"expectation_id": "E1", "status": "match"}]},
            "visual": {"visual_violations": []},
        }

    fake = SimpleNamespace(
        run_two_stage_with_dual_stage2=run_two_stage,
        PROMPT_VARIANT=None,
        generate_content_text=lambda *args, **kwargs: "{}",
    )
    monkeypatch.setattr(adapter, "_load_source_module", lambda *args, **kwargs: fake)

    response = adapter.run_dimension(request)

    assert response["records"][0]["raw_score"] == 75.0
    assert observed["weights"] == (0.5, 0.5)
    assert observed["prompt_text"] == request["records"][0]["prompt_text"]
    assert observed["save_result_path"] is None
    assert response["records"][0]["raw_details"]["semantic_physics_score"] == 80
    assert response["records"][0]["raw_details"]["visual_physics_score"] == 70


def test_holistic_keeps_source_overall_subscores_confidence_and_evidence(adapter, tmp_path, monkeypatch):
    request = _request(tmp_path, "Holistic")
    observed = {}

    def judge(video_path, prompt_text, timeout_s):
        observed.update(video_path=video_path, prompt_text=prompt_text, timeout_s=timeout_s)
        return {
            "plot_alignment_score": 77,
            "subscores": {
                "narrative_alignment": 10,
                "shot_alignment": 20,
                "visual_attribute_alignment": 30,
                "audio_alignment": 40,
            },
            "confidence": 0.65,
            "observed_event_log": [{"where": "beginning", "audio": "bell"}],
            "checks": [{"constraint_id": "A1", "status": "match"}],
            "violations": [],
        }

    fake = SimpleNamespace(
        judge_plot_matching=judge,
        MODEL_NAME=None,
        PROMPT_VARIANT=None,
        generate_content_text=lambda *args, **kwargs: "{}",
    )
    monkeypatch.setattr(adapter, "_load_source_module", lambda *args, **kwargs: fake)

    response = adapter.run_dimension(request)
    record = response["records"][0]

    assert record["raw_score"] == 77.0
    assert record["raw_details"]["subscores"]["audio_alignment"] == 40
    assert record["raw_details"]["confidence"] == 0.65
    assert record["raw_details"]["observed_event_log"][0]["audio"] == "bell"
    assert observed["prompt_text"] == request["records"][0]["prompt_text"]
    assert response["raw_summary"]["overall"]["mean_plot_alignment_score_total"] == 77.0


def test_malformed_source_score_is_failed_without_hidden_zero(adapter, tmp_path, monkeypatch):
    request = _request(tmp_path, "Holistic")
    fake = SimpleNamespace(
        judge_plot_matching=lambda *args, **kwargs: {
            "plot_alignment_score": float("nan"),
            "subscores": {
                "narrative_alignment": 0,
                "shot_alignment": 0,
                "visual_attribute_alignment": 0,
                "audio_alignment": 0,
            },
            "confidence": 0.0,
        },
        MODEL_NAME=None,
        PROMPT_VARIANT=None,
        generate_content_text=lambda *args, **kwargs: "{}",
    )
    monkeypatch.setattr(adapter, "_load_source_module", lambda *args, **kwargs: fake)

    response = adapter.run_dimension(request)
    record = response["records"][0]

    assert record["status"] == "failed"
    assert record["raw_score"] is None
    assert record["raw_details"]["plot_alignment_score"] is None
    assert "invalid plot_alignment_score" in record["raw_details"]["_adapter_validation_error"]
    assert response["counts"]["scored"] == 0
    assert response["raw_summary"]["overall"]["mean_plot_alignment_score_total"] is None


def test_preflight_covers_every_id_and_never_imports_source(adapter, tmp_path, monkeypatch):
    request = _request(tmp_path, "Holistic", prompt="  ")
    missing = dict(request["records"][0])
    missing.update(id="missing-video", prompt_text="real prompt", staged_video_path=str(tmp_path / "none.mp4"))
    request["records"].append(missing)
    monkeypatch.setattr(
        adapter,
        "_load_source_module",
        lambda *args: pytest.fail("source must not load when all records fail preflight"),
    )

    response = adapter.run_dimension(request)

    assert [(row["id"], row["status"]) for row in response["records"]] == [
        ("record-é-猫", "missing_prompt"),
        ("missing-video", "missing_video"),
    ]
    assert response["counts"]["requested"] == 2
    assert response["counts"]["eligibility_unknown"] == 2
    assert response["counts"]["located"] == 1


def test_provider_credentials_are_redacted_from_structured_errors(
    adapter, tmp_path, monkeypatch
):
    request = _request(tmp_path, "Holistic")
    secret = "private-test-key-123"
    monkeypatch.setenv("GEMINI_API_KEY", secret)

    def fail(*args, **kwargs):
        raise RuntimeError(f"https://example.invalid?key={secret}&x=1")

    fake = SimpleNamespace(
        judge_plot_matching=fail,
        MODEL_NAME=None,
        PROMPT_VARIANT=None,
        generate_content_text=lambda *args, **kwargs: "{}",
    )
    monkeypatch.setattr(adapter, "_load_source_module", lambda *args, **kwargs: fake)

    response = adapter.run_dimension(request)

    encoded = json.dumps(response)
    assert secret not in encoded
    assert "<redacted>" in response["records"][0]["error"]


def test_request_rejects_unknown_revision_and_boolean_timeout(adapter, tmp_path):
    request = _request(tmp_path, "Hi-Phy")
    request["source_revision"] = "wrong"
    with pytest.raises(adapter.AdapterContractError, match="source_revision"):
        adapter.run_dimension(request)

    request.pop("source_revision")
    request["config"]["timeout_s"] = True
    response = adapter.run_dimension(request)
    assert response["records"][0]["status"] == "failed"
    assert "timeout_s" in response["records"][0]["error"]

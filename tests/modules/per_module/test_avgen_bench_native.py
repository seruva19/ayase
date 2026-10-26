"""Light contract tests for the source-backed AVGen native adapters."""

import ast
import contextlib
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from ayase.third_party.avgen_bench import adapters_native as native


def _pinned_source_checkout() -> Path:
    raw = os.environ.get("AYASE_AVGEN_TEST_SOURCE")
    if not raw:
        pytest.skip("AYASE_AVGEN_TEST_SOURCE does not name a pinned source checkout")
    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        pytest.skip("AYASE_AVGEN_TEST_SOURCE checkout not present")
    return path


def _request(tmp_path: Path, dimension: str, records=None):
    root = tmp_path / "источник с пробелом"
    root.mkdir(exist_ok=True)
    work = tmp_path / "рабочая папка"
    output = tmp_path / "результаты"
    videos = tmp_path / "видео"
    for path in (work, output, videos):
        path.mkdir(exist_ok=True)
    if records is None:
        video = videos / "ролик с пробелом.mp4"
        video.touch()
        records = [
            {
                "id": "запись-1",
                "video_path": str(video),
                "staged_video_path": str(video),
                "safe_content": "ролик с пробелом",
                "content": "Original title",
                "category": "категория",
                "prompt_text": "Original prompt text",
                "mode": "custom",
            }
        ]
    return {
        "schema": 1,
        "dimension": dimension,
        "source_dir": str(root),
        "models_dir": str(tmp_path / "models"),
        "work_dir": str(work),
        "videos_dir": str(videos),
        "prompts_dir": str(tmp_path / "prompts"),
        "face_prompts_dir": str(tmp_path / "face prompts"),
        "output_dir": str(output),
        "run_tag": "тест с пробелом",
        "assets": {},
        "config": {},
        "records": records,
    }


def _asset_file(tmp_path: Path, name: str) -> str:
    path = tmp_path / "модели с пробелом" / name
    path.parent.mkdir(exist_ok=True)
    path.touch()
    return str(path)


def _asset_dir(tmp_path: Path, name: str) -> str:
    path = tmp_path / "модели с пробелом" / name
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def test_vis_uses_joint_qalign_video_call(monkeypatch, tmp_path):
    calls = {}

    class Scores:
        def tolist(self):
            return [0.625]

    class Scorer:
        def __init__(self, pretrained, device):
            calls["init"] = (pretrained, device)

        def __call__(self, videos):
            calls["videos"] = videos
            return Scores()

    source = SimpleNamespace(
        QAlignVideoScorer=Scorer,
        load_video=lambda path: ["frame-0", "frame-1", "frame-2"],
    )

    @contextlib.contextmanager
    def local_nested_assets(model_dir):
        yield {
            "strategy": "test constructor hook",
            "model_dir": str(model_dir),
            "local_files_only": True,
            "redirect_counts": {"tokenizer": 1, "image_processor": 1},
        }

    monkeypatch.setattr(native, "_load_source_module", lambda *args: source)
    monkeypatch.setattr(native, "_qalign_local_nested_assets", local_nested_assets)
    request = _request(tmp_path, "Vis")
    request["assets"]["onealign_model_dir"] = _asset_dir(tmp_path, "one align")

    result = native.run_dimension(request)

    assert result["records"][0]["raw_score"] == pytest.approx(0.625)
    assert calls["videos"] == [["frame-0", "frame-1", "frame-2"]]
    assert result["records"][0]["raw_details"]["frames_sampled"] == 3


def test_actual_qalign_constructor_redirects_both_nested_assets_and_restores(tmp_path):
    source_root = _pinned_source_checkout()
    source_file = source_root / "eval" / "Q-Align" / "q_align" / "model" / "modeling_mplug_owl2.py"
    tree = ast.parse(source_file.read_text(encoding="utf-8"))
    source_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MPLUGOwl2LlamaForCausalLM"
    )
    constructor = next(
        node
        for node in source_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    calls = []

    class Parent:
        def __init__(self, config=None):
            pass

    class Base(Parent):
        pass

    class Instance(Base):
        def post_init(self):
            pass

    class Tokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append({"kind": "tokenizer", "path": path, "kwargs": kwargs})
            return lambda values: {"input_ids": [[0, index] for index in range(len(values))]}

    class Processor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            calls.append({"kind": "processor", "path": path, "kwargs": kwargs})
            return object()

    module = ModuleType("_test_actual_qalign_model_constructor")
    module.__dict__.update(
        LlamaForCausalLM=Base,
        MPLUGOwl2LlamaModel=lambda config: None,
        AutoTokenizer=Tokenizer,
        CLIPImageProcessor=Processor,
        nn=SimpleNamespace(Linear=lambda *args, **kwargs: object()),
    )
    exec(
        compile(ast.Module(body=[constructor], type_ignores=[]), str(source_file), "exec"),
        module.__dict__,
    )
    model_dir = Path(_asset_dir(tmp_path, "verified onealign snapshot")).resolve()

    with native._qalign_local_nested_assets(model_dir, modeling_module=module) as provenance:
        module.__dict__["__init__"](
            Instance(), SimpleNamespace(hidden_size=2, vocab_size=6)
        )

    assert calls == [
        {
            "kind": "tokenizer",
            "path": str(model_dir),
            "kwargs": {"local_files_only": True},
        },
        {
            "kind": "processor",
            "path": str(model_dir),
            "kwargs": {"local_files_only": True},
        },
    ]
    assert provenance["redirect_counts"] == {"tokenizer": 1, "image_processor": 1}
    assert module.AutoTokenizer is Tokenizer
    assert module.CLIPImageProcessor is Processor


def test_aud_uses_native_extraction_original_sr_and_selects_pq(monkeypatch, tmp_path):
    calls = {}
    predictor = object()

    def initialize_predictor(*, ckpt):
        calls["checkpoint"] = ckpt
        return predictor

    def score_one_video_audio(actual_predictor, video_path):
        calls["score"] = (actual_predictor, video_path)
        return {"CE": 1.0, "CU": 2.0, "PC": 3.0, "PQ": 7.25}

    source = SimpleNamespace(
        METRICS=["CE", "CU", "PC", "PQ"],
        initialize_predictor=initialize_predictor,
        score_one_video_audio=score_one_video_audio,
        safe_mean=lambda values: sum(values) / len(values),
    )
    monkeypatch.setattr(native, "_load_source_module", lambda *args: source)
    request = _request(tmp_path, "Aud")
    request["assets"]["audiobox_checkpoint"] = _asset_file(tmp_path, "audio box.pt")

    result = native.run_dimension(request)

    record = result["records"][0]
    assert record["raw_score"] == pytest.approx(7.25)
    assert record["raw_details"]["metrics"] == {"CE": 1.0, "CU": 2.0, "PC": 3.0, "PQ": 7.25}
    assert calls["score"][0] is predictor
    assert isinstance(calls["score"][1], Path)


def test_av_selects_continuous_softmax_expectation_not_argmax(monkeypatch, tmp_path):
    calls = {}
    transformer = SimpleNamespace(params=SimpleNamespace(off_head_cfg=SimpleNamespace(params=SimpleNamespace(out_features=5))))
    cfg = SimpleNamespace(
        data=SimpleNamespace(max_off_sec=1.0),
        model=SimpleNamespace(params=SimpleNamespace(transformer=transformer)),
    )

    class Model:
        def load_state_dict(self, state):
            calls["state"] = state

        def eval(self):
            calls["eval"] = True

    class Torch:
        @staticmethod
        def device(name):
            return f"device:{name}"

        @staticmethod
        def load(path, map_location, weights_only):
            calls["load"] = (path, map_location, weights_only)
            return {"model": {"weight": 1}}

    def reencode(path, vfps, afps, in_size):
        calls["reencode"] = (path, vfps, afps, in_size)
        return path

    def predict(**kwargs):
        calls["predict"] = kwargs
        return -0.125, -0.5, 0

    source = SimpleNamespace(
        torch=Torch,
        OmegaConf=SimpleNamespace(load=lambda path: cfg),
        patch_config=lambda value: value,
        make_class_grid=lambda left, right, count: [left, 0.0, right],
        get_model=lambda actual_cfg, device: (None, Model()),
        get_transforms=lambda actual_cfg, splits: {"test": "source-transform"},
        maybe_reencode_if_needed=reencode,
        predict_offset_for_video=predict,
        safe_mean=lambda values: sum(values) / len(values),
    )
    monkeypatch.setattr(native, "_load_source_module", lambda *args: source)
    request = _request(tmp_path, "AV")
    request["assets"].update(
        synchformer_checkpoint=_asset_file(tmp_path, "sync former.pt"),
        synchformer_config=_asset_file(tmp_path, "sync config.yaml"),
    )

    result = native.run_dimension(request)

    record = result["records"][0]
    assert record["raw_score"] == pytest.approx(0.125)
    assert record["raw_details"]["pred_offset_sec_argmax"] == pytest.approx(-0.5)
    assert result["raw_summary"]["mean_abs_offset_cont_sec"] == pytest.approx(0.125)
    assert calls["reencode"][1:] == (25, 16000, 256)


def test_lip_pools_all_passing_tracks_at_launcher_threshold(monkeypatch, tmp_path):
    video_a = tmp_path / "видео" / "A clip.mp4"
    video_b = tmp_path / "видео" / "B clip.mp4"
    video_a.parent.mkdir(exist_ok=True)
    video_a.touch()
    video_b.touch()
    records = [
        {"id": "A", "staged_video_path": str(video_a), "category": "x"},
        {"id": "B", "staged_video_path": str(video_b), "category": "x"},
    ]
    request = _request(tmp_path, "Lip", records)
    request["assets"].update(
        syncnet_checkpoint=_asset_file(tmp_path, "syncnet v2.model"),
        s3fd_checkpoint=_asset_file(tmp_path, "s3fd face.pth"),
    )
    request["config"]["ffmpeg_path"] = _asset_file(tmp_path, "ffmpeg.exe")
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "inference.py").touch()
    monkeypatch.setattr(native, "_build_syncnet_overlay", lambda *args: overlay)

    thresholds = []

    def eval_one_video(cls, path, inference_py, data_dir, *args):
        thresholds.append(args[-2])
        if path.name.startswith("A"):
            return {
                "status": "ok",
                "ran": True,
                "tracks_total": 2,
                "pass_tracks": [{"offset": 0, "conf": 1.0, "min_dist": 4.0}],
            }
        return {
            "status": "ok",
            "ran": True,
            "tracks_total": 3,
            "pass_tracks": [
                {"offset": -8, "conf": 2.0, "min_dist": 3.0},
                {"offset": 8, "conf": 2.0, "min_dist": 3.0},
                {"offset": 8, "conf": 2.0, "min_dist": 3.0},
            ],
        }

    source = SimpleNamespace(
        eval_one_video=eval_one_video,
        safe_mean=lambda values: sum(values) / len(values),
    )
    monkeypatch.setattr(native, "_load_source_module", lambda *args: source)

    result = native.run_dimension(request)

    assert thresholds == [1.0, 1.0]
    assert result["raw_summary"]["mean_abs_offset_frames"] == pytest.approx(6.0)
    assert result["raw_summary"]["n_tracks_pass_conf"] == 4
    assert result["records"][1]["raw_details"]["passing_tracks"][0] == {
        "offset_frames": -8,
        "confidence": 2.0,
        "distance": 3.0,
    }
    assert result["records"][1]["raw_score"] == pytest.approx(8.0)


def test_face_calls_source_evaluator_and_keeps_40_60_breakdown(monkeypatch, tmp_path):
    videos_root = tmp_path / "видео"
    staged = videos_root / "news" / "safe title.mp4"
    staged.parent.mkdir(parents=True)
    staged.touch()
    request = _request(
        tmp_path,
        "Face",
        [
            {
                "id": "face-1",
                "staged_video_path": str(staged),
                "category": "news",
                "safe_content": "safe title",
                "content": "Original title",
                "prompt_text": "Prompt unchanged",
                "expected_faces": {"min_ids": 1, "max_ids": 2},
            }
        ],
    )
    buffalo = Path(_asset_dir(tmp_path, "insight root")) / "models" / "buffalo_l"
    buffalo.mkdir(parents=True)
    request["assets"]["buffalo_dir"] = str(buffalo.parent.parent)
    calls = {}

    class Analyzer:
        pass

    def evaluate_category(**kwargs):
        calls["category"] = kwargs
        item = kwargs["label_items"][0]
        assert item["content"] == "safe title"
        assert item["prompt"] == "Prompt unchanged"
        return [
            {
                "content": "safe title",
                "status": "ok",
                "final_score": 73.0,
                "expected_primary_faces": {"min_ids": 1, "max_ids": 2},
                "breakdown": {"score_count_40": 40.0, "score_stability_60": 33.0},
                "observed": {"k_primary_pred": 1},
                "metrics": {"total_face_dets": 50},
                "dbscan": {"eps": 0.65, "min_samples": 1},
            }
        ], {"category": "news", "count_scored": 1, "mean_score": 73.0}

    source = SimpleNamespace(MultiFaceConsistencyV2=Analyzer, evaluate_category=evaluate_category)
    monkeypatch.setattr(native, "_load_source_module", lambda *args: source)

    class FaceAnalysis:
        def __init__(self, **kwargs):
            calls["face_init"] = kwargs

        def prepare(self, **kwargs):
            calls["prepare"] = kwargs

    monkeypatch.setattr(
        native,
        "_import_dependency",
        lambda name: SimpleNamespace(FaceAnalysis=FaceAnalysis),
    )

    result = native.run_dimension(request)

    record = result["records"][0]
    assert record["raw_score"] == pytest.approx(73.0)
    assert record["raw_details"]["score_formula"] == "40*count_range + 60*primary_stability"
    assert calls["face_init"]["root"] == str(buffalo.parent.parent)
    assert result["raw_summary"]["canonical_expected_annotations"] == 220
    assert result["raw_summary"]["canonical_expected_label_eligible"] == 103


def test_face_max_zero_is_ineligible_before_missing_video_or_backend(monkeypatch, tmp_path):
    missing = tmp_path / "does not exist.mp4"
    request = _request(
        tmp_path,
        "Face",
        [
            {
                "id": "no-face",
                "staged_video_path": str(missing),
                "expected_faces": {"min_ids": 0, "max_ids": 0},
            }
        ],
    )
    monkeypatch.setattr(
        native,
        "_load_source_module",
        lambda *args: pytest.fail("no-face labels must not initialize the backend"),
    )

    result = native.run_dimension(request)

    assert result["records"][0]["status"] == "ineligible"
    assert result["records"][0]["eligible"] is False
    assert "raw_score" not in result["records"][0]


def test_lophy_uses_pc_32frames_source_parser_and_generation_kwargs(monkeypatch, tmp_path):
    calls = {}

    def build_model_and_processor(**kwargs):
        calls["build"] = kwargs
        return "model", "processor", "tokenizer"

    def score_video(**kwargs):
        calls["score"] = kwargs
        return 0, "No reliable rating was emitted."

    source = SimpleNamespace(
        torch=SimpleNamespace(bfloat16="bf16"),
        generate_kwargs={
            "do_sample": False,
            "top_k": 1,
            "temperature": 0.001,
            "max_length": 256,
        },
        build_model_and_processor=build_model_and_processor,
        build_prompt=lambda task: f"source prompt:{task}",
        score_video=score_video,
        parse_score_from_output=lambda output: 0,
        safe_mean=lambda values: sum(values) / len(values),
    )
    monkeypatch.setattr(native, "_load_source_module", lambda *args: source)
    request = _request(tmp_path, "Lo-Phy")
    request["assets"]["videophy_model_dir"] = _asset_dir(tmp_path, "video phy model")

    result = native.run_dimension(request)

    record = result["records"][0]
    assert record["raw_score"] == 0.0
    assert record["raw_details"]["parser"]["branch"] == "fallback_zero"
    assert calls["score"]["num_frames"] == 32
    assert calls["score"]["prompt"] == "source prompt:pc"
    assert calls["build"]["dtype"] == "bf16"


@pytest.mark.parametrize(
    ("text", "score", "branch", "matched"),
    [
        ("FIVE (5)", 5, "substring", "five"),
        ("score: 0", 0, "substring", "0"),
        ("unparseable", 0, "fallback_zero", None),
    ],
)
def test_videophy_parser_provenance_matches_released_order(text, score, branch, matched):
    provenance = native._videophy_parse_provenance(text)
    assert provenance == {"branch": branch, "matched": matched, "parsed_score": score}


def test_syncnet_overlay_removes_shell_invocation_and_injects_assets(tmp_path):
    source = _pinned_source_checkout() / "eval" / "syncnet_python"

    overlay = native._build_syncnet_overlay(source, tmp_path)
    inference = (overlay / "inference.py").read_text(encoding="utf-8")
    detector = (overlay / "detectors" / "s3fd" / "__init__.py").read_text(encoding="utf-8")
    instance = (overlay / "SyncNetInstance.py").read_text(encoding="utf-8")

    assert "shell=True" not in inference
    assert "AYASE_AVGEN_SYNCNET_CHECKPOINT" in inference
    assert "AYASE_AVGEN_S3FD_CHECKPOINT" in detector
    assert instance.count("AYASE_AVGEN_FFMPEG") == 2


def test_real_syncnet_parser_extracts_all_printed_tracks():
    source_root = _pinned_source_checkout() / "eval" / "syncnet_python"
    source = native._load_source_module(
        "_test_avgen_syncnet_parser", source_root / "batch_eval.py", source_root
    )
    output = """
    Track 0:
      AV offset:  -8 frames
      Min dist:   3.250
      Confidence: 1.000
    Track 1:
      AV offset:  +2 frames
      Min dist:   4.500
      Confidence: 0.999
    """

    assert source.parse_tracks(output) == [
        {"offset": -8, "min_dist": 3.25, "conf": 1.0},
        {"offset": 2, "min_dist": 4.5, "conf": 0.999},
    ]


def test_missing_video_is_represented_without_loading_backend(monkeypatch, tmp_path):
    request = _request(
        tmp_path,
        "Vis",
        [{"id": "missing", "staged_video_path": str(tmp_path / "missing.mp4")}],
    )
    monkeypatch.setattr(
        native,
        "_load_source_module",
        lambda *args: pytest.fail("missing media must not initialize a backend"),
    )

    result = native.run_dimension(request)

    assert result["records"] == [
        {
            "id": "missing",
            "status": "missing_video",
            "eligible": True,
            "raw_details": {},
            "error": f"video not found: {(tmp_path / 'missing.mp4').resolve()}",
        }
    ]
    assert result["counts"]["requested"] == 1
    assert result["counts"]["located"] == 0

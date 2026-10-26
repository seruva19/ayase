"""Public AVGen-Bench orchestration tests with every provisioner/worker mocked."""

import copy
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ayase.models import CaptionMetadata, Sample
from ayase.third_party.avgen_bench.protocol import (
    DIMENSION_ORDER,
    SAMPLE_SCORE_FIELDS,
    SOURCE_DIMENSION_KEYS,
    canonical_face_annotations,
    canonical_prompts,
)

SOURCE_TO_INTERNAL = {value: key for key, value in SOURCE_DIMENSION_KEYS.items()}


@pytest.fixture(autouse=True)
def _exercise_mocked_public_orchestrator(monkeypatch):
    """The suite's global light mode must not suppress explicitly mocked workers."""

    from ayase.modules.avgen_bench import AVGenBenchModule

    monkeypatch.setattr(AVGenBenchModule, "_global_test_mode", False)


class FakePipeline:
    def __init__(self):
        self.metrics = {}

    def add_dataset_metric(self, name, value):
        self.metrics[name] = value


def _sample(path: Path, caption="A custom audiovisual prompt", **metadata):
    path.write_bytes(b"not decoded by mocked workers")
    caption_metadata = (
        CaptionMetadata(text=caption, length=len(caption)) if caption is not None else None
    )
    return Sample(path=path, is_video=True, caption=caption_metadata, metadata=metadata)


def _raw_scores(internal, index):
    values = {
        "visual": (0.2, 0.8),
        "audio": (4.0, 8.0),
        "av_sync": (0.1, 0.3),
        "lip_sync": (0.0, 8.0),
        "text": (20.0, 80.0),
        "face": (30.0, 70.0),
        "music": (40.0, 60.0),
        "speech": (50.0, 70.0),
        "low_physics": (1.0, 5.0),
        "high_physics": (20.0, 100.0),
        "holistic": (40.0, 80.0),
    }
    return values[internal][index % 2]


def _result_for_request(request, *, failure=None, failed_record=None, canonical=False):
    internal = SOURCE_TO_INTERNAL[request["dimension"]]
    if failure == internal:
        raise RuntimeError(f"synthetic {internal} component crash")
    rows = []
    for index, record in enumerate(request["records"]):
        if failed_record == (internal, index):
            rows.append(
                {
                    "id": record["id"],
                    "status": "failed",
                    "eligible": None,
                    "raw_score": None,
                    "raw_details": {},
                    "error": "synthetic record failure",
                }
            )
            continue
        if canonical and internal == "face":
            annotation = record.get("expected_faces")
            if annotation is None:
                status, eligible = "missing_face_label", None
            elif annotation["primary_faces"]["max_ids"] == 0:
                status, eligible = "ineligible", False
            else:
                status, eligible = "scored", True
            raw = 75.0 if status == "scored" else None
            details = {}
        elif (
            canonical
            and internal == "music"
            and record["category"] != "musical_instrument_tutorial"
        ):
            status, eligible, raw, details = "ineligible", False, None, {}
        else:
            status, eligible = "scored", True
            raw = 50.0 if canonical else _raw_scores(internal, index)
            details = {}
            if internal == "lip_sync":
                if canonical:
                    details["passing_tracks"] = [
                        {"offset_frames": 2, "confidence": 1.0, "distance": 4.0}
                    ]
                    raw = 2.0
                elif index % 2 == 0:
                    details["passing_tracks"] = [
                        {"offset_frames": 0, "confidence": 1.0, "distance": 4.0}
                    ]
                else:
                    details["passing_tracks"] = [
                        {"offset_frames": 8, "confidence": 2.0, "distance": 4.0},
                        {"offset_frames": -8, "confidence": 3.0, "distance": 4.0},
                        {"offset_frames": 8, "confidence": 4.0, "distance": 4.0},
                    ]
        rows.append(
            {
                "id": record["id"],
                "status": status,
                "eligible": eligible,
                "raw_score": raw,
                "raw_details": details,
                "evidence_path": str(Path(request["output_dir"]) / f"{record['id']}.json"),
            }
        )
    scored = [row["raw_score"] for row in rows if row["status"] == "scored"]
    return {
        "schema": 1,
        "source_revision": request["source_revision"],
        "dimension": request["dimension"],
        "protocol_identity": "mock-protocol",
        "backend_identity": "mock-backend",
        "adapter_digest": "1" * 64,
        "asset_identities": request["asset_identities"],
        "records": rows,
        "raw_summary": {
            "count_scored": len(scored),
            "raw_mean": sum(scored) / len(scored) if scored else None,
        },
        "counts": {
            "requested": len(rows),
            "scored": len(scored),
            "failed": sum(row["status"] == "failed" for row in rows),
        },
        "log_path": str(Path(request["output_dir"]) / "worker.log"),
    }


def _install_fakes(monkeypatch, tmp_path, *, failure=None, failed_record=None, canonical=False):
    calls = {
        "runtime": [],
        "assets": [],
        "asset_overrides": {},
        "workers": [],
        "requests": [],
        "ffmpeg": [],
    }
    source = tmp_path / "source"
    source.mkdir(exist_ok=True)
    ffmpeg = tmp_path / "ffmpeg.exe"
    ffmpeg.write_bytes(b"mock executable")

    def stage_records(records, work_dir):
        work = Path(work_dir)
        videos, prompts, faces = work / "videos", work / "prompts", work / "faces"
        for directory in (videos, prompts, faces):
            directory.mkdir(parents=True, exist_ok=True)
        staged = []
        for row in records:
            copied = copy.deepcopy(row)
            copied["staged_video_path"] = copied["video_path"]
            copied["safe_content"] = "mock-" + copied["id"]
            staged.append(copied)
        return {
            "schema": 1,
            "records": staged,
            "videos_dir": str(videos.resolve()),
            "prompts_dir": str(prompts.resolve()),
            "face_prompts_dir": str(faces.resolve()),
        }

    def ensure_runtime(dimension, models_dir, *, test_mode=False):
        assert not test_mode
        calls["runtime"].append(dimension)
        return Path(sys.executable)

    def execute_worker(request, interpreter, *, api_key=None, timeout_s=0, test_mode=False):
        assert not test_mode
        assert Path(interpreter).is_absolute()
        assert timeout_s == 14400
        serialized = repr(request)
        assert "synthetic-private-key" not in serialized
        for name in ("source_dir", "models_dir", "work_dir", "videos_dir", "output_dir"):
            assert Path(request[name]).is_absolute()
        calls["workers"].append(request["dimension"])
        calls["requests"].append(copy.deepcopy(request))
        return _result_for_request(
            request, failure=failure, failed_record=failed_record, canonical=canonical
        )

    runtime = SimpleNamespace(
        ensure_source=lambda models_dir, source_override=None, test_mode=False: source,
        stage_records=stage_records,
        ensure_runtime=ensure_runtime,
        execute_worker=execute_worker,
        runtime_ffmpeg=lambda interpreter, models_dir: (
            calls["ffmpeg"].append(calls["runtime"][-1]) or ffmpeg
        ),
    )

    def ensure_assets(dimension, models_dir, *, overrides=None, test_mode=False):
        assert not test_mode
        calls["assets"].append(dimension)
        calls["asset_overrides"][dimension] = copy.deepcopy(overrides)
        return {
            "paths": {"mock_asset": tmp_path / dimension / "model.bin"},
            "asset_identities": {"mock_asset": {"dimension": dimension}},
        }

    asset_keys = {
        "Vis": ("onealign_model_dir",),
        "Aud": ("audiobox_checkpoint",),
        "AV": ("synchformer_checkpoint", "synchformer_config"),
        "Lip": ("syncnet_checkpoint", "s3fd_checkpoint"),
        "Text": ("ocr_det_dir", "ocr_rec_dir"),
        "Face": ("buffalo_dir",),
        "Music": ("basic_pitch_model",),
        "Speech": ("whisper_model_dir",),
        "Lo-Phy": ("videophy_model_dir",),
        "Hi-Phy": (),
        "Holistic": (),
    }
    assets = SimpleNamespace(
        ensure_assets=ensure_assets,
        required_asset_keys=lambda dimension: asset_keys[dimension],
    )
    import ayase.modules.avgen_bench as module_source

    monkeypatch.setattr(module_source, "_load_runtime", lambda: runtime)
    monkeypatch.setattr(module_source, "_load_assets_runtime", lambda: assets)
    return calls


def _module(tmp_path, **config):
    from ayase.modules.avgen_bench import AVGenBenchModule

    module = AVGenBenchModule(
        {
            "models_dir": str(tmp_path / "models"),
            "api_key": "synthetic-private-key",
            **config,
        }
    )
    module.pipeline = FakePipeline()
    module.setup()
    return module


def test_module_contract_and_setup_are_deferred(tmp_path):
    from ayase.modules.avgen_bench import AVGenBenchModule

    assert AVGenBenchModule.name == "avgen_bench"
    assert AVGenBenchModule.input_type == "video"
    assert len(AVGenBenchModule.models) == 12
    assert set(SAMPLE_SCORE_FIELDS.values()) <= set(AVGenBenchModule.metric_info)
    assert AVGenBenchModule.provenance["avgen_bench_score"] == "adapted"
    module = _module(tmp_path)
    assert module._backend == "deferred:eleven-specialists"
    assert module._ml_available


def test_dimension_major_execution_and_independent_raw_aggregation(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch, tmp_path)
    face_labels = {"primary_faces": {"min_ids": 1, "max_ids": 2}}
    samples = [
        _sample(
            tmp_path / "a.mp4",
            avgen_bench_category="custom",
            avgen_bench_expected_faces=face_labels,
        ),
        _sample(
            tmp_path / "b.mp4",
            avgen_bench_category="custom",
            avgen_bench_expected_faces=face_labels,
        ),
    ]
    module = _module(tmp_path)

    assert all(module.process(sample) is sample for sample in samples)
    module.post_process(samples)

    source_order = [SOURCE_DIMENSION_KEYS[key] for key in DIMENSION_ORDER]
    assert calls["runtime"] == calls["assets"] == calls["workers"] == source_order
    assert calls["ffmpeg"] == ["AV", "Lip", "Music"]
    requests = {request["dimension"]: request for request in calls["requests"]}
    assert "device" not in requests["Vis"]["config"]
    assert "device" not in requests["AV"]["config"]
    assert "device" not in requests["Lo-Phy"]["config"]
    assert requests["Text"]["config"]["device"] == "auto"
    stats = module.pipeline.metrics
    assert stats["avgen_bench_basic_score"] == pytest.approx(55.0)
    assert stats["avgen_bench_cross_score"] == pytest.approx(42.5)
    assert stats["avgen_bench_fine_score"] == pytest.approx(390.0 / 7.0)
    assert stats["avgen_bench_score"] == pytest.approx(741.0 / 14.0)
    assert stats["avgen_bench_coverage"] == pytest.approx(1.0)
    lip = stats["avgen_bench_metrics"]["aggregation"]["dimensions"]["lip_sync"]
    assert lip["raw_mean"] == 6.0
    assert lip["evidence_count"] == 4
    assert lip["normalized_score"] == 25.0
    assert stats["avgen_bench_metrics"]["canonical_comparable"] is False

    expected_per_sample = {
        "visual": (20.0, 80.0),
        "audio": (40.0, 80.0),
        "av_sync": (80.0, 40.0),
        "lip_sync": (100.0, 0.0),
        "text": (20.0, 80.0),
        "face": (30.0, 70.0),
        "music": (40.0, 60.0),
        "speech": (50.0, 70.0),
        "low_physics": (20.0, 100.0),
        "high_physics": (20.0, 100.0),
        "holistic": (40.0, 80.0),
    }
    for index, sample in enumerate(samples):
        assert sample.metadata["avgen_bench"]["status"] == "evaluated"
        assert len(sample.metadata["avgen_bench"]["components"]) == 11
        assert sample.metadata["avgen_bench"]["components"]["music"]["status"] == "scored"
        assert sample.quality_metrics.avgen_bench_music_score is not None
        for internal, values in expected_per_sample.items():
            assert getattr(sample.quality_metrics, SAMPLE_SCORE_FIELDS[internal]) == pytest.approx(
                values[index]
            )


def test_failed_records_are_absent_not_fake_zero(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path, failed_record=("visual", 1))
    samples = [_sample(tmp_path / "a.mp4"), _sample(tmp_path / "b.mp4")]
    module = _module(tmp_path)
    module.post_process(samples)

    visual = module.pipeline.metrics["avgen_bench_metrics"]["aggregation"]["dimensions"]["visual"]
    assert visual["raw_mean"] == 0.2
    assert visual["normalized_score"] == 20.0
    assert visual["evidence_count"] == 1
    assert samples[1].quality_metrics.avgen_bench_visual_score is None
    assert samples[1].metadata["avgen_bench"]["components"]["visual"]["status"] == "failed"


def test_component_failure_does_not_stop_remaining_heads_or_create_zero(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch, tmp_path, failure="audio")
    sample = _sample(
        tmp_path / "a.mp4",
        avgen_bench_expected_faces={"primary_faces": {"min_ids": 1, "max_ids": 1}},
    )
    module = _module(tmp_path)
    module.post_process([sample])

    assert calls["workers"] == [SOURCE_DIMENSION_KEYS[key] for key in DIMENSION_ORDER]
    assert sample.quality_metrics.avgen_bench_audio_score is None
    assert sample.metadata["avgen_bench"]["components"]["audio"]["status"] == "failed"
    audio = module.pipeline.metrics["avgen_bench_metrics"]["aggregation"]["dimensions"]["audio"]
    assert audio["raw_mean"] is None
    assert not audio["available"]
    assert module.pipeline.metrics["avgen_bench_coverage"] == pytest.approx(0.9)


def test_missing_credentials_preflight_skips_all_five_remote_provisioners(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    calls = _install_fakes(monkeypatch, tmp_path, canonical=True)
    face = next(
        item
        for item in canonical_face_annotations()
        if item.category == "musical_instrument_tutorial" and item.max_ids > 0
    )
    sample = _sample(
        tmp_path / "a.mp4",
        avgen_bench_category=face.category,
        avgen_bench_content=face.content,
    )
    module = _module(tmp_path, api_key=None, input_mode="canonical")
    module.post_process([sample])

    local = ["Vis", "Aud", "AV", "Lip", "Face", "Lo-Phy"]
    assert calls["runtime"] == calls["assets"] == calls["workers"] == local
    for internal in ("text", "music", "speech", "high_physics", "holistic"):
        component = sample.metadata["avgen_bench"]["components"][internal]
        assert component["status"] == "missing_credentials"
        assert getattr(sample.quality_metrics, SAMPLE_SCORE_FIELDS[internal]) is None
    assert module.pipeline.metrics["avgen_bench_coverage"] == pytest.approx(4 / 10 + 6 / 35)
    assert module.last_metadata["gemini_credentials_available"] is False


def test_test_mode_skips_all_runtime_asset_worker_and_api_actions(tmp_path, monkeypatch):
    import ayase.modules.avgen_bench as module_source

    monkeypatch.setattr(
        module_source,
        "_load_runtime",
        lambda: pytest.fail("test mode must not import the runtime"),
    )
    monkeypatch.setattr(
        module_source,
        "_load_assets_runtime",
        lambda: pytest.fail("test mode must not import asset provisioning"),
    )
    sample = _sample(tmp_path / "a.mp4")
    module = module_source.AVGenBenchModule({"test_mode": True})
    module.pipeline = FakePipeline()

    module.setup()
    assert module.process(sample) is sample
    module.post_process([sample])
    assert module.pipeline.metrics == {}
    assert "avgen_bench" not in sample.metadata


def test_custom_caption_is_primary_and_default_identity_is_not_filename_derived(
    tmp_path, monkeypatch
):
    calls = _install_fakes(monkeypatch, tmp_path)
    sample = _sample(
        tmp_path / "semantic-looking-file-name.mp4",
        caption="Caption wins",
        avgen_bench_prompt="Explicit fallback",
        avgen_bench_expected_faces={
            "primary_faces": {"min_ids": 1, "max_ids": 2},
            "primary_characters": [],
            "constraints": {},
        },
    )
    module = _module(tmp_path)
    module.post_process([sample])

    record = calls["requests"][0]["records"][0]
    assert record["category"] == "custom"
    assert record["content"] == "custom-000001"
    assert record["prompt_text"] == "Caption wins"
    assert record["expected_faces"]["primary_faces"] == {"min_ids": 1, "max_ids": 2}
    assert sample.metadata["avgen_bench"]["input"]["provenance"]["prompt"] == "sample_caption"


def test_captionless_custom_keeps_native_heads_and_preflights_prompted_heads(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch, tmp_path)
    sample = _sample(tmp_path / "captionless.mp4", caption=None)
    module = _module(tmp_path)
    module.post_process([sample])

    assert calls["workers"] == ["Vis", "Aud", "AV", "Lip", "Lo-Phy"]
    assert calls["runtime"] == calls["assets"] == calls["workers"]
    metadata = sample.metadata["avgen_bench"]
    assert metadata["input"]["prompt"] == ""
    assert metadata["input"]["provenance"]["prompt"] == "absent"
    assert metadata["status"] == "partial"
    assert len(metadata["components"]) == 11
    for internal in ("visual", "audio", "av_sync", "lip_sync", "low_physics"):
        assert metadata["components"][internal]["status"] == "scored"
    for internal in ("text", "speech", "high_physics", "holistic"):
        assert metadata["components"][internal]["status"] == "missing_prompt"
    assert metadata["components"]["face"]["status"] == "missing_face_label"
    assert metadata["components"]["music"]["status"] == "missing_prompt"


def test_captionless_custom_face_labels_run_face_without_remote_judges(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch, tmp_path)
    sample = _sample(
        tmp_path / "captionless-face.mp4",
        caption=None,
        avgen_bench_expected_faces={"primary_faces": {"min_ids": 1, "max_ids": 2}},
    )
    module = _module(tmp_path)
    module.post_process([sample])

    assert calls["workers"] == ["Vis", "Aud", "AV", "Lip", "Face", "Lo-Phy"]
    assert sample.metadata["avgen_bench"]["components"]["face"]["status"] == "scored"
    assert sample.quality_metrics.avgen_bench_face_score == 30.0


def test_mixed_blank_prompts_filter_only_remote_worker_rows(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch, tmp_path)
    labels = {"primary_faces": {"min_ids": 1, "max_ids": 1}}
    blank = _sample(tmp_path / "blank.mp4", caption=None, avgen_bench_expected_faces=labels)
    prompted = _sample(tmp_path / "prompted.mp4", avgen_bench_expected_faces=labels)
    module = _module(tmp_path)
    module.post_process([blank, prompted])

    requests = {request["dimension"]: request for request in calls["requests"]}
    for source_key in ("Vis", "Aud", "AV", "Lip", "Face", "Lo-Phy"):
        assert len(requests[source_key]["records"]) == 2
    for source_key in ("Text", "Speech", "Hi-Phy", "Holistic"):
        assert [row["id"] for row in requests[source_key]["records"]] == ["sample-000002"]
    for internal in ("text", "speech", "high_physics", "holistic"):
        assert blank.metadata["avgen_bench"]["components"][internal]["status"] == "missing_prompt"
        assert prompted.metadata["avgen_bench"]["components"][internal]["status"] == "scored"
    assert len(blank.metadata["avgen_bench"]["components"]) == 11
    assert len(prompted.metadata["avgen_bench"]["components"]) == 11


@pytest.mark.parametrize(
    ("expected_faces", "expected_status"),
    [
        (None, "missing_face_label"),
        ({"primary_faces": {"min_ids": 0, "max_ids": 0}}, "ineligible"),
    ],
)
def test_face_without_positive_labels_skips_runtime_and_assets(
    tmp_path, monkeypatch, expected_faces, expected_status
):
    calls = _install_fakes(monkeypatch, tmp_path)
    metadata = {}
    if expected_faces is not None:
        metadata["avgen_bench_expected_faces"] = expected_faces
    sample = _sample(tmp_path / "face-preflight.mp4", **metadata)
    module = _module(tmp_path)
    module.post_process([sample])

    assert "Face" not in calls["runtime"]
    assert "Face" not in calls["assets"]
    component = sample.metadata["avgen_bench"]["components"]["face"]
    assert component["status"] == expected_status
    assert sample.quality_metrics.avgen_bench_face_score is None


def test_music_source_selection_preflights_irrelevant_canonical_category(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch, tmp_path, canonical=True)
    face = next(
        item for item in canonical_face_annotations() if item.category == "ads" and item.max_ids > 0
    )
    sample = _sample(
        tmp_path / "ads.mp4",
        avgen_bench_category=face.category,
        avgen_bench_content=face.content,
    )
    module = _module(tmp_path, input_mode="canonical")
    module.post_process([sample])

    assert "Music" not in calls["runtime"]
    assert "Music" not in calls["assets"]
    assert sample.metadata["avgen_bench"]["components"]["music"]["status"] == "ineligible"
    assert sample.quality_metrics.avgen_bench_music_score is None


def test_music_mixed_categories_preserves_source_wrapper_row_selection(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch, tmp_path, canonical=True)
    music_face = next(
        item
        for item in canonical_face_annotations()
        if item.category == "musical_instrument_tutorial" and item.max_ids > 0
    )
    ads_face = next(
        item for item in canonical_face_annotations() if item.category == "ads" and item.max_ids > 0
    )
    samples = [
        _sample(
            tmp_path / "music.mp4",
            avgen_bench_category=music_face.category,
            avgen_bench_content=music_face.content,
        ),
        _sample(
            tmp_path / "ads.mp4",
            avgen_bench_category=ads_face.category,
            avgen_bench_content=ads_face.content,
        ),
    ]
    module = _module(tmp_path, input_mode="canonical")
    module.post_process(samples)

    music_request = next(
        request for request in calls["requests"] if request["dimension"] == "Music"
    )
    assert len(music_request["records"]) == 2
    assert samples[0].metadata["avgen_bench"]["components"]["music"]["status"] == "scored"
    assert samples[1].metadata["avgen_bench"]["components"]["music"]["status"] == "ineligible"


def test_flat_asset_overrides_are_filtered_per_head_and_unknown_keys_fail(tmp_path, monkeypatch):
    onealign = tmp_path / "onealign"
    audio = tmp_path / "audiobox.pt"
    calls = _install_fakes(monkeypatch, tmp_path)
    sample = _sample(tmp_path / "a.mp4")
    module = _module(
        tmp_path,
        asset_overrides={
            "onealign_model_dir": onealign,
            "audiobox_checkpoint": audio,
        },
    )
    module.post_process([sample])

    assert calls["asset_overrides"]["Vis"] == {"onealign_model_dir": str(onealign.resolve())}
    assert calls["asset_overrides"]["Aud"] == {"audiobox_checkpoint": str(audio.resolve())}
    assert calls["asset_overrides"]["AV"] is None

    import ayase.modules.avgen_bench as module_source

    assets_runtime = module_source._load_assets_runtime()
    bad = _module(tmp_path, asset_overrides={"onealgin_model_dir": onealign})
    with pytest.raises(ValueError, match="unknown flat"):
        bad._asset_overrides("visual", assets_runtime)


def test_worker_config_recursively_removes_secret_keys(tmp_path):
    module = _module(
        tmp_path,
        dimension_config={
            "Vis": {
                "api_key": "top-secret",
                "nested": {"auth_token": "nested-secret", "kept": 3},
            }
        },
    )
    config = module._worker_config("visual")
    assert "api_key" not in config
    assert config["nested"] == {"kept": 3}
    assert "top-secret" not in repr(config)
    assert "nested-secret" not in repr(config)


def test_custom_requires_prompt_and_ambiguous_basename_mapping_is_never_selected(
    tmp_path, monkeypatch
):
    first_dir, second_dir = tmp_path / "one", tmp_path / "two"
    first_dir.mkdir()
    second_dir.mkdir()
    samples = [
        _sample(first_dir / "same.mp4", caption=None),
        _sample(second_dir / "same.mp4", caption=None),
    ]
    import ayase.modules.avgen_bench as module_source

    monkeypatch.setattr(
        module_source,
        "_load_runtime",
        lambda: pytest.fail("invalid inputs must not provision a runtime"),
    )
    module = module_source.AVGenBenchModule(
        {
            "models_dir": str(tmp_path / "models"),
            "input_mapping": {"same.mp4": {"prompt": "must not be selected"}},
        }
    )
    module.pipeline = FakePipeline()
    module.setup()
    module.post_process(samples)

    assert all(sample.metadata["avgen_bench"]["status"] == "invalid_input" for sample in samples)
    assert all(
        "ambiguous basename" in sample.metadata["avgen_bench"]["error"] for sample in samples
    )
    assert module.pipeline.metrics["avgen_bench_score"] is None
    assert module.pipeline.metrics["avgen_bench_coverage"] == 0.0


def test_canonical_mode_uses_exact_bundled_prompt_and_released_face_annotation(
    tmp_path, monkeypatch
):
    calls = _install_fakes(monkeypatch, tmp_path, canonical=True)
    face = next(
        item
        for item in canonical_face_annotations()
        if item.category == "musical_instrument_tutorial" and item.max_ids > 0
    )
    prompt = next(
        item
        for item in canonical_prompts()
        if (item.category, item.content) == (face.category, face.content)
    )
    sample = _sample(
        tmp_path / "canonical.mp4",
        avgen_bench_category=prompt.category,
        avgen_bench_content=prompt.content,
        avgen_bench_prompt="must be ignored",
        avgen_bench_expected_faces={"primary_faces": {"min_ids": 99, "max_ids": 99}},
    )
    module = _module(tmp_path, input_mode="canonical")
    module.post_process([sample])

    record = calls["requests"][0]["records"][0]
    assert record["prompt_text"] == prompt.prompt
    assert record["expected_faces"]["content"] == prompt.content
    ignored = sample.metadata["avgen_bench"]["input"]["provenance"]["ignored_canonical_overrides"]
    assert ignored == ["sample_metadata:prompt", "sample_metadata:expected_faces"]
    assert module.pipeline.metrics["avgen_bench_coverage"] == 1.0
    assert not module.last_metadata["canonical_comparable"]
    assert module.last_metadata["canonical_identity_coverage"] == pytest.approx(1 / 235)


def test_only_complete_235_identity_run_is_canonical_comparable(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path, canonical=True)
    video = tmp_path / "shared.mp4"
    video.write_bytes(b"mock")
    samples = [
        Sample(
            path=video,
            is_video=True,
            metadata={
                "avgen_bench_category": prompt.category,
                "avgen_bench_content": prompt.content,
            },
        )
        for prompt in canonical_prompts()
    ]
    module = _module(tmp_path, input_mode="canonical")
    module.post_process(samples)

    assert module.last_metadata["canonical_complete_identity_coverage"]
    assert module.last_metadata["canonical_required_observations_complete"]
    assert module.last_metadata["canonical_comparable"]
    assert module.last_metadata["canonical_universe"]["total_prompts"] == 235
    assert module.last_metadata["canonical_universe"]["total_face_annotations"] == 220
    assert module.last_metadata["canonical_universe"]["face_max_ids_positive"] == 103
    assert len(module.last_metadata["canonical_universe"]["missing_face_annotations"]) == 15


def test_second_run_clears_stale_scores_and_uses_fresh_work_directory(tmp_path, monkeypatch):
    calls = _install_fakes(monkeypatch, tmp_path, canonical=True)
    face = next(
        item
        for item in canonical_face_annotations()
        if item.category == "musical_instrument_tutorial" and item.max_ids > 0
    )
    sample = _sample(
        tmp_path / "a.mp4",
        avgen_bench_category=face.category,
        avgen_bench_content=face.content,
    )
    module = _module(tmp_path, input_mode="canonical")
    module.post_process([sample])
    first_run_dir = module.last_metadata["run_dir"]
    assert sample.quality_metrics.avgen_bench_visual_score == 100.0

    def all_failed(request, interpreter, **kwargs):
        calls["requests"].append(copy.deepcopy(request))
        return {
            "schema": 1,
            "source_revision": request["source_revision"],
            "dimension": request["dimension"],
            "records": [
                {
                    "id": row["id"],
                    "status": "failed",
                    "eligible": None,
                    "raw_score": None,
                    "raw_details": {},
                    "error": "second run failure",
                }
                for row in request["records"]
            ],
            "raw_summary": {},
            "counts": {"requested": len(request["records"]), "scored": 0, "failed": 1},
        }

    import ayase.modules.avgen_bench as module_source

    runtime = module_source._load_runtime()
    runtime.execute_worker = all_failed
    module.on_execute()
    module.post_process([sample])

    assert module.last_metadata["run_dir"] != first_run_dir
    assert module.pipeline.metrics["avgen_bench_score"] is None
    assert module.pipeline.metrics["avgen_bench_coverage"] == 0.0
    for field in SAMPLE_SCORE_FIELDS.values():
        assert getattr(sample.quality_metrics, field) is None
    assert all(
        component["status"] == "failed"
        for component in sample.metadata["avgen_bench"]["components"].values()
    )


def test_all_declared_sample_scores_are_finite_zero_to_one_hundred(tmp_path, monkeypatch):
    _install_fakes(monkeypatch, tmp_path, canonical=True)
    faces = [
        item
        for item in canonical_face_annotations()
        if item.category == "musical_instrument_tutorial" and item.max_ids > 0
    ][:2]
    samples = [
        _sample(
            tmp_path / f"{index}.mp4",
            avgen_bench_category=face.category,
            avgen_bench_content=face.content,
        )
        for index, face in enumerate(faces)
    ]
    module = _module(tmp_path, input_mode="canonical")
    module.post_process(samples)
    for sample in samples:
        for field in SAMPLE_SCORE_FIELDS.values():
            value = getattr(sample.quality_metrics, field)
            assert math.isfinite(value)
            assert 0.0 <= value <= 100.0

"""Compatibility contracts for the split lip-sync implementations."""

import json
from pathlib import Path

import pytest

from ayase.models import QualityMetrics, Sample, observe_metric_writes
from ayase.pipeline import (
    ModuleRegistry,
    Pipeline,
    PipelineModule,
    instantiate_module_requests,
)


def test_quality_metrics_exposes_only_four_canonical_lip_sync_fields():
    fields = {name for name in QualityMetrics.model_fields if name.startswith("lse_")}
    assert fields == {
        "lse_c_verse",
        "lse_d_verse",
        "lse_c_syncnet",
        "lse_d_syncnet",
    }


def test_legacy_metric_input_defaults_to_verse_and_accepts_explicit_syncnet_context():
    verse = QualityMetrics.model_validate({"lse_c": 1.5, "lse_d": 2.5})
    assert verse.lse_c_verse == 1.5
    assert verse.lse_d_verse == 2.5
    assert verse.lse_c_syncnet is None

    syncnet = QualityMetrics.model_validate(
        {"lse_c": 3.5, "lse_d": 4.5},
        context={"lip_sync_protocol": "wav2lip"},
    )
    assert syncnet.lse_c_syncnet == 3.5
    assert syncnet.lse_d_syncnet == 4.5
    assert syncnet.lse_c_verse is None


def test_legacy_and_canonical_input_must_agree():
    accepted = QualityMetrics.model_validate({"lse_c": 1.0, "lse_c_verse": 1.0})
    assert accepted.lse_c_verse == 1.0
    with pytest.raises(ValueError, match="Conflicting values"):
        QualityMetrics.model_validate({"lse_c": 1.0, "lse_c_verse": 2.0})


def test_legacy_attribute_target_is_explicit_per_instance():
    metrics = QualityMetrics(lse_c_verse=1.0, lse_c_syncnet=2.0)
    with pytest.warns(DeprecationWarning):
        assert metrics.lse_c == 1.0
    metrics._set_lip_sync_legacy_protocol("wav2lip")
    with pytest.warns(DeprecationWarning):
        assert metrics.lse_c == 2.0


def test_legacy_attribute_writes_route_to_selected_canonical_fields_and_observer():
    observed = []
    verse = QualityMetrics()
    with observe_metric_writes(lambda _metrics, field: observed.append(field)):
        verse.lse_c = 1.0
        verse.lse_d = 2.0
    assert verse.lse_c_verse == 1.0 and verse.lse_d_verse == 2.0
    assert observed == ["lse_c_verse", "lse_d_verse"]
    assert verse.model_dump()["lse_c"] == 1.0

    observed.clear()
    syncnet = QualityMetrics()
    syncnet._set_lip_sync_legacy_protocol("wav2lip")
    with observe_metric_writes(lambda _metrics, field: observed.append(field)):
        syncnet.lse_c = 3.0
        syncnet.lse_d = 4.0
    assert syncnet.lse_c_syncnet == 3.0 and syncnet.lse_d_syncnet == 4.0
    assert syncnet.lse_c_verse is None and syncnet.lse_d_verse is None
    assert observed == ["lse_c_syncnet", "lse_d_syncnet"]
    assert syncnet.model_dump()["lse_c"] == 3.0

    observed.clear()
    canonical = QualityMetrics()
    with observe_metric_writes(lambda _metrics, field: observed.append(field)):
        canonical.lse_c_syncnet = 5.0
    assert canonical.model_dump()["lse_c_syncnet"] == 5.0
    assert observed == ["lse_c_syncnet"]


@pytest.mark.parametrize(
    ("protocol", "canonical_c", "canonical_d"),
    [
        ("verse_bench", "lse_c_verse", "lse_d_verse"),
        ("wav2lip", "lse_c_syncnet", "lse_d_syncnet"),
    ],
)
def test_direct_model_dumps_preserve_legacy_facade_schema(protocol, canonical_c, canonical_d):
    metrics = QualityMetrics.model_validate(
        {"lse_c": 1.25, "lse_d": 2.25},
        context={"lip_sync_protocol": protocol},
    )
    dumped = metrics.model_dump()
    assert dumped["lse_c"] == 1.25
    assert dumped["lse_d"] == 2.25
    assert canonical_c not in dumped
    assert canonical_d not in dumped
    json_metrics = json.loads(metrics.model_dump_json())
    assert json_metrics["lse_c"] == 1.25
    assert {key for key in json_metrics if key.startswith("lse_")} == {"lse_c", "lse_d"}

    sample = Sample(path=Path("clip.mp4"), is_video=True, quality_metrics=metrics)
    nested = sample.model_dump(mode="json")["quality_metrics"]
    assert nested["lse_c"] == 1.25
    assert canonical_c not in nested
    assert json.loads(sample.model_dump_json())["quality_metrics"]["lse_d"] == 2.25
    json_nested = json.loads(sample.model_dump_json())["quality_metrics"]
    assert {key for key in json_nested if key.startswith("lse_")} == {"lse_c", "lse_d"}


def test_legacy_serialization_moves_only_selected_pair_and_keeps_internals_canonical():
    metrics = QualityMetrics(
        lse_c_verse=1.0,
        lse_d_verse=2.0,
        lse_c_syncnet=3.0,
        lse_d_syncnet=4.0,
        metric_provenance={
            "lse_c_verse": "adapted",
            "lse_d_verse": "adapted",
            "lse_c_syncnet": "adapted",
            "lse_d_syncnet": "adapted",
        },
    )
    metrics._set_lip_sync_legacy_protocol("wav2lip")
    dumped = metrics.model_dump()
    assert dumped["lse_c"] == 3.0 and dumped["lse_d"] == 4.0
    assert "lse_c_syncnet" not in dumped and "lse_d_syncnet" not in dumped
    assert dumped["lse_c_verse"] == 1.0 and dumped["lse_d_verse"] == 2.0
    assert set(metrics.non_null_metrics()) == {
        "lse_c_verse",
        "lse_d_verse",
        "lse_c_syncnet",
        "lse_d_syncnet",
    }
    assert set(metrics.canonical_model_dump()) >= set(metrics.non_null_metrics())


def test_legacy_marker_survives_copy_and_contextual_sample_ingress():
    metrics = QualityMetrics(lse_c_syncnet=3.0, lse_d_syncnet=4.0)
    metrics._set_lip_sync_legacy_protocol("wav2lip")
    copied = metrics.model_copy(deep=True)
    assert copied.model_dump()["lse_c"] == 3.0

    restored = Sample.model_validate(
        {
            "path": "clip.mp4",
            "is_video": True,
            "quality_metrics": {"lse_c": 3.0, "lse_d": 4.0},
        },
        context={"lip_sync_protocol": "wav2lip"},
    )
    assert restored.quality_metrics is not None
    assert restored.quality_metrics.lse_c_syncnet == 3.0
    assert restored.model_dump(mode="json")["quality_metrics"]["lse_c"] == 3.0


def test_canonical_metrics_direct_serialization_is_unchanged():
    metrics = QualityMetrics(lse_c_syncnet=3.0, lse_d_syncnet=4.0)
    assert "lse_c" not in metrics.model_dump()
    sample = Sample(path=Path("clip.mp4"), is_video=True, quality_metrics=metrics)
    nested = sample.model_dump(mode="json")["quality_metrics"]
    assert nested["lse_c_syncnet"] == 3.0
    assert "lse_c" not in nested


def test_pipeline_state_dump_remains_canonical_for_legacy_facade_results():
    metrics = QualityMetrics(lse_c_syncnet=3.0, lse_d_syncnet=4.0)
    metrics._set_lip_sync_legacy_protocol("wav2lip")
    sample = Sample(path=Path("clip.mp4"), is_video=True, quality_metrics=metrics)
    stored = Pipeline._dump_sample_state("clip", sample)["quality_metrics"]
    assert stored["lse_c_syncnet"] == 3.0
    assert stored["lse_d_syncnet"] == 4.0
    assert "lse_c" not in stored and "lse_d" not in stored


def test_skipped_legacy_pipeline_stamps_existing_metrics_without_adding_values(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not-decoded-in-test-mode")
    module = instantiate_module_requests(
        [("lip_sync", {"test_mode": True, "allow_provenance": ["adapted"]})]
    )[0]
    pipeline = Pipeline([module])
    sample = Sample(
        path=media,
        is_video=True,
        quality_metrics=QualityMetrics(blur_score=12.0),
    )
    pipeline.start()
    try:
        result = pipeline.process_sample(sample)
    finally:
        pipeline.stop()

    assert result.quality_metrics is not None
    assert result.quality_metrics.non_null_count() == 1
    direct = result.model_dump(mode="json")["quality_metrics"]
    assert direct["blur_score"] == 12.0
    assert direct["lse_c"] is None and direct["lse_d"] is None
    assert {key for key in direct if key.startswith("lse_")} == {"lse_c", "lse_d"}
    json_direct = json.loads(result.model_dump_json())["quality_metrics"]
    assert {key for key in json_direct if key.startswith("lse_")} == {"lse_c", "lse_d"}

    cached = Pipeline._dump_sample_state(str(media), result)["quality_metrics"]
    assert {key for key in cached if key.startswith("lse_")} == {
        "lse_c_verse",
        "lse_d_verse",
        "lse_c_syncnet",
        "lse_d_syncnet",
    }
    assert result.quality_metrics.non_null_metrics() == {"blur_score": 12.0}


def test_pipeline_load_state_restores_old_wav2lip_keys_with_protocol_context(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"not-decoded-by-this-test")
    module = instantiate_module_requests(
        [("lip_sync", {"test_mode": True, "protocol": "wav2lip"})]
    )[0]
    pipeline = Pipeline([module])
    canonical_sample = Sample(
        path=media,
        is_video=True,
        quality_metrics=QualityMetrics(lse_c_syncnet=3.0, lse_d_syncnet=4.0),
    )
    key = str(media)
    state = {
        "pipeline_fingerprint": pipeline._pipeline_fingerprint(),
        "results": {
            key: {
                **canonical_sample.canonical_model_dump(mode="json"),
                "quality_metrics": {"lse_c": 3.0, "lse_d": 4.0},
            }
        },
        "cache_manifest": {key: pipeline._sample_state_manifest(canonical_sample)},
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    pipeline.load_state(state_path)

    restored = pipeline.results[key].quality_metrics
    assert restored is not None
    assert restored.lse_c_syncnet == 3.0 and restored.lse_d_syncnet == 4.0
    assert restored.lse_c_verse is None and restored.lse_d_verse is None
    assert restored.model_dump()["lse_c"] == 3.0


def test_registry_keeps_legacy_factory_out_of_canonical_module_listing():
    ModuleRegistry.discover_modules()
    listed = ModuleRegistry.list_modules(packaged_only=True)
    assert "lip_sync_verse" in listed
    assert "lip_sync_syncnet" in listed
    assert "lip_sync" not in listed
    assert ModuleRegistry.get_module("lip_sync") is not None


def test_legacy_factory_routes_protocol_and_marks_compatibility_mode():
    from ayase.modules.lip_sync import (
        LipSyncModule,
        LipSyncSyncNetModule,
        LipSyncVerseModule,
    )

    verse = LipSyncModule(config={"test_mode": True})
    syncnet = LipSyncModule(config={"test_mode": True, "protocol": "wav2lip"})
    assert isinstance(verse, LipSyncVerseModule)
    assert isinstance(syncnet, LipSyncSyncNetModule)
    assert verse._requested_module_name == "lip_sync"
    assert syncnet._legacy_lip_sync_protocol == "wav2lip"


def test_alias_and_canonical_same_config_execute_once_and_conflicts_fail():
    common = {"test_mode": True, "allow_provenance": ["adapted"]}
    modules = instantiate_module_requests(
        [("lip_sync", dict(common)), ("lip_sync_verse", dict(common))]
    )
    assert [module.name for module in modules] == ["lip_sync_verse"]

    with pytest.raises(ValueError, match="Conflicting configurations"):
        instantiate_module_requests(
            [
                ("lip_sync", dict(common)),
                ("lip_sync_verse", {**common, "window_size": 99}),
            ]
        )


def test_unrelated_duplicate_module_instances_and_configs_are_preserved():
    class UnrelatedProbe(PipelineModule):
        name = "unnamed_module"

        def process(self, sample):
            return sample

    first = UnrelatedProbe({"slot": 1})
    second = UnrelatedProbe({"slot": 2})
    pipeline = Pipeline([first, second])
    assert pipeline.modules == [first, second]
    assert pipeline.modules[0].config["slot"] == 1
    assert pipeline.modules[1].config["slot"] == 2


def test_legacy_hook_name_and_export_follow_resolved_module_without_duplicate_fields():
    module = instantiate_module_requests([("lip_sync", {"test_mode": True})])[0]
    pipeline = Pipeline([module])

    def callback(sample):
        return sample

    pipeline.add_hook("lip_sync", before=callback)
    assert pipeline._hooks[module.name]["before"] is callback

    sample = Sample(
        path=Path("talking-head.mp4"),
        is_video=True,
        quality_metrics=QualityMetrics(
            lse_c_verse=1.25,
            lse_d_verse=2.25,
            metric_provenance={"lse_c_verse": "adapted", "lse_d_verse": "adapted"},
            metric_backends={"lip_sync_verse": "verse"},
        ),
    )
    dumped = pipeline.dump_sample(sample)["quality_metrics"]
    assert dumped["lse_c"] == 1.25
    assert dumped["lse_d"] == 2.25
    assert "lse_c_verse" not in dumped
    assert "lse_d_verse" not in dumped
    assert dumped["metric_provenance"] == {"lse_c": "adapted", "lse_d": "adapted"}
    assert dumped["metric_backends"] == {"lip_sync": "verse"}

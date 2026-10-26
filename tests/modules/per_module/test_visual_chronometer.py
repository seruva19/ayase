"""Pending physical-FPS estimation stays distinct from measured motion."""


def test_visual_chronometer_is_unimplemented_and_inert(video_sample):
    from ayase.modules.visual_chronometer import VisualChronometerModule

    module = VisualChronometerModule({"test_mode": True})
    metadata = module.get_metadata()
    assert metadata["implementation_status"] == "unimplemented"
    assert "demo checkpoint exists" in metadata["implementation_notes"]
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None or (
        video_sample.quality_metrics.visual_chronometer_phyfps is None
    )


def test_visual_chronometer_stays_out_of_available_metrics():
    from ayase.metric_catalog import build_metric_catalog
    from ayase.metrics_doc import compute_external_backend_partition
    from ayase.pipeline import ModuleRegistry
    from ayase.modules.visual_chronometer import VisualChronometerModule

    assert "visual_chronometer" not in ModuleRegistry.list_modules(include_external_backends=False)
    modules, fields = compute_external_backend_partition(["visual_chronometer"])
    assert modules == {"visual_chronometer"}
    assert fields == {"visual_chronometer_phyfps"}
    module = build_metric_catalog(["visual_chronometer"]).module("visual_chronometer")
    assert module.implementation_status == "unimplemented"
    assert module.metrics[0].direction == "—"


def test_visual_chronometer_never_mounts_in_pipeline(monkeypatch, video_sample):
    from ayase.pipeline import Pipeline
    from ayase.modules.visual_chronometer import VisualChronometerModule

    module = VisualChronometerModule()

    def unexpected_mount():
        raise AssertionError("unimplemented metrics must not attempt setup or downloads")

    monkeypatch.setattr(module, "on_mount", unexpected_mount)
    pipeline = Pipeline([module])
    pipeline.start()
    assert pipeline.process_sample(video_sample) is video_sample
    assert pipeline._availability_excluded[module.name] == "implementation_unavailable"
    pipeline.stop()

"""Unavailable CMMS is discoverable without pretending to compute quality."""


def test_cmms_declares_missing_head_and_never_fabricates_score(image_sample):
    from ayase.modules.cmms import CMMSModule

    module = CMMSModule({"test_mode": True})
    metadata = module.get_metadata()
    assert metadata["implementation_status"] == "unimplemented"
    assert "regression" in metadata["implementation_notes"]
    assert "cmms_score" in metadata["output_fields"]
    assert module.process(image_sample) is image_sample
    assert image_sample.quality_metrics is None or image_sample.quality_metrics.cmms_score is None


def test_cmms_is_excluded_from_delivered_registry_and_fields():
    from ayase.metrics_doc import compute_external_backend_partition
    from ayase.pipeline import ModuleRegistry
    from ayase.modules.cmms import CMMSModule

    assert ModuleRegistry.get_module("cmms") is CMMSModule
    assert "cmms" not in ModuleRegistry.list_modules(include_external_backends=False)
    modules, fields = compute_external_backend_partition(["cmms"])
    assert modules == {"cmms"}
    assert fields == {"cmms_score"}

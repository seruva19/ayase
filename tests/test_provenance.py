"""Provenance contract tests.

Ayase metrics must correspond to a published definition, standard, or
official implementation.  Every packaged module declares a ``provenance``
field map ("published" | "adapted" | "own" | "utility") plus ``sources`` and
``deviations``; these tests keep the declarations honest:

* classes are one of the four allowed values and keys name real fields;
* an ``own`` field must not use a published metric name
  (``published_names.txt``);
* ``adapted`` fields must say how they deviate (``deviations``);
* ``published``/``adapted`` fields must cite their source (``sources``);
* the pipeline excludes own/adapted-only modules unless
  ``allow_provenance`` opts in, while utility side-channels keep service
  modules enabled;
* ``QualityMetrics.metric_provenance`` is stamped per written field and
  excluded from metric counts.
"""

import pytest

from ayase.models import DatasetStats, QualityMetrics, Sample
from ayase.pipeline import ModuleRegistry, Pipeline, PipelineModule

VALID_CLASSES = {"published", "adapted", "own", "utility"}
PUBLISHED_NAMES_PATH = (
    __import__("pathlib").Path(__file__).resolve().parent.parent
    / "src" / "ayase" / "published_names.txt"
)


@pytest.fixture(scope="module")
def registry():
    ModuleRegistry.discover_modules()
    return ModuleRegistry.list_modules()


@pytest.fixture(scope="module")
def all_field_names():
    return (
        set(QualityMetrics.model_fields)
        | set(Sample.model_fields)
        | set(DatasetStats.model_fields)
    )


@pytest.fixture(scope="module")
def published_names():
    return {
        line.strip()
        for line in PUBLISHED_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


def _is_packaged(cls) -> bool:
    """Only modules shipped under ``src/ayase/modules/`` are audited — test
    fixtures and external plugins are not required to declare provenance."""
    try:
        import inspect

        path = inspect.getfile(cls).replace("\\", "/")
    except (TypeError, OSError):
        return False
    return "/ayase/modules/" in path


def _classes(registry):
    out = {}
    for name in registry:
        cls = ModuleRegistry.get_module(name)
        if cls is None or not _is_packaged(cls):
            continue
        try:
            prov = cls.field_provenance()
        except Exception:
            prov = {}
        if not prov and isinstance(getattr(cls, "provenance", None), str):
            prov = {"<module>": cls.provenance}
        out[name] = (cls, prov)
    return out


def test_every_module_declares_provenance(registry):
    missing = []
    for name in registry:
        cls = ModuleRegistry.get_module(name)
        if cls is None or not _is_packaged(cls):
            continue
        prov = getattr(cls, "provenance", None)
        if not prov:
            missing.append(name)
    assert not missing, f"modules without provenance: {missing}"


def test_provenance_classes_valid(registry):
    for name, (cls, prov) in _classes(registry).items():
        for field, value in prov.items():
            assert value in VALID_CLASSES, (
                f"{name}.{field}: unknown provenance class {value!r}"
            )


def test_provenance_keys_are_real_fields(registry, all_field_names):
    for name, (cls, prov) in _classes(registry).items():
        for field in prov:
            assert field == "<module>" or field in all_field_names, (
                f"{name}: provenance key {field!r} is not a declared field"
            )


def test_every_inferred_output_has_provenance(registry):
    missing = []
    for name, (cls, prov) in _classes(registry).items():
        meta = cls.get_metadata()
        outputs = set(meta["output_fields"]) | set(meta["dataset_output_fields"])
        missing.extend(f"{name}.{field}" for field in sorted(outputs - set(prov)))
    assert not missing, f"output fields without provenance: {missing}"


def test_own_fields_do_not_use_published_names(registry, published_names):
    offenders = set()
    for name, (cls, prov) in _classes(registry).items():
        for field, value in prov.items():
            if value == "own" and field in published_names:
                offenders.add(f"{name}.{field}")
    assert not offenders, (
        "own-class fields must not use published metric names: "
        f"{sorted(offenders)}"
    )


def test_adapted_fields_have_deviations(registry):
    missing = []
    for name in registry:
        cls = ModuleRegistry.get_module(name)
        if cls is None or not _is_packaged(cls):
            continue
        prov = cls.field_provenance()
        dev = cls._resolve_field_map(
            "deviations", list(cls.get_metadata().get("output_fields", {}))
        )
        for field, value in prov.items():
            if value == "adapted" and not dev.get(field):
                missing.append(f"{name}.{field}")
    assert not missing, f"adapted fields without deviations: {missing}"


def test_published_and_adapted_fields_have_sources(registry):
    missing = []
    for name in registry:
        cls = ModuleRegistry.get_module(name)
        if cls is None or not _is_packaged(cls):
            continue
        prov = cls.field_provenance()
        src = cls._resolve_field_map(
            "sources", list(cls.get_metadata().get("output_fields", {}))
        )
        for field, value in prov.items():
            if value in ("published", "adapted") and not src.get(field):
                missing.append(f"{name}.{field}")
    assert not missing, f"published/adapted fields without sources: {missing}"


def test_published_and_adapted_sources_are_traceable(registry):
    untraceable = []
    for name, (cls, prov) in _classes(registry).items():
        src = cls._resolve_field_map(
            "sources", list(cls.get_metadata().get("output_fields", {}))
        )
        for field, value in prov.items():
            source = str(src.get(field, "")).lower()
            if value in ("published", "adapted") and not (
                "http://" in source or "https://" in source or "doi:" in source
            ):
                untraceable.append(f"{name}.{field}")
    assert not untraceable, (
        "published/adapted sources need a URL or DOI: " f"{untraceable}"
    )


def test_metadata_exposes_provenance(registry):
    for name in ("vmaf", "fvd", "facesim", "object_detection"):
        cls = ModuleRegistry.get_module(name)
        if cls is None:
            continue
        meta = cls.get_metadata()
        assert "provenance" in meta and meta["provenance"], name
        assert "sources" in meta
        assert "deviations" in meta


class _OwnModule(PipelineModule):
    name = "test_own_only"
    description = "test"
    provenance = {"test_own_score": "own"}

    def process(self, sample):
        return sample


class _PublishedModule(PipelineModule):
    name = "test_published_only"
    description = "test"
    provenance = "published"

    def process(self, sample):
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.vmaf = 50.0
        return sample


class _UtilitySideChannel(PipelineModule):
    name = "test_utility_side_channel"
    description = "test"
    provenance = {"test_own_score": "own", "detections": "utility"}

    def process(self, sample):
        return sample


class _MixedModule(PipelineModule):
    name = "test_mixed"
    description = "test"
    provenance = {"vmaf": "published", "brisque": "own"}

    def process(self, sample):
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.vmaf = 50.0
        sample.quality_metrics.brisque = 25.0
        return sample


class _MixedDatasetModule(PipelineModule):
    name = "test_mixed_dataset"
    description = "test"
    provenance = {"fvd": "published", "msswd": "own"}

    def process(self, sample):
        return sample


class _SameValueModule(PipelineModule):
    name = "test_same_value"
    provenance = {"vmaf": "published"}

    def process(self, sample):
        sample.quality_metrics.vmaf = 50.0
        return sample


class _NoWritePublishedModule(PipelineModule):
    name = "test_no_write_published"
    provenance = {"vmaf": "published"}

    def process(self, sample):
        return sample


class _ClearValueModule(PipelineModule):
    name = "test_clear_value"
    provenance = {"vmaf": "published"}

    def process(self, sample):
        sample.quality_metrics.vmaf = None
        return sample


class _ExternalBackendModule(PipelineModule):
    name = "test_external_backend"
    provenance = {"vmaf": "published"}
    requires_external_backend = True

    def process(self, sample):
        raise AssertionError("external-backend module must not execute")


class _AvailableExternalBackendModule(_ExternalBackendModule):
    name = "test_available_external_backend"

    def setup(self):
        self._backend = "custom"

    def process(self, sample):
        sample.quality_metrics = QualityMetrics(vmaf=33.0)
        return sample


class _FailingWriterModule(PipelineModule):
    name = "test_failing_writer"
    provenance = {"vmaf": "published"}

    def process(self, sample):
        sample.quality_metrics.vmaf = 99.0
        raise RuntimeError("failed after write")


class _FailingBatchWriterModule(_FailingWriterModule):
    name = "test_failing_batch_writer"

    def process_batch(self, samples):
        for sample in samples:
            sample.quality_metrics.vmaf = 99.0
        raise RuntimeError("batch failed after write")


def test_default_pipeline_excludes_own_and_adapted():
    p = Pipeline([_OwnModule(), _PublishedModule()])
    assert "test_own_only" in p._provenance_excluded
    assert "test_published_only" not in p._provenance_excluded


def test_allow_provenance_override():
    p = Pipeline([_OwnModule()], allow_provenance=["own"])
    assert not p._provenance_excluded
    p = Pipeline([_OwnModule()], allow_provenance="own")
    assert not p._provenance_excluded


def test_utility_side_channel_keeps_module_enabled():
    p = Pipeline([_UtilitySideChannel()])
    assert not p._provenance_excluded


def test_per_module_config_allow_provenance():
    m = _OwnModule(config={"allow_provenance": ["own"]})
    p = Pipeline([m])
    assert not p._provenance_excluded


def test_per_module_opt_in_does_not_enable_other_modules():
    opted_in = _OwnModule(config={"allow_provenance": ["own"]})
    other = _OwnModule()
    other.name = "test_other_own"
    p = Pipeline([opted_in, other])
    assert "test_own_only" not in p._provenance_excluded
    assert "test_other_own" in p._provenance_excluded


def test_unknown_provenance_class_rejected():
    with pytest.raises(ValueError, match="Unknown provenance classes"):
        Pipeline([_PublishedModule()], allow_provenance=["publshed"])


def test_metric_provenance_stamped(tmp_path):
    p = Pipeline([_PublishedModule()])
    p.start()  # mounts modules; process_sample skips unmounted ones
    try:
        sample = Sample(path=tmp_path / "x.png", is_video=False)
        out = p.process_sample(sample)
    finally:
        p.stop()
    assert out.quality_metrics.metric_provenance.get("vmaf") == "published"
    # bookkeeping field is not itself counted as a metric
    assert "metric_provenance" not in out.quality_metrics.non_null_metrics()
    assert "metric_provenance" not in out.quality_metrics.metric_count_fields() \
        if hasattr(out.quality_metrics, "metric_count_fields") else True


def test_metric_provenance_not_in_non_null():
    qm = QualityMetrics(vmaf=50.0)
    qm.metric_provenance["vmaf"] = "published"
    metrics = qm.non_null_metrics()
    assert "vmaf" in metrics
    assert "metric_provenance" not in metrics


def test_unwritten_existing_field_is_not_stamped(tmp_path):
    p = Pipeline([_PublishedModule()])
    sample = Sample(
        path=tmp_path / "x.png",
        is_video=False,
        quality_metrics=QualityMetrics(vmaf=50.0),
    )
    before = p._metric_state(sample)
    p._persist_provenance(sample, _PublishedModule(), before)
    assert "vmaf" not in sample.quality_metrics.metric_provenance


def test_mixed_module_drops_disallowed_fields(tmp_path):
    module = _MixedModule()
    p = Pipeline([module])
    module._mounted = True
    sample = p.process_sample(Sample(path=tmp_path / "x.png", is_video=False))
    assert sample.quality_metrics.vmaf == 50.0
    assert sample.quality_metrics.brisque is None
    assert sample.quality_metrics.metric_provenance == {"vmaf": "published"}


def test_mixed_dataset_module_drops_disallowed_fields():
    module = _MixedDatasetModule()
    p = Pipeline([module])
    p._active_dataset_module = module
    try:
        p.add_dataset_metric("fvd", 1.0)
        p.add_dataset_metric("msswd", 2.0)
    finally:
        p._active_dataset_module = None
    assert p.stats.fvd == 1.0
    assert p.stats.msswd is None
    assert p.stats.metric_provenance == {"fvd": "published"}


def test_same_value_assignment_updates_provenance(tmp_path):
    module = _SameValueModule()
    module._mounted = True
    p = Pipeline([module])
    sample = Sample(
        path=tmp_path / "x.png",
        is_video=False,
        quality_metrics=QualityMetrics(
            vmaf=50.0, metric_provenance={"vmaf": "adapted"}
        ),
    )
    out = p.process_sample(sample)
    assert out.quality_metrics.metric_provenance["vmaf"] == "published"


def test_clearing_value_clears_stale_provenance(tmp_path):
    module = _ClearValueModule()
    module._mounted = True
    p = Pipeline([module])
    sample = Sample(
        path=tmp_path / "x.png",
        is_video=False,
        quality_metrics=QualityMetrics(
            vmaf=50.0, metric_provenance={"vmaf": "published"}
        ),
    )
    out = p.process_sample(sample)
    assert out.quality_metrics.vmaf is None
    assert "vmaf" not in out.quality_metrics.metric_provenance


def test_before_hook_write_is_not_attributed_to_module(tmp_path):
    module = _NoWritePublishedModule()
    module._mounted = True
    p = Pipeline([module])

    def before(sample):
        sample.quality_metrics = QualityMetrics(vmaf=12.0)
        return sample

    p.add_hook(module.name, before=before)
    out = p.process_sample(Sample(path=tmp_path / "x.png", is_video=False))
    assert out.quality_metrics.vmaf == 12.0
    assert "vmaf" not in out.quality_metrics.metric_provenance


def test_external_backend_module_is_unavailable_not_complete(tmp_path):
    module = _ExternalBackendModule()
    p = Pipeline([module])
    p.start()
    out = p.process_sample(Sample(path=tmp_path / "x.png", is_video=False))
    assert out.quality_metrics is None
    status = p.get_run_status()
    assert status["complete"] is False
    assert status["availability_excluded"] == {
        "test_external_backend": "external_backend_unavailable"
    }


def test_configured_external_backend_can_run(tmp_path):
    module = _AvailableExternalBackendModule()
    module.setup()
    p = Pipeline([module])
    p.start()
    out = p.process_sample(Sample(path=tmp_path / "x.png", is_video=False))
    assert out.quality_metrics.vmaf == 33.0
    assert p.get_run_status()["complete"] is True


def test_metadata_cache_invalidates_declared_fields():
    original = _PublishedModule.description
    first = _PublishedModule.get_metadata()
    try:
        _PublishedModule.description = "changed description"
        second = _PublishedModule.get_metadata()
    finally:
        _PublishedModule.description = original
    assert first["description"] != second["description"]


def test_failed_writer_restores_value_and_provenance(tmp_path):
    module = _FailingWriterModule()
    module._mounted = True
    p = Pipeline([module])
    sample = Sample(
        path=tmp_path / "x.png",
        is_video=False,
        quality_metrics=QualityMetrics(
            vmaf=12.0, metric_provenance={"vmaf": "adapted"}
        ),
    )
    out = p.process_sample(sample)
    assert out.quality_metrics.vmaf == 12.0
    assert out.quality_metrics.metric_provenance["vmaf"] == "adapted"


def test_failed_batch_writer_restores_value_and_provenance(tmp_path):
    module = _FailingBatchWriterModule()
    module._mounted = True
    p = Pipeline([module])
    sample = Sample(
        path=tmp_path / "x.png",
        is_video=False,
        quality_metrics=QualityMetrics(
            vmaf=12.0, metric_provenance={"vmaf": "adapted"}
        ),
    )
    out = p._process_module_batch(module, [sample])[0]
    assert out.quality_metrics.vmaf == 12.0
    assert out.quality_metrics.metric_provenance["vmaf"] == "adapted"


def test_after_hook_failure_keeps_successful_lineage(tmp_path):
    module = _PublishedModule()
    module._mounted = True
    p = Pipeline([module])

    def after(sample):
        raise RuntimeError("after failed")

    p.add_hook(module.name, after=after)
    out = p.process_sample(Sample(path=tmp_path / "x.png", is_video=False))
    assert out.quality_metrics.vmaf == 50.0
    assert out.quality_metrics.metric_provenance["vmaf"] == "published"


def test_after_hook_metric_is_not_attributed_to_module(tmp_path):
    module = _NoWritePublishedModule()
    module._mounted = True
    p = Pipeline([module])

    def after(sample):
        sample.quality_metrics = QualityMetrics(vmaf=12.0)
        return sample

    p.add_hook(module.name, after=after)
    out = p.process_sample(Sample(path=tmp_path / "x.png", is_video=False))
    assert out.quality_metrics.vmaf == 12.0
    assert "vmaf" not in out.quality_metrics.metric_provenance


def test_batch_after_hook_failure_keeps_successful_lineage(tmp_path):
    module = _PublishedModule()
    module._mounted = True
    p = Pipeline([module])

    def after(sample):
        raise RuntimeError("after failed")

    p.add_hook(module.name, after=after)
    sample = Sample(path=tmp_path / "x.png", is_video=False)
    out = p._process_module_batch(module, [sample])[0]
    assert out.quality_metrics.vmaf == 50.0
    assert out.quality_metrics.metric_provenance["vmaf"] == "published"


def test_batch_after_hook_metric_is_not_attributed_to_module(tmp_path):
    module = _NoWritePublishedModule()
    module._mounted = True
    p = Pipeline([module])

    def after(sample):
        sample.quality_metrics = QualityMetrics(vmaf=12.0)
        return sample

    p.add_hook(module.name, after=after)
    sample = Sample(path=tmp_path / "x.png", is_video=False)
    out = p._process_module_batch(module, [sample])[0]
    assert out.quality_metrics.vmaf == 12.0
    assert "vmaf" not in out.quality_metrics.metric_provenance

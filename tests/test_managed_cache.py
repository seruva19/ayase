"""Cache fingerprints for isolated metric implementations."""

from pathlib import Path

from ayase.backend_dependencies import BACKEND_DEPENDENCIES, BackendDependency
from ayase.pipeline import Pipeline, PipelineModule


class _ManagedCacheProbe(PipelineModule):
    name = "managed_cache_probe"

    def process(self, sample):
        return sample


class _ImplementationSourceProbe(PipelineModule):
    name = "implementation_source_cache_probe"
    implementation_sources = ("helper.py",)

    def process(self, sample):
        return sample


def test_managed_spec_changes_digest_without_provisioning(monkeypatch):
    import ayase.managed_native as managed_native
    import ayase.managed_runtime as managed_runtime

    def fail_provision(*args, **kwargs):
        raise AssertionError("cache fingerprinting must not provision an environment")

    monkeypatch.setattr(managed_runtime, "ensure_managed_environment", fail_provision)
    monkeypatch.setattr(managed_native, "ensure_managed_native_environment", fail_provision)
    module = _ManagedCacheProbe({"test_mode": True})
    first_spec = BackendDependency(
        module.name,
        ("example-backend==1.0",),
        python_version="3.11",
        pip_options=("--index-url", "https://example.invalid/simple"),
        conda_packages=("example-native=1",),
    )
    monkeypatch.setitem(BACKEND_DEPENDENCIES, module.name, first_spec)

    first = Pipeline._module_source_digest(module)
    assert Pipeline._module_source_digest(module) == first

    monkeypatch.setitem(
        BACKEND_DEPENDENCIES,
        module.name,
        BackendDependency(
            module.name,
            ("example-backend==2.0",),
            python_version="3.11",
            pip_options=first_spec.pip_options,
            conda_packages=first_spec.conda_packages,
        ),
    )
    assert Pipeline._module_source_digest(module) != first


def test_implementation_source_change_invalidates_digest(tmp_path, monkeypatch):
    import ayase.pipeline as pipeline_module

    module_source = tmp_path / "metric.py"
    helper_source = tmp_path / "helper.py"
    module_source.write_text("# metric implementation\n", encoding="utf-8")
    helper_source.write_text("VALUE = 1\n", encoding="utf-8")
    original_getfile = pipeline_module.inspect.getfile

    def getfile(subject):
        if subject is _ImplementationSourceProbe:
            return str(module_source)
        return original_getfile(subject)

    monkeypatch.setattr(pipeline_module.inspect, "getfile", getfile)
    module = _ImplementationSourceProbe({"test_mode": True})
    first = Pipeline._module_source_digest(module)
    assert Pipeline._module_source_digest(module) == first

    helper_source.write_text("VALUE = 2\n", encoding="utf-8")
    assert Pipeline._module_source_digest(module) != first

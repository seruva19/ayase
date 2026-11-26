from weakref import ref

from ayase.config import AyaseConfig
from ayase.models import Sample
from ayase.pipeline import AyasePipeline, ModuleRegistry, PipelineModule
from ayase.runtime import shared_model_weights, shared_runtime_resource


def test_facade_reuses_only_explicit_weights_and_keeps_run_state_isolated(tmp_path, monkeypatch):
    loaded = []
    transient = []

    class Resource:
        pass

    class Probe(PipelineModule):
        name = "test_weight_pool_probe"

        def on_mount(self):
            self.weights = shared_model_weights(self, ("test-weights",), lambda: new(loaded))
            self.temporary = shared_runtime_resource(
                self, ("test-temporary",), lambda: new(transient)
            )
            self.calls = 0
            self._mounted = True

        def process(self, sample):
            self.calls += 1
            sample.metadata["calls"] = self.calls
            return sample

        def teardown(self):
            self.weights = None
            self.temporary = None

    def new(records):
        resource = Resource()
        records.append(ref(resource))
        return resource

    monkeypatch.setattr(
        ModuleRegistry, "_modules", {**ModuleRegistry._modules, Probe.name: Probe}
    )
    facade = AyasePipeline(
        config=AyaseConfig(), modules=[Probe.name], reuse_model_weights=True
    )
    hooks = []

    def before(sample):
        hooks.append(sample.path)
        return sample

    facade.pipeline.add_hook(Probe.name, before=before)
    for idx in range(2):
        path = tmp_path / f"{idx}.png"
        results = facade.run(tmp_path, samples=[Sample(path=path, is_video=False)])
        assert list(results) == [str(path)]
        assert results[str(path)].metadata["calls"] == 1
        assert facade.pipeline.modules[0].weights is None
    assert len(hooks) == 2
    assert len(loaded) == 1
    assert loaded[0]() is not None
    assert len(transient) == 2

    facade.release_model_weights()
    assert loaded[0]() is None
    facade.run(tmp_path, samples=[])
    assert len(loaded) == 2

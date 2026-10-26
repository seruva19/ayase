"""Offline regressions for executable source evidence in metric documentation."""

from ayase.metrics_doc import (
    _detect_dataset_fields_written,
    _detect_fields_read,
    _detect_fields_written,
    _detect_gpu,
    _detect_hf_models,
    _detect_source_links,
    _get_source,
)
from ayase.pipeline import PipelineModule


def test_static_health_recognizes_caption_output_but_not_empty_metric_container():
    from ayase.metrics_doc import _static_checks

    meta = {"name": "caption_fixture", "output_fields": {}}
    assert not _static_checks('sample.caption = Caption(text="hello")', meta)
    assert "no output fields and no validation issues" in _static_checks(
        "sample.quality_metrics = QualityMetrics()", meta
    )


def test_transcription_module_is_not_reported_as_outputless():
    from ayase.metrics_doc import _static_checks
    from ayase.modules.asr_transcribe import ASRTranscribeModule

    assert not _static_checks(
        _get_source(ASRTranscribeModule), ASRTranscribeModule.get_metadata(), ASRTranscribeModule
    )


def _store_motion(sample):
    sample.quality_metrics.motion_score = 1.0


class _SourceHints(PipelineModule):
    """Prose: sample.quality_metrics.vmaf = 0; sample.reference_path; sample.caption.text."""

    def process(self, sample):
        # sample.quality_metrics.motion_score = 99
        if sample.quality_metrics.blur_score == 0:
            return sample
        sample.quality_metrics.brightness = 1.0
        return sample


class _SiblingWithOtherInputs(PipelineModule):
    def process(self, sample):
        if not sample.is_video or sample.reference_path is None or sample.caption is None:
            return sample
        return sample


class _ParentWithHelper(PipelineModule):
    def process(self, sample):
        self._store(sample)
        return sample

    def _store(self, sample):
        sample.quality_metrics.vmaf = 50.0


class _InheritedHelper(_ParentWithHelper):
    pass


class _OverrideHelper(_ParentWithHelper):
    def _store(self, sample):
        _store_motion(sample)


class _OverrideProcess(_ParentWithHelper):
    def process(self, sample):
        sample.quality_metrics.brightness = 1
        return sample


class _AugmentProcess(_ParentWithHelper):
    def process(self, sample):
        sample = super().process(sample)
        sample.quality_metrics.brightness = 1
        return sample


_HF_MODEL_ID = "openai/clip-vit-base-patch32"


class _GlobalModelDefault(PipelineModule):
    def setup(self):
        self.model = self.loader.from_pretrained(_HF_MODEL_ID)

    def process(self, sample):
        return sample


class _DeclaredInputs(_SiblingWithOtherInputs):
    input_type = "img +ref"


class _InheritedDynamicField(PipelineModule):
    metric_field = "vmaf"

    def process(self, sample):
        setattr(sample.quality_metrics, self.metric_field, 1.0)
        return sample


class _DynamicChild(_InheritedDynamicField):
    metric_field = "motion_score"


def test_metric_read_and_write_contexts_ignore_prose_and_comparisons():
    source = '''
def process(sample):
    """sample.quality_metrics.vmaf = 0"""
    # sample.quality_metrics.motion_score = 1
    sample.quality_metrics.blur_score = 1
    if sample.quality_metrics.brightness_score == 1:
        sample.quality_metrics.contrast_score += 1
    return sample
'''
    assert _detect_fields_written(source) == {"blur_score", "contrast_score"}
    assert _detect_fields_read(source) == {"brightness_score", "contrast_score"}


def test_metric_aliases_setattr_and_getattr_are_scoped_to_function():
    source = """
def process(sample):
    metrics = sample.quality_metrics
    other = metrics
    other.blur_score = getattr(metrics, "motion_score", 0)
    setattr(metrics, "vmaf", 3)
    metrics = unrelated
    metrics.contrast_score = 7
def unrelated_function():
    other.brightness_score = 1
"""
    assert _detect_fields_written(source) == {"blur_score", "vmaf"}
    assert _detect_fields_read(source) == {"motion_score"}


def test_metric_alias_branches_start_independently_and_join_possible_bindings():
    leaked_alias = """
def process(sample, flag):
    if flag:
        qm = sample.quality_metrics
    else:
        qm.blur_score = 1
"""
    possible_alias = """
def process(sample, flag):
    if flag:
        qm = sample.quality_metrics
    else:
        qm = unrelated
    qm.blur_score = 1
"""
    assert _detect_fields_written(leaked_alias) == set()
    assert _detect_fields_written(possible_alias) == {"blur_score"}


def test_declared_fields_are_code_not_examples():
    source = '''
"""metric_field = "vmaf"""
# metric_field_name = "blur_score"
metric_field_name: str = "motion_score"
'''
    assert _detect_fields_written(source) == {"motion_score"}


def test_dataset_writes_ignore_examples():
    source = '''
def post_process(self, samples):
    """self.pipeline.add_dataset_metric("fake", 1)"""
    self.pipeline.add_dataset_metric("avgen_bench_score", 1)
'''
    assert _detect_dataset_fields_written(source) == {"avgen_bench_score"}


def test_metadata_ignores_siblings_comments_and_comparison_reads():
    metadata = _SourceHints.get_metadata()
    assert set(metadata["output_fields"]) == {"brightness"}
    assert metadata["input_type"] == "img/vid"
    assert _SiblingWithOtherInputs.infer_input_type() == "vid +ref +cap"


def test_explicit_media_type_retains_consumed_input_hints_without_duplicates():
    assert _DeclaredInputs.infer_input_type() == "img +ref +cap"
    assert _DeclaredInputs.get_metadata()["input_type"] == "img +ref +cap"


def test_effective_inherited_helpers_and_local_free_functions_are_included():
    assert set(_InheritedHelper.get_metadata()["output_fields"]) == {"vmaf"}
    assert set(_OverrideHelper.get_metadata()["output_fields"]) == {"motion_score"}
    assert _detect_fields_written(_get_source(_OverrideHelper)) == {"motion_score"}
    assert set(_OverrideProcess.get_metadata()["output_fields"]) == {"brightness"}
    assert set(_AugmentProcess.get_metadata()["output_fields"]) == {"brightness", "vmaf"}


def test_referenced_literal_globals_are_available_to_model_loader_analysis():
    assert _detect_hf_models(_get_source(_GlobalModelDefault)) == [_HF_MODEL_ID]


def test_source_and_metadata_caches_invalidate_replaced_methods(monkeypatch):
    assert set(_SourceHints.get_metadata()["output_fields"]) == {"brightness"}
    monkeypatch.setattr(_SourceHints, "process", _ParentWithHelper.process)
    monkeypatch.setattr(_SourceHints, "_store", _ParentWithHelper._store, raising=False)
    assert set(_SourceHints.get_metadata()["output_fields"]) == {"vmaf"}


def test_resolved_dynamic_metric_field_overrides_parent_declaration():
    assert set(_DynamicChild.get_metadata()["output_fields"]) == {"motion_score"}


def test_real_compatibility_alias_retains_inherited_metric_writers():
    from ayase.modules.basic import BasicCompatModule, BasicQualityModule

    expected = set(BasicQualityModule.get_metadata()["output_fields"])
    assert "blur_score" in expected
    assert set(BasicCompatModule.get_metadata()["output_fields"]) == expected
    assert "blur_score" in _detect_fields_written(_get_source(BasicCompatModule))


def test_real_explicit_media_types_are_preserved():
    from ayase.modules.cmms import CMMSModule
    from ayase.modules.i2i_learned import I2ILearnedModule

    assert CMMSModule.infer_input_type() == CMMSModule.input_type
    assert I2ILearnedModule.infer_input_type().startswith("img")
    assert I2ILearnedModule.infer_input_type().count("+ref") == 1


def test_real_audio_and_audio_visual_inputs_keep_media_kind():
    from ayase.modules.asr_wer import ASRWERModule
    from ayase.modules.beat_consistency import BeatConsistencyModule
    from ayase.modules.clap_score import LAIONCLAPScoreModule

    assert ASRWERModule.infer_input_type() == "audio +cap"
    assert BeatConsistencyModule.infer_input_type().startswith("vid")
    assert LAIONCLAPScoreModule.infer_input_type() == "audio +cap"


def test_real_reference_collection_config_is_an_input_hint():
    from ayase.modules.chd import CHDModule

    assert CHDModule.infer_input_type() == "image +ref"


def test_hf_links_require_loader_evidence_or_explicit_model_declaration():
    source = '''
"""Example: "fictional/not-a-model"""
path = "checkpoints/encoder"
model = Model.from_pretrained("openai/clip-vit-base-patch32")
'''
    assert _detect_hf_models(source) == ["openai/clip-vit-base-patch32"]
    assert "fictional" not in _detect_source_links(source, _SourceHints)
    assert "checkpoints/encoder" not in _detect_source_links(source, _SourceHints)


def test_declared_hf_model_uses_explicit_url_without_a_loader():
    class Declared:
        models = [
            {
                "id": "example/weights",
                "type": "huggingface",
                "url": "https://huggingface.co/example/weights/tree/release",
            }
        ]

    assert _detect_hf_models("", Declared) == ["example/weights"]
    assert Declared.models[0]["url"] in _detect_source_links("", Declared)


def test_generic_cpu_device_and_cuda_prose_are_not_gpu_evidence():
    assert not _detect_gpu('"""A cuda implementation could be added."""\nmodel.to(device)')
    assert not _detect_gpu("model.to(self._device)")
    assert _detect_gpu('device = "cuda" if torch.cuda.is_available() else "cpu"')

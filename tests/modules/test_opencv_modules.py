def test_ti_si_basics():
    from ayase.modules.ti_si import TISIModule
    from .conftest import _test_module_basics

    _test_module_basics(TISIModule, "ti_si")


def test_ti_si_image(image_sample):
    from ayase.modules.ti_si import TISIModule

    m = TISIModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.spatial_information is not None
    assert result.quality_metrics.spatial_information >= 0


def test_ti_si_video(video_sample):
    from ayase.modules.ti_si import TISIModule

    m = TISIModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.temporal_information is not None
    assert result.quality_metrics.spatial_information is not None


def test_flicker_detection_basics():
    from ayase.modules.flicker_detection import FlickerDetectionModule
    from .conftest import _test_module_basics

    _test_module_basics(FlickerDetectionModule, "flicker_detection")


def test_flicker_detection_video(video_sample):
    from ayase.modules.flicker_detection import FlickerDetectionModule

    m = FlickerDetectionModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.flicker_score is not None
    assert 0 <= result.quality_metrics.flicker_score <= 100


def test_flicker_detection_image(image_sample):
    from ayase.modules.flicker_detection import FlickerDetectionModule

    m = FlickerDetectionModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.flicker_score is None


def test_judder_stutter_basics():
    from ayase.modules.judder_stutter import JudderStutterModule
    from .conftest import _test_module_basics

    _test_module_basics(JudderStutterModule, "judder_stutter")


def test_judder_stutter_video(video_sample):
    from ayase.modules.judder_stutter import JudderStutterModule

    m = JudderStutterModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.judder_score is not None
    assert result.quality_metrics.stutter_score is not None
    assert 0 <= result.quality_metrics.judder_score <= 100
    assert 0 <= result.quality_metrics.stutter_score <= 100


def test_production_quality_basics():
    from ayase.modules.production_quality import ProductionQualityModule
    from .conftest import _test_module_basics

    _test_module_basics(ProductionQualityModule, "production_quality")


def test_production_quality_image(image_sample):
    from ayase.modules.production_quality import ProductionQualityModule

    m = ProductionQualityModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.white_balance_score is not None
    assert result.quality_metrics.focus_quality is not None
    assert result.quality_metrics.banding_severity is not None


def test_production_quality_video(video_sample):
    from ayase.modules.production_quality import ProductionQualityModule

    m = ProductionQualityModule()
    result = m.process(video_sample)
    qm = result.quality_metrics
    assert qm is not None
    assert qm.color_grading_score is not None
    assert qm.exposure_consistency is not None


def test_scene_complexity_basics():
    from ayase.modules.scene_complexity import SceneComplexityModule
    from .conftest import _test_module_basics

    _test_module_basics(SceneComplexityModule, "scene_complexity")


def test_compression_artifacts_basics():
    from ayase.modules.compression_artifacts import CompressionArtifactsModule
    from .conftest import _test_module_basics

    _test_module_basics(CompressionArtifactsModule, "compression_artifacts")


def test_codec_specific_quality_basics():
    from ayase.modules.codec_specific_quality import CodecSpecificQualityModule
    from .conftest import _test_module_basics

    _test_module_basics(CodecSpecificQualityModule, "codec_specific_quality")


def test_codec_specific_quality_video(video_sample):
    from ayase.modules.codec_specific_quality import CodecSpecificQualityModule

    m = CodecSpecificQualityModule()
    m.setup()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.codec_artifacts is not None


def test_object_permanence_basics():
    from ayase.modules.object_permanence import ObjectPermanenceModule
    from .conftest import _test_module_basics

    _test_module_basics(ObjectPermanenceModule, "object_permanence")


def test_bias_detection_basics():
    from ayase.modules.bias_detection import BiasDetectionModule
    from .conftest import _test_module_basics

    _test_module_basics(BiasDetectionModule, "bias_detection")


def test_bias_detection_setup():
    from ayase.modules.bias_detection import BiasDetectionModule

    m = BiasDetectionModule()
    m.setup()
    assert m._ml_available


def test_dynamics_range_basics():
    from ayase.modules.dynamics_range import DynamicsRangeModule
    from .conftest import _test_module_basics

    _test_module_basics(DynamicsRangeModule, "dynamics_range")


def test_dynamics_range_video(video_sample):
    from ayase.modules.dynamics_range import DynamicsRangeModule

    m = DynamicsRangeModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.dynamics_range is not None
    assert result.quality_metrics.dynamics_range >= 0


def test_dynamics_range_image(image_sample):
    from ayase.modules.dynamics_range import DynamicsRangeModule

    m = DynamicsRangeModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.dynamics_range is None


def test_dynamics_controllability_basics():
    from ayase.modules.dynamics_controllability import DynamicsControllabilityModule
    from .conftest import _test_module_basics

    _test_module_basics(DynamicsControllabilityModule, "dynamics_controllability")


def test_dynamics_controllability_keywords():
    from ayase.modules.dynamics_controllability import DynamicsControllabilityModule

    m = DynamicsControllabilityModule()
    assert m._extract_expected_motion("a fast car racing") > 0.5
    assert m._extract_expected_motion("a still lake with calm water") < 0.3
    assert m._extract_expected_motion("a landscape") == 0.5


def test_dynamics_controllability_no_caption(video_sample):
    from ayase.modules.dynamics_controllability import DynamicsControllabilityModule

    m = DynamicsControllabilityModule()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.dynamics_controllability is None


def test_dynamics_controllability_image(image_sample):
    from ayase.modules.dynamics_controllability import DynamicsControllabilityModule

    m = DynamicsControllabilityModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.dynamics_controllability is None

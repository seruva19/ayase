def test_deepfake_detection_basics():
    from ayase.modules.deepfake_detection import DeepfakeDetectionModule
    from .conftest import _test_module_basics

    _test_module_basics(DeepfakeDetectionModule, "deepfake_detection")


def test_deepfake_detection_image(image_sample):
    from ayase.modules.deepfake_detection import DeepfakeDetectionModule

    m = DeepfakeDetectionModule({})
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.deepfake_probability is not None


def test_watermark_robustness_basics():
    from ayase.modules.watermark_robustness import WatermarkRobustnessModule
    from .conftest import _test_module_basics

    _test_module_basics(WatermarkRobustnessModule, "watermark_robustness")


def test_watermark_robustness_image(image_sample):
    from ayase.modules.watermark_robustness import WatermarkRobustnessModule

    m = WatermarkRobustnessModule()
    m.setup()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.watermark_strength is not None
    assert 0 <= result.quality_metrics.watermark_strength <= 1


def test_harmful_content_basics():
    from ayase.modules.harmful_content import HarmfulContentModule
    from .conftest import _test_module_basics

    _test_module_basics(HarmfulContentModule, "harmful_content")

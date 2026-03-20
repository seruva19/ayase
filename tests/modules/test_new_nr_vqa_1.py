from ayase.models import QualityMetrics

from .conftest import _test_module_basics


# ---------------------------------------------------------------------------
# RAPIQUE
# ---------------------------------------------------------------------------


def test_rapique_basics():
    from ayase.modules.rapique import RAPIQUEModule

    _test_module_basics(RAPIQUEModule, "rapique")


def test_rapique_video(video_sample):
    from ayase.modules.rapique import RAPIQUEModule

    video_sample.quality_metrics = QualityMetrics()
    m = RAPIQUEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.rapique_score is not None
    assert 0 <= result.quality_metrics.rapique_score <= 1


def test_rapique_image(image_sample):
    from ayase.modules.rapique import RAPIQUEModule

    image_sample.quality_metrics = QualityMetrics()
    m = RAPIQUEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.rapique_score is not None
    assert 0 <= result.quality_metrics.rapique_score <= 1


# ---------------------------------------------------------------------------
# CONVIQT
# ---------------------------------------------------------------------------


def test_conviqt_basics():
    from ayase.modules.conviqt import CONVIQTModule

    _test_module_basics(CONVIQTModule, "conviqt")


def test_conviqt_video(video_sample):
    from ayase.modules.conviqt import CONVIQTModule

    video_sample.quality_metrics = QualityMetrics()
    m = CONVIQTModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.conviqt_score is not None
    assert 0 <= result.quality_metrics.conviqt_score <= 1


def test_conviqt_image(image_sample):
    from ayase.modules.conviqt import CONVIQTModule

    image_sample.quality_metrics = QualityMetrics()
    m = CONVIQTModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.conviqt_score is not None
    assert 0 <= result.quality_metrics.conviqt_score <= 1


# ---------------------------------------------------------------------------
# StableVQA (VIDEO ONLY)
# ---------------------------------------------------------------------------


def test_stablevqa_basics():
    from ayase.modules.stablevqa import StableVQAModule

    _test_module_basics(StableVQAModule, "stablevqa")


def test_stablevqa_video(video_sample):
    from ayase.modules.stablevqa import StableVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = StableVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.stablevqa_score is not None
    assert 0 <= result.quality_metrics.stablevqa_score <= 1


def test_stablevqa_image(image_sample):
    from ayase.modules.stablevqa import StableVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = StableVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.stablevqa_score is None


# ---------------------------------------------------------------------------
# MaxVQA
# ---------------------------------------------------------------------------


def test_maxvqa_basics():
    from ayase.modules.maxvqa import MaxVQAModule

    _test_module_basics(MaxVQAModule, "maxvqa")


def test_maxvqa_video(video_sample):
    from ayase.modules.maxvqa import MaxVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = MaxVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.maxvqa_score is not None
    assert 0 <= result.quality_metrics.maxvqa_score <= 1


def test_maxvqa_image(image_sample):
    from ayase.modules.maxvqa import MaxVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = MaxVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.maxvqa_score is not None
    assert 0 <= result.quality_metrics.maxvqa_score <= 1


# ---------------------------------------------------------------------------
# BVQI
# ---------------------------------------------------------------------------


def test_bvqi_basics():
    from ayase.modules.bvqi import BVQIModule

    _test_module_basics(BVQIModule, "bvqi")


def test_bvqi_video(video_sample):
    from ayase.modules.bvqi import BVQIModule

    video_sample.quality_metrics = QualityMetrics()
    m = BVQIModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.bvqi_score is not None
    assert 0 <= result.quality_metrics.bvqi_score <= 1


def test_bvqi_image(image_sample):
    from ayase.modules.bvqi import BVQIModule

    image_sample.quality_metrics = QualityMetrics()
    m = BVQIModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.bvqi_score is not None
    assert 0 <= result.quality_metrics.bvqi_score <= 1


# ---------------------------------------------------------------------------
# ModularBVQA
# ---------------------------------------------------------------------------


def test_modularbvqa_basics():
    from ayase.modules.modularbvqa import ModularBVQAModule

    _test_module_basics(ModularBVQAModule, "modularbvqa")


def test_modularbvqa_video(video_sample):
    from ayase.modules.modularbvqa import ModularBVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = ModularBVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.modularbvqa_score is not None
    assert 0 <= result.quality_metrics.modularbvqa_score <= 1


def test_modularbvqa_image(image_sample):
    from ayase.modules.modularbvqa import ModularBVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = ModularBVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.modularbvqa_score is not None
    assert 0 <= result.quality_metrics.modularbvqa_score <= 1


# ---------------------------------------------------------------------------
# PTMVQA
# ---------------------------------------------------------------------------


def test_ptmvqa_basics():
    from ayase.modules.ptmvqa import PTMVQAModule

    _test_module_basics(PTMVQAModule, "ptmvqa")


def test_ptmvqa_video(video_sample):
    from ayase.modules.ptmvqa import PTMVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = PTMVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.ptmvqa_score is not None
    assert 0 <= result.quality_metrics.ptmvqa_score <= 1


def test_ptmvqa_image(image_sample):
    from ayase.modules.ptmvqa import PTMVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = PTMVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.ptmvqa_score is not None
    assert 0 <= result.quality_metrics.ptmvqa_score <= 1


# ---------------------------------------------------------------------------
# CLIPVQA
# ---------------------------------------------------------------------------


def test_clipvqa_basics():
    from ayase.modules.clipvqa import CLIPVQAModule

    _test_module_basics(CLIPVQAModule, "clipvqa")


def test_clipvqa_video(video_sample):
    from ayase.modules.clipvqa import CLIPVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = CLIPVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.clipvqa_score is not None
    assert 0 <= result.quality_metrics.clipvqa_score <= 1


def test_clipvqa_image(image_sample):
    from ayase.modules.clipvqa import CLIPVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = CLIPVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.clipvqa_score is not None
    assert 0 <= result.quality_metrics.clipvqa_score <= 1


# ---------------------------------------------------------------------------
# DisCoVQA
# ---------------------------------------------------------------------------


def test_discovqa_basics():
    from ayase.modules.discovqa import DisCoVQAModule

    _test_module_basics(DisCoVQAModule, "discovqa")


def test_discovqa_video(video_sample):
    from ayase.modules.discovqa import DisCoVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = DisCoVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.discovqa_score is not None
    assert 0 <= result.quality_metrics.discovqa_score <= 1


def test_discovqa_image(image_sample):
    from ayase.modules.discovqa import DisCoVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = DisCoVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.discovqa_score is not None
    assert 0 <= result.quality_metrics.discovqa_score <= 1


# ---------------------------------------------------------------------------
# ZoomVQA
# ---------------------------------------------------------------------------


def test_zoomvqa_basics():
    from ayase.modules.zoomvqa import ZoomVQAModule

    _test_module_basics(ZoomVQAModule, "zoomvqa")


def test_zoomvqa_video(video_sample):
    from ayase.modules.zoomvqa import ZoomVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = ZoomVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result is video_sample
    assert result.quality_metrics.zoomvqa_score is not None
    assert 0 <= result.quality_metrics.zoomvqa_score <= 1


def test_zoomvqa_image(image_sample):
    from ayase.modules.zoomvqa import ZoomVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = ZoomVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result is image_sample
    assert result.quality_metrics.zoomvqa_score is not None
    assert 0 <= result.quality_metrics.zoomvqa_score <= 1

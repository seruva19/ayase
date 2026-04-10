import numpy as np

from ayase.models import QualityMetrics


def test_vmaf_basics():
    from ayase.modules.vmaf import VMAFModule
    from .conftest import _test_module_basics

    _test_module_basics(VMAFModule, "vmaf")


def test_vmaf_no_reference(video_sample):
    from ayase.modules.vmaf import VMAFModule

    m = VMAFModule()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.vmaf is None


def test_vmaf_image(image_sample):
    from ayase.modules.vmaf import VMAFModule

    m = VMAFModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.vmaf is None


def test_vmaf_config():
    from ayase.modules.vmaf import VMAFModule

    m = VMAFModule({"vmaf_model": "vmaf_4k_v0.6.1", "subsample": 2})
    assert m.vmaf_model == "vmaf_4k_v0.6.1"
    assert m.subsample == 2


def test_ms_ssim_basics():
    from ayase.modules.ms_ssim import MSSSIMModule
    from .conftest import _test_module_basics

    _test_module_basics(MSSSIMModule, "ms_ssim")


def test_ms_ssim_is_reference_based():
    from ayase.modules.ms_ssim import MSSSIMModule
    from ayase.base_modules import ReferenceBasedModule

    assert issubclass(MSSSIMModule, ReferenceBasedModule)


def test_ms_ssim_no_reference(image_sample):
    from ayase.modules.ms_ssim import MSSSIMModule

    m = MSSSIMModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.ms_ssim is None


def test_ms_ssim_config():
    from ayase.modules.ms_ssim import MSSSIMModule

    m = MSSSIMModule({"scales": 3, "subsample": 4})
    assert m.scales == 3
    assert m.subsample == 4


def test_t2v_score_basics():
    from ayase.modules.t2v_score import T2VScoreModule
    from .conftest import _test_module_basics

    _test_module_basics(T2VScoreModule, "t2v_score")


def test_t2v_score_no_caption(video_sample):
    from ayase.modules.t2v_score import T2VScoreModule

    m = T2VScoreModule()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.t2v_score is None


def test_t2v_score_image(image_sample):
    from ayase.modules.t2v_score import T2VScoreModule

    m = T2VScoreModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.t2v_score is None


def test_t2v_score_quality_simple():
    from ayase.modules.t2v_score import T2VScoreModule

    m = T2VScoreModule()
    frames = [np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(4)]
    score = m._compute_video_quality_simple(frames)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_vif_basics():
    from ayase.modules.vif import VIFModule
    from .conftest import _test_module_basics

    _test_module_basics(VIFModule, "vif")


def test_vif_is_reference_based():
    from ayase.modules.vif import VIFModule
    from ayase.base_modules import ReferenceBasedModule

    assert issubclass(VIFModule, ReferenceBasedModule)


def test_vif_no_reference(image_sample):
    from ayase.modules.vif import VIFModule

    m = VIFModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.vif is None


def test_niqe_basics():
    from ayase.modules.niqe import NIQEModule
    from .conftest import _test_module_basics

    _test_module_basics(NIQEModule, "niqe")


def test_niqe_is_no_reference():
    from ayase.modules.niqe import NIQEModule
    from ayase.base_modules import NoReferenceModule

    assert issubclass(NIQEModule, NoReferenceModule)


def test_niqe_no_setup(image_sample):
    from ayase.modules.niqe import NIQEModule

    m = NIQEModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.niqe is None


def test_naturalness_basics():
    from ayase.modules.naturalness import NaturalnessModule
    from .conftest import _test_module_basics

    _test_module_basics(NaturalnessModule, "naturalness")


def test_naturalness_is_no_reference():
    from ayase.modules.naturalness import NaturalnessModule
    from ayase.base_modules import NoReferenceModule

    assert issubclass(NaturalnessModule, NoReferenceModule)


def test_naturalness_nss():
    from ayase.modules.naturalness import NaturalnessModule

    m = NaturalnessModule()
    m._ml_available = True
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    score = m._compute_nss_naturalness(frame)
    assert isinstance(score, float)
    assert 0 <= score <= 1


def test_naturalness_mscn():
    from ayase.modules.naturalness import NaturalnessModule

    m = NaturalnessModule()
    gray = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
    mscn = m._compute_mscn_features(gray)
    assert mscn.shape == gray.shape


def test_hdr_sdr_vqa_basics():
    from ayase.modules.hdr_sdr_vqa import HDRSDRVQAModule
    from .conftest import _test_module_basics

    _test_module_basics(HDRSDRVQAModule, "hdr_sdr_vqa")


def test_hdr_sdr_vqa_image(image_sample):
    from ayase.modules.hdr_sdr_vqa import HDRSDRVQAModule

    m = HDRSDRVQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert (
        result.quality_metrics.sdr_quality is not None
        or result.quality_metrics.hdr_quality is not None
    )


def test_hdr_sdr_vqa_video(video_sample):
    from ayase.modules.hdr_sdr_vqa import HDRSDRVQAModule

    m = HDRSDRVQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert (
        result.quality_metrics.hdr_quality is not None
        or result.quality_metrics.sdr_quality is not None
    )


def test_hdr_sdr_detect_hdr():
    from ayase.modules.hdr_sdr_vqa import HDRSDRVQAModule

    m = HDRSDRVQAModule()
    bright = np.full((64, 64, 3), 240, dtype=np.uint8)
    assert m._detect_hdr(bright) is False
    normal = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert m._detect_hdr(normal) is False
    hdr16 = np.full((64, 64, 3), 1024, dtype=np.uint16)
    assert m._detect_hdr(hdr16) is True


def test_video_memorability_basics():
    from ayase.modules.video_memorability import VideoMemorabilityModule
    from .conftest import _test_module_basics

    _test_module_basics(VideoMemorabilityModule, "video_memorability")


def test_video_memorability_image(image_sample):
    from ayase.modules.video_memorability import VideoMemorabilityModule

    m = VideoMemorabilityModule()
    result = m.process(image_sample)
    # Without ML backend, module skips and returns sample unchanged
    if not m._ml_available:
        assert result.quality_metrics is None or result.quality_metrics.video_memorability is None
    else:
        assert result.quality_metrics is not None
        assert result.quality_metrics.video_memorability is not None
        assert 0 <= result.quality_metrics.video_memorability <= 1


def test_video_memorability_video(video_sample):
    from ayase.modules.video_memorability import VideoMemorabilityModule

    m = VideoMemorabilityModule()
    result = m.process(video_sample)
    # Without ML backend, module skips and returns sample unchanged
    if not m._ml_available:
        assert result.quality_metrics is None or result.quality_metrics.video_memorability is None
    else:
        assert result.quality_metrics is not None
        assert result.quality_metrics.video_memorability is not None
        assert 0 <= result.quality_metrics.video_memorability <= 1


def test_usability_rate_basics():
    from ayase.modules.usability_rate import UsabilityRateModule
    from .conftest import _test_module_basics

    _test_module_basics(UsabilityRateModule, "usability_rate")


def test_usability_rate_no_metrics(image_sample):
    from ayase.modules.usability_rate import UsabilityRateModule

    m = UsabilityRateModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.usability_rate is not None
    assert result.quality_metrics.usability_rate == 50.0


def test_usability_rate_with_metrics(image_sample):
    from ayase.modules.usability_rate import UsabilityRateModule

    m = UsabilityRateModule()
    image_sample.quality_metrics = QualityMetrics(technical_score=80.0)
    result = m.process(image_sample)
    assert result.quality_metrics.usability_rate == 100.0


def test_usability_rate_below_threshold(image_sample):
    from ayase.modules.usability_rate import UsabilityRateModule

    m = UsabilityRateModule()
    image_sample.quality_metrics = QualityMetrics(technical_score=30.0)
    result = m.process(image_sample)
    assert result.quality_metrics.usability_rate == 0.0


def test_llm_descriptive_qa_basics():
    from ayase.modules.llm_descriptive_qa import LLMDescriptiveQAModule
    from .conftest import _test_module_basics

    _test_module_basics(LLMDescriptiveQAModule, "llm_descriptive_qa")


def test_llm_descriptive_qa_no_setup(image_sample):
    from ayase.modules.llm_descriptive_qa import LLMDescriptiveQAModule

    m = LLMDescriptiveQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.confidence_score is None


def test_llm_descriptive_qa_extract_issues():
    from ayase.modules.llm_descriptive_qa import LLMDescriptiveQAModule

    m = LLMDescriptiveQAModule()
    issues = m._extract_issues("The image has slight blur and some noise artifacts")
    assert "blur" in issues
    assert "noise" in issues
    assert "artifact" in issues


def test_llm_descriptive_qa_config():
    from ayase.modules.llm_descriptive_qa import LLMDescriptiveQAModule

    m = LLMDescriptiveQAModule({"num_frames": 8, "use_openai": True})
    assert m.num_frames == 8
    assert m.use_openai is True


def test_reference_and_meta_fields():
    qm = QualityMetrics()
    for field in ["vmaf", "ms_ssim", "vif"]:
        assert hasattr(qm, field)
    for field in [
        "niqe",
        "scene_complexity",
        "compression_artifacts",
        "naturalness_score",
        "video_memorability",
    ]:
        assert hasattr(qm, field)
    for field in ["t2v_score", "t2v_alignment", "t2v_quality"]:
        assert hasattr(qm, field)
    for field in ["dynamics_range", "dynamics_controllability"]:
        assert hasattr(qm, field)
    for field in ["hdr_quality", "sdr_quality"]:
        assert hasattr(qm, field)
    for field in ["usability_rate", "confidence_score"]:
        assert hasattr(qm, field)


def test_reference_and_meta_dataset_stats_fields():
    from ayase.models import DatasetStats

    ds = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
    for field in ["fvd", "kvd", "fvmd"]:
        assert hasattr(ds, field)

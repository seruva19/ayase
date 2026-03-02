import numpy as np

from ayase.models import QualityMetrics, Sample


def test_cambi_basics():
    from ayase.modules.cambi import CAMBIModule
    from .conftest import _test_module_basics

    _test_module_basics(CAMBIModule, "cambi")


def test_cambi_image(image_sample):
    from ayase.modules.cambi import CAMBIModule

    m = CAMBIModule()
    m._ml_available = True
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.cambi is None


def test_cambi_no_ffmpeg(video_sample):
    from ayase.modules.cambi import CAMBIModule

    m = CAMBIModule()
    assert m._ml_available is False
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.cambi is None


def test_xpsnr_basics():
    from ayase.modules.xpsnr import XPSNRModule
    from .conftest import _test_module_basics

    _test_module_basics(XPSNRModule, "xpsnr")


def test_xpsnr_no_reference(video_sample):
    from ayase.modules.xpsnr import XPSNRModule

    m = XPSNRModule()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.xpsnr is None


def test_xpsnr_image(image_sample):
    from ayase.modules.xpsnr import XPSNRModule

    m = XPSNRModule()
    m._ml_available = True
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.xpsnr is None


def test_vmaf_phone_basics():
    from ayase.modules.vmaf_phone import VMAFPhoneModule
    from .conftest import _test_module_basics

    _test_module_basics(VMAFPhoneModule, "vmaf_phone")


def test_vmaf_phone_no_reference(video_sample):
    from ayase.modules.vmaf_phone import VMAFPhoneModule

    m = VMAFPhoneModule()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.vmaf_phone is None


def test_vmaf_4k_basics():
    from ayase.modules.vmaf_4k import VMAF4KModule
    from .conftest import _test_module_basics

    _test_module_basics(VMAF4KModule, "vmaf_4k")


def test_vmaf_4k_no_reference(video_sample):
    from ayase.modules.vmaf_4k import VMAF4KModule

    m = VMAF4KModule()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.vmaf_4k is None


def test_visqol_basics():
    from ayase.modules.visqol import ViSQOLModule
    from .conftest import _test_module_basics

    _test_module_basics(ViSQOLModule, "visqol")


def test_visqol_image(image_sample):
    from ayase.modules.visqol import ViSQOLModule

    m = ViSQOLModule()
    m._ml_available = True
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.visqol is None


def test_visqol_no_backend(video_sample):
    from ayase.modules.visqol import ViSQOLModule

    m = ViSQOLModule()
    assert m._ml_available is False
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.visqol is None


def test_dnsmos_basics():
    from ayase.modules.dnsmos import DNSMOSModule
    from .conftest import _test_module_basics

    _test_module_basics(DNSMOSModule, "dnsmos")


def test_dnsmos_image(image_sample):
    from ayase.modules.dnsmos import DNSMOSModule

    m = DNSMOSModule()
    m._ml_available = True
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.dnsmos_overall is None


def test_dnsmos_no_backend(video_sample):
    from ayase.modules.dnsmos import DNSMOSModule

    m = DNSMOSModule()
    assert m._ml_available is False
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.dnsmos_overall is None


def test_pu_metrics_basics():
    from ayase.modules.pu_metrics import PUMetricsModule
    from .conftest import _test_module_basics

    _test_module_basics(PUMetricsModule, "pu_metrics")


def test_pu_metrics_no_reference(image_sample):
    from ayase.modules.pu_metrics import PUMetricsModule

    m = PUMetricsModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.pu_psnr is None


def test_pu_metrics_setup():
    from ayase.modules.pu_metrics import PUMetricsModule

    m = PUMetricsModule()
    m.setup()
    assert m._ml_available is True


def test_pu21_encode():
    from ayase.modules.pu_metrics import _pu21_encode

    linear = np.array([0.0, 0.5, 1.0])
    encoded = _pu21_encode(linear)
    assert len(encoded) == 3
    assert encoded[0] < encoded[1] < encoded[2]


def test_pu_metrics_compute():
    from ayase.modules.pu_metrics import PUMetricsModule

    m = PUMetricsModule()
    m.setup()
    ref = np.full((64, 64, 3), 5000.0, dtype=np.float32)
    dist = ref.copy()
    dist[:32, :32] = 7000.0
    psnr = m._compute_pu_psnr(ref, dist)
    assert psnr > 0
    ssim = m._compute_pu_ssim(ref, dist)
    assert 0.0 <= ssim <= 1.0


def test_pu_metrics_identical():
    from ayase.modules.pu_metrics import PUMetricsModule

    m = PUMetricsModule()
    ref = np.full((64, 64, 3), 5000.0, dtype=np.float32)
    psnr = m._compute_pu_psnr(ref, ref)
    assert psnr >= 90.0


def test_hdr_metadata_basics():
    from ayase.modules.hdr_metadata import HDRMetadataModule
    from .conftest import _test_module_basics

    _test_module_basics(HDRMetadataModule, "hdr_metadata")


def test_hdr_metadata_image(image_sample):
    from ayase.modules.hdr_metadata import HDRMetadataModule

    m = HDRMetadataModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.max_fall is None


def test_hdr_metadata_video(video_sample):
    from ayase.modules.hdr_metadata import HDRMetadataModule

    m = HDRMetadataModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.max_fall is not None
    assert result.quality_metrics.max_fall >= 0
    assert result.quality_metrics.max_cll is not None
    assert result.quality_metrics.max_cll >= 0


def test_bt709_luminance():
    from ayase.modules.hdr_metadata import _bt709_luminance

    white = np.array([[[255, 255, 255]]], dtype=np.float32)
    lum = _bt709_luminance(white)
    assert abs(lum[0, 0] - 255.0) < 1.0


def test_hdr_vdp_basics():
    from ayase.modules.hdr_vdp import HDRVDPModule
    from .conftest import _test_module_basics

    _test_module_basics(HDRVDPModule, "hdr_vdp")


def test_hdr_vdp_no_reference(image_sample):
    from ayase.modules.hdr_vdp import HDRVDPModule

    m = HDRVDPModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.hdr_vdp is None


def test_hdr_vdp_setup():
    from ayase.modules.hdr_vdp import HDRVDPModule

    m = HDRVDPModule()
    m.setup()
    assert m._ml_available is True
    assert m._backend == "approx"


def test_hdr_vdp_reference(tmp_dir):
    from ayase.modules.hdr_vdp import HDRVDPModule

    m = HDRVDPModule()
    m.setup()
    ref_img = np.full((64, 64, 3), 128, dtype=np.uint8)
    dist_img = ref_img.copy()
    dist_img[:32, :32] = 200
    ref_path = tmp_dir / "ref.png"
    dist_path = tmp_dir / "dist.png"
    import cv2

    cv2.imwrite(str(ref_path), ref_img)
    cv2.imwrite(str(dist_path), dist_img)
    score = m.compute_reference_score(dist_path, ref_path)
    assert score is not None
    assert score > 0


def test_delta_ictcp_basics():
    from ayase.modules.delta_ictcp import DeltaICtCpModule
    from .conftest import _test_module_basics

    _test_module_basics(DeltaICtCpModule, "delta_ictcp")


def test_delta_ictcp_no_reference(image_sample):
    from ayase.modules.delta_ictcp import DeltaICtCpModule

    m = DeltaICtCpModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.delta_ictcp is None


def test_delta_ictcp_pq():
    from ayase.modules.delta_ictcp import _linear_to_pq

    linear = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    pq = _linear_to_pq(linear)
    for i in range(len(pq) - 1):
        assert pq[i] <= pq[i + 1]


def test_delta_ictcp_identical(tmp_dir):
    from ayase.modules.delta_ictcp import DeltaICtCpModule

    m = DeltaICtCpModule()
    m._ml_available = True
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    p = tmp_dir / "img.png"
    import cv2

    cv2.imwrite(str(p), img)
    score = m.compute_reference_score(p, p)
    assert score is not None
    assert score < 0.01


def test_delta_ictcp_different(tmp_dir):
    from ayase.modules.delta_ictcp import DeltaICtCpModule

    m = DeltaICtCpModule()
    m._ml_available = True
    ref = np.full((64, 64, 3), 128, dtype=np.uint8)
    dist = np.full((64, 64, 3), 50, dtype=np.uint8)
    ref_path = tmp_dir / "ref.png"
    dist_path = tmp_dir / "dist.png"
    import cv2

    cv2.imwrite(str(ref_path), ref)
    cv2.imwrite(str(dist_path), dist)
    score = m.compute_reference_score(dist_path, ref_path)
    assert score is not None
    assert score > 0


def test_strred_basics():
    from ayase.modules.strred import STRREDModule
    from .conftest import _test_module_basics

    _test_module_basics(STRREDModule, "strred")


def test_strred_no_reference(video_sample):
    from ayase.modules.strred import STRREDModule

    m = STRREDModule()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.strred is None


def test_strred_setup():
    from ayase.modules.strred import STRREDModule

    m = STRREDModule()
    m.setup()
    assert m._ml_available is True
    assert m._backend in ("skvideo", "approx")


def test_cgvqm_basics():
    from ayase.modules.cgvqm import CGVQMModule
    from .conftest import _test_module_basics

    _test_module_basics(CGVQMModule, "cgvqm")


def test_cgvqm_no_reference(image_sample):
    from ayase.modules.cgvqm import CGVQMModule

    m = CGVQMModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.cgvqm is None


def test_cgvqm_setup():
    from ayase.modules.cgvqm import CGVQMModule

    m = CGVQMModule()
    m.setup()
    assert m._ml_available is True
    assert m._backend == "approx"


def test_cgvqm_reference(tmp_dir):
    from ayase.modules.cgvqm import CGVQMModule

    m = CGVQMModule()
    m.setup()
    ref_img = np.full((64, 64, 3), 128, dtype=np.uint8)
    dist_img = ref_img.copy()
    dist_img[:32, :32] = 200
    ref_path = tmp_dir / "ref.png"
    dist_path = tmp_dir / "dist.png"
    import cv2

    cv2.imwrite(str(ref_path), ref_img)
    cv2.imwrite(str(dist_path), dist_img)
    score = m.compute_reference_score(dist_path, ref_path)
    assert score is not None
    assert 0.0 <= score <= 100.0


def test_ciede2000_basics():
    from ayase.modules.ciede2000 import CIEDE2000Module
    from .conftest import _test_module_basics

    _test_module_basics(CIEDE2000Module, "ciede2000")


def test_ciede2000_no_reference(image_sample):
    from ayase.modules.ciede2000 import CIEDE2000Module

    m = CIEDE2000Module()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.ciede2000 is None


def test_ciede2000_identical(tmp_dir):
    from ayase.modules.ciede2000 import CIEDE2000Module

    m = CIEDE2000Module()
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    p = tmp_dir / "img.png"
    import cv2

    cv2.imwrite(str(p), img)
    score = m.compute_reference_score(p, p)
    assert score is not None
    assert score < 0.1


def test_ciede2000_different(tmp_dir):
    from ayase.modules.ciede2000 import CIEDE2000Module

    m = CIEDE2000Module()
    ref = np.full((64, 64, 3), 128, dtype=np.uint8)
    dist = np.full((64, 64, 3), 50, dtype=np.uint8)
    ref_path = tmp_dir / "ref.png"
    dist_path = tmp_dir / "dist.png"
    import cv2

    cv2.imwrite(str(ref_path), ref)
    cv2.imwrite(str(dist_path), dist)
    score = m.compute_reference_score(dist_path, ref_path)
    assert score is not None
    assert score > 0


def test_ciede2000_vectorized():
    from ayase.modules.ciede2000 import _ciede2000_pixel

    lab1 = np.array([[[50.0, 0.0, 0.0]]])
    lab2 = np.array([[[50.0, 0.0, 0.0]]])
    de = _ciede2000_pixel(lab1, lab2)
    assert de[0, 0] < 0.01


def test_psnr_hvs_basics():
    from ayase.modules.psnr_hvs import PSNRHVSModule
    from .conftest import _test_module_basics

    _test_module_basics(PSNRHVSModule, "psnr_hvs")


def test_psnr_hvs_no_reference(image_sample):
    from ayase.modules.psnr_hvs import PSNRHVSModule

    m = PSNRHVSModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.psnr_hvs is None


def test_psnr_hvs_setup():
    from ayase.modules.psnr_hvs import PSNRHVSModule

    m = PSNRHVSModule()
    m.setup()
    assert m._ml_available is True


def test_psnr_hvs_identical(tmp_dir):
    from ayase.modules.psnr_hvs import PSNRHVSModule

    m = PSNRHVSModule()
    m.setup()
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    p = tmp_dir / "img.png"
    import cv2

    cv2.imwrite(str(p), img)
    score = m.compute_reference_score(p, p)
    assert score is not None
    assert score >= 90.0


def test_psnr_hvs_different(tmp_dir):
    from ayase.modules.psnr_hvs import PSNRHVSModule

    m = PSNRHVSModule()
    m.setup()
    ref = np.full((64, 64, 3), 128, dtype=np.uint8)
    dist = ref.copy()
    dist[:32, :32] = 200
    ref_path = tmp_dir / "ref.png"
    dist_path = tmp_dir / "dist.png"
    import cv2

    cv2.imwrite(str(ref_path), ref)
    cv2.imwrite(str(dist_path), dist)
    score = m.compute_reference_score(dist_path, ref_path)
    assert score is not None
    assert score > 0
    assert score < 100


def test_bd_rate_basics():
    from ayase.modules.bd_rate import BDRateModule
    from .conftest import _test_module_basics

    _test_module_basics(BDRateModule, "bd_rate")


def test_bd_rate_no_metadata(video_sample):
    from ayase.modules.bd_rate import BDRateModule

    m = BDRateModule()
    assert len(m._feature_cache) == 0


def test_bd_rate_function():
    from ayase.modules.bd_rate import _bd_rate

    rates = [500, 1000, 2000, 4000]
    quality = [30.0, 35.0, 40.0, 45.0]
    bd = _bd_rate(rates, quality, rates, quality)
    assert abs(bd) < 1.0


def test_bd_rate_better_codec():
    from ayase.modules.bd_rate import _bd_rate

    rates1 = [500, 1000, 2000, 4000]
    quality1 = [30.0, 35.0, 40.0, 45.0]
    rates2 = [350, 700, 1400, 2800]
    quality2 = [30.0, 35.0, 40.0, 45.0]
    bd = _bd_rate(rates1, quality1, rates2, quality2)
    assert bd < 0


def test_p1203_basics():
    from ayase.modules.p1203 import P1203Module
    from .conftest import _test_module_basics

    _test_module_basics(P1203Module, "p1203")


def test_p1203_image(image_sample):
    from ayase.modules.p1203 import P1203Module

    m = P1203Module()
    m._ml_available = True
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.p1203_mos is None


def test_p1203_no_metadata(video_sample):
    from ayase.modules.p1203 import P1203Module

    m = P1203Module()
    m.setup()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.p1203_mos is None


def test_p1203_with_metadata(synthetic_video):
    from ayase.models import VideoMetadata
    from ayase.modules.p1203 import P1203Module

    m = P1203Module()
    m.setup()
    sample = Sample(
        path=synthetic_video,
        is_video=True,
        video_metadata=VideoMetadata(
            width=1920,
            height=1080,
            frame_count=300,
            fps=30.0,
            duration=10.0,
            codec="h264",
            bitrate=5000000,
            file_size=6250000,
        ),
    )
    result = m.process(sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.p1203_mos is not None
    assert 1.0 <= result.quality_metrics.p1203_mos <= 5.0


def test_p1203_low_bitrate_lower_mos(synthetic_video):
    from ayase.models import VideoMetadata
    from ayase.modules.p1203 import P1203Module

    m = P1203Module()
    m.setup()
    high_br = Sample(
        path=synthetic_video,
        is_video=True,
        video_metadata=VideoMetadata(
            width=1920,
            height=1080,
            frame_count=300,
            fps=30.0,
            duration=10.0,
            codec="h264",
            bitrate=10000000,
            file_size=12500000,
        ),
    )
    low_br = Sample(
        path=synthetic_video,
        is_video=True,
        video_metadata=VideoMetadata(
            width=1920,
            height=1080,
            frame_count=300,
            fps=30.0,
            duration=10.0,
            codec="h264",
            bitrate=500000,
            file_size=625000,
        ),
    )
    r_high = m.process(high_br)
    r_low = m.process(low_br)
    assert r_high.quality_metrics.p1203_mos >= r_low.quality_metrics.p1203_mos


def test_industry_fields():
    qm = QualityMetrics()
    for field in ["cambi", "xpsnr", "vmaf_phone", "vmaf_4k"]:
        assert hasattr(qm, field)
    for field in ["visqol", "dnsmos_overall", "dnsmos_sig", "dnsmos_bak"]:
        assert hasattr(qm, field)
    for field in ["pu_psnr", "pu_ssim", "max_fall", "max_cll", "hdr_vdp", "delta_ictcp"]:
        assert hasattr(qm, field)
    for field in ["ciede2000", "psnr_hvs", "psnr_hvs_m", "cgvqm", "strred", "p1203_mos"]:
        assert hasattr(qm, field)


def test_industry_dataset_stats_fields():
    from ayase.models import DatasetStats

    ds = DatasetStats(total_samples=0, valid_samples=0, invalid_samples=0, total_size=0)
    for field in ["bd_rate", "bd_psnr"]:
        assert hasattr(ds, field)

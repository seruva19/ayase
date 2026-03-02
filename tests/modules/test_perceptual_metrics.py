import numpy as np


def test_ilniqe_basics():
    from ayase.modules.ilniqe import ILNIQEModule
    from .conftest import _test_module_basics

    _test_module_basics(ILNIQEModule, "ilniqe")


def test_ilniqe_no_ml(image_sample):
    from ayase.modules.ilniqe import ILNIQEModule

    m = ILNIQEModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.ilniqe is None


def test_ilniqe_config():
    from ayase.modules.ilniqe import ILNIQEModule

    m = ILNIQEModule({"subsample": 5, "warning_threshold": 40.0})
    assert m.subsample == 5
    assert m.warning_threshold == 40.0


def test_nrqm_basics():
    from ayase.modules.nrqm import NRQMModule
    from .conftest import _test_module_basics

    _test_module_basics(NRQMModule, "nrqm")


def test_nrqm_no_ml(image_sample):
    from ayase.modules.nrqm import NRQMModule

    m = NRQMModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.nrqm is None


def test_pi_basics():
    from ayase.modules.pi_metric import PIModule
    from .conftest import _test_module_basics

    _test_module_basics(PIModule, "pi")


def test_pi_no_ml(image_sample):
    from ayase.modules.pi_metric import PIModule

    m = PIModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.pi_score is None


def test_piqe_basics():
    from ayase.modules.piqe import PIQEModule
    from .conftest import _test_module_basics

    _test_module_basics(PIQEModule, "piqe")


def test_piqe_no_ml(image_sample):
    from ayase.modules.piqe import PIQEModule

    m = PIQEModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.piqe is None


def test_piqe_config():
    from ayase.modules.piqe import PIQEModule

    m = PIQEModule({"warning_threshold": 60.0})
    assert m.warning_threshold == 60.0


def test_maclip_basics():
    from ayase.modules.maclip import MACLIPModule
    from .conftest import _test_module_basics

    _test_module_basics(MACLIPModule, "maclip")


def test_maclip_no_ml(image_sample):
    from ayase.modules.maclip import MACLIPModule

    m = MACLIPModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.maclip_score is None


def test_dmm_basics():
    from ayase.modules.dmm import DMMModule
    from .conftest import _test_module_basics

    _test_module_basics(DMMModule, "dmm")


def test_dmm_no_reference(image_sample):
    from ayase.modules.dmm import DMMModule

    m = DMMModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.dmm is None


def test_wadiqam_fr_basics():
    from ayase.modules.wadiqam_fr import WaDIQaMFRModule
    from .conftest import _test_module_basics

    _test_module_basics(WaDIQaMFRModule, "wadiqam_fr")


def test_wadiqam_fr_no_reference(image_sample):
    from ayase.modules.wadiqam_fr import WaDIQaMFRModule

    m = WaDIQaMFRModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.wadiqam_fr is None


def test_ssimc_basics():
    from ayase.modules.ssimc import SSIMCModule
    from .conftest import _test_module_basics

    _test_module_basics(SSIMCModule, "ssimc")


def test_ssimc_no_reference(image_sample):
    from ayase.modules.ssimc import SSIMCModule

    m = SSIMCModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.ssimc is None


def test_ssimulacra2_basics():
    from ayase.modules.ssimulacra2 import SSIMULACRA2Module
    from .conftest import _test_module_basics

    _test_module_basics(SSIMULACRA2Module, "ssimulacra2")


def test_ssimulacra2_no_reference(image_sample):
    from ayase.modules.ssimulacra2 import SSIMULACRA2Module

    m = SSIMULACRA2Module()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.ssimulacra2 is None


def test_ssimulacra2_config():
    from ayase.modules.ssimulacra2 import SSIMULACRA2Module

    m = SSIMULACRA2Module({"subsample": 3, "warning_threshold": 30.0})
    assert m.subsample == 3
    assert m.warning_threshold == 30.0


def test_butteraugli_basics():
    from ayase.modules.butteraugli import ButteraugliModule
    from .conftest import _test_module_basics

    _test_module_basics(ButteraugliModule, "butteraugli")


def test_butteraugli_no_reference(image_sample):
    from ayase.modules.butteraugli import ButteraugliModule

    m = ButteraugliModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.butteraugli is None


def test_butteraugli_approx_setup():
    from ayase.modules.butteraugli import ButteraugliModule

    m = ButteraugliModule()
    m.setup()
    assert m._ml_available is True
    assert m._backend == "approx"


def test_butteraugli_approx_reference(tmp_dir):
    from ayase.modules.butteraugli import ButteraugliModule

    m = ButteraugliModule()
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
    assert score >= 0.0


def test_flip_basics():
    from ayase.modules.flip_metric import FLIPModule
    from .conftest import _test_module_basics

    _test_module_basics(FLIPModule, "flip")


def test_flip_no_reference(image_sample):
    from ayase.modules.flip_metric import FLIPModule

    m = FLIPModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.flip_score is None


def test_flip_approx_setup():
    from ayase.modules.flip_metric import FLIPModule

    m = FLIPModule()
    m.setup()
    assert m._ml_available is True
    assert m._backend == "approx"


def test_flip_approx_reference(tmp_dir):
    from ayase.modules.flip_metric import FLIPModule

    m = FLIPModule()
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
    assert 0.0 <= score <= 1.0


def test_flip_identical_images(tmp_dir):
    from ayase.modules.flip_metric import FLIPModule

    m = FLIPModule()
    m.setup()
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    p1 = tmp_dir / "img1.png"
    p2 = tmp_dir / "img2.png"
    import cv2

    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), img)
    score = m.compute_reference_score(p1, p2)
    assert score is not None
    assert score < 0.01


def test_vmaf_neg_basics():
    from ayase.modules.vmaf_neg import VMAFNEGModule
    from .conftest import _test_module_basics

    _test_module_basics(VMAFNEGModule, "vmaf_neg")


def test_vmaf_neg_no_reference(video_sample):
    from ayase.modules.vmaf_neg import VMAFNEGModule

    m = VMAFNEGModule()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.vmaf_neg is None


def test_vmaf_neg_image(image_sample):
    from ayase.modules.vmaf_neg import VMAFNEGModule

    m = VMAFNEGModule()
    m._ml_available = True
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.vmaf_neg is None


def test_perceptual_metric_fields():
    from ayase.models import QualityMetrics

    qm = QualityMetrics()
    for field in ["ssimulacra2", "butteraugli", "flip_score", "vmaf_neg"]:
        assert hasattr(qm, field)
    for field in ["ilniqe", "nrqm", "pi_score", "piqe", "maclip_score"]:
        assert hasattr(qm, field)
    for field in ["dmm", "wadiqam_fr", "ssimc"]:
        assert hasattr(qm, field)

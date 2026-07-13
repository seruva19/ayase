from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

from ayase.models import QualityMetrics, Sample


def test_finevq_basics():
    from ayase.modules.finevq import FineVQModule
    from .conftest import _test_module_basics

    _test_module_basics(FineVQModule, "finevq")


def test_finevq_video(video_sample):
    from ayase.modules.finevq import FineVQModule

    video_sample.quality_metrics = QualityMetrics()
    m = FineVQModule()
    result = m.process(video_sample)
    # Without ML backend, module skips gracefully
    assert result.quality_metrics.finevq_score is None


def test_finevq_image(image_sample):
    from ayase.modules.finevq import FineVQModule

    image_sample.quality_metrics = QualityMetrics()
    m = FineVQModule()
    result = m.process(image_sample)
    # Without ML backend, module skips gracefully
    assert result.quality_metrics.finevq_score is None


def test_kvq_basics():
    from ayase.modules.kvq import KVQModule
    from .conftest import _test_module_basics

    _test_module_basics(KVQModule, "kvq")


def test_kvq_video(video_sample):
    from ayase.modules.kvq import KVQModule

    video_sample.quality_metrics = QualityMetrics()
    m = KVQModule()
    result = m.process(video_sample)
    # Without ML backend, module skips gracefully
    assert result.quality_metrics.kvq_score is None


def test_kvq_image(image_sample):
    from ayase.modules.kvq import KVQModule

    image_sample.quality_metrics = QualityMetrics()
    m = KVQModule()
    result = m.process(image_sample)
    # Without ML backend, module skips gracefully
    assert result.quality_metrics.kvq_score is None


def test_rqvqa_basics():
    from ayase.modules.rqvqa import RQVQAModule
    from .conftest import _test_module_basics

    _test_module_basics(RQVQAModule, "rqvqa")


def test_rqvqa_video(video_sample):
    from ayase.modules.rqvqa import RQVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = RQVQAModule()
    result = m.process(video_sample)
    # Without ML backend, module skips gracefully
    assert result.quality_metrics.rqvqa_score is None


def test_videval_basics():
    from ayase.modules.videval import VIDEVALModule
    from .conftest import _test_module_basics

    _test_module_basics(VIDEVALModule, "videval")


def test_videval_video(video_sample):
    from ayase.modules.videval import VIDEVALModule

    video_sample.quality_metrics = QualityMetrics()
    m = VIDEVALModule()
    result = m.process(video_sample)
    # Without ML backend, module skips gracefully
    assert result.quality_metrics.videval_score is None


def test_videval_image(image_sample):
    from ayase.modules.videval import VIDEVALModule

    image_sample.quality_metrics = QualityMetrics()
    m = VIDEVALModule()
    result = m.process(image_sample)
    # Without ML backend, module skips gracefully
    assert result.quality_metrics.videval_score is None


def test_tlvqm_basics():
    from ayase.modules.tlvqm import TLVQMModule
    from .conftest import _test_module_basics

    _test_module_basics(TLVQMModule, "tlvqm")


def test_tlvqm_video(video_sample):
    """The handcrafted-feature tier was removed: without the CNN-TLVQM
    backend loaded (setup not run), the score must stay unset."""
    from ayase.modules.tlvqm import TLVQMModule

    video_sample.quality_metrics = QualityMetrics()
    m = TLVQMModule()
    result = m.process(video_sample)
    assert m._ml_available is False
    assert result.quality_metrics.tlvqm_score is None


def test_funque_basics():
    from ayase.modules.funque import FUNQUEModule
    from .conftest import _test_module_basics

    _test_module_basics(FUNQUEModule, "funque")


def test_funque_video(video_sample):
    """Real funque package or nothing (SSIM-proxy tier removed)."""
    from ayase.modules.funque import FUNQUEModule

    video_sample.quality_metrics = QualityMetrics()
    video_sample.reference_path = video_sample.path  # FR metric
    m = FUNQUEModule()
    m.setup()
    result = m.process(video_sample)
    if m._backend == "funque":
        assert result.quality_metrics.funque_score is not None
        assert 0.0 <= result.quality_metrics.funque_score <= 1.0
    else:
        assert m._backend == "unavailable"
        assert result.quality_metrics.funque_score is None


def test_movie_basics():
    from ayase.modules.movie import MOVIEModule
    from .conftest import _test_module_basics

    _test_module_basics(MOVIEModule, "movie")


def test_movie_video_no_reference_is_unset(video_sample):
    """MOVIE is full-reference: without a reference nothing is fabricated."""
    from ayase.modules.movie import MOVIEModule

    video_sample.quality_metrics = QualityMetrics()
    m = MOVIEModule()
    result = m.process(video_sample)
    assert m._backend == "unavailable"
    assert result.quality_metrics.movie_score is None


def test_movie_video_with_reference(video_sample):
    """With a reference the real Gabor FR computation produces a score."""
    from ayase.modules.movie import MOVIEModule

    video_sample.quality_metrics = QualityMetrics()
    video_sample.reference_path = video_sample.path
    m = MOVIEModule()
    result = m.process(video_sample)
    assert result.quality_metrics.movie_score is not None
    assert 0.0 <= result.quality_metrics.movie_score <= 1.0


def test_st_greed_basics():
    from ayase.modules.st_greed import STGREEDModule
    from .conftest import _test_module_basics

    _test_module_basics(STGREEDModule, "st_greed")


def test_st_greed_video_no_reference_is_unset(video_sample):
    """ST-GREED is full-reference: without a reference nothing is fabricated."""
    from ayase.modules.st_greed import STGREEDModule

    video_sample.quality_metrics = QualityMetrics()
    m = STGREEDModule()
    result = m.process(video_sample)
    assert result.quality_metrics.st_greed_score is None


def test_st_greed_video_with_reference(video_sample):
    """With a reference the real entropic-difference FR computation runs."""
    from ayase.modules.st_greed import STGREEDModule

    video_sample.quality_metrics = QualityMetrics()
    video_sample.reference_path = video_sample.path
    m = STGREEDModule()
    result = m.process(video_sample)
    assert result.quality_metrics.st_greed_score is not None
    assert 0.0 <= result.quality_metrics.st_greed_score <= 1.0


def test_st_greed_image(image_sample):
    from ayase.modules.st_greed import STGREEDModule

    image_sample.quality_metrics = QualityMetrics()
    m = STGREEDModule()
    result = m.process(image_sample)
    assert result.quality_metrics.st_greed_score is None


def test_c3dvqa_basics():
    from ayase.modules.c3dvqa import C3DVQAModule
    from .conftest import _test_module_basics

    _test_module_basics(C3DVQAModule, "c3dvqa")


def test_c3dvqa_video(video_sample):
    """Real trained C3DVQA backend or nothing (3D-gradient proxy removed)."""
    from ayase.modules.c3dvqa import C3DVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = C3DVQAModule()
    m.setup()
    result = m.process(video_sample)
    if m._backend == "c3dvqa":
        assert result.quality_metrics.c3dvqa_score is not None
    else:
        assert m._backend == "unavailable"
        assert result.quality_metrics.c3dvqa_score is None


def test_flolpips_basics():
    from ayase.modules.flolpips import FloLPIPSModule
    from .conftest import _test_module_basics

    _test_module_basics(FloLPIPSModule, "flolpips")


def test_flolpips_video(video_sample):
    """FloLPIPS is full-reference and needs the RAFT+LPIPS backend; without
    setup the backend stays unavailable and no score is fabricated."""
    from ayase.modules.flolpips import FloLPIPSModule

    video_sample.quality_metrics = QualityMetrics()
    m = FloLPIPSModule({"subsample": 2, "size": 64})
    # setup() not called → backend must remain honest-unavailable
    result = m.process(video_sample)
    assert m._backend == "unavailable"
    assert result.quality_metrics.flolpips is None


def test_flolpips_image(image_sample):
    from ayase.modules.flolpips import FloLPIPSModule

    image_sample.quality_metrics = QualityMetrics()
    m = FloLPIPSModule()
    result = m.process(image_sample)
    assert result.quality_metrics.flolpips is None


def test_hdr_vqm_basics():
    from ayase.modules.hdr_vqm import HDRVQMModule
    from .conftest import _test_module_basics

    _test_module_basics(HDRVQMModule, "hdr_vqm")


def test_hdr_vqm_video(video_sample):
    """HDR-VQM is full-reference (PU21 + wavelets): scores only with a
    reference and the pywt backend, no NR proxy."""
    from ayase.modules.hdr_vqm import HDRVQMModule

    video_sample.quality_metrics = QualityMetrics()
    m = HDRVQMModule()
    m.setup()

    # No reference → no score, ever
    result = m.process(video_sample)
    assert result.quality_metrics.hdr_vqm is None

    video_sample.reference_path = video_sample.path
    result = m.process(video_sample)
    if m._backend == "pu21_wavelet":
        assert result.quality_metrics.hdr_vqm is not None
        assert 0.0 <= result.quality_metrics.hdr_vqm <= 1.0
    else:
        assert m._backend == "unavailable"
        assert result.quality_metrics.hdr_vqm is None


def test_hdr_vqm_image(image_sample):
    from ayase.modules.hdr_vqm import HDRVQMModule

    image_sample.quality_metrics = QualityMetrics()
    image_sample.reference_path = image_sample.path
    m = HDRVQMModule()
    m.setup()
    result = m.process(image_sample)
    if m._backend == "pu21_wavelet":
        assert result.quality_metrics.hdr_vqm is not None
    else:
        assert m._backend == "unavailable"
        assert result.quality_metrics.hdr_vqm is None


def test_st_lpips_basics():
    from ayase.modules.st_lpips import STLPIPSModule
    from .conftest import _test_module_basics

    _test_module_basics(STLPIPSModule, "st_lpips")


def test_st_lpips_video(video_sample):
    from ayase.modules.st_lpips import STLPIPSModule

    video_sample.quality_metrics = QualityMetrics()
    m = STLPIPSModule()
    result = m.process(video_sample)
    # Without ML backend, module skips gracefully
    assert result.quality_metrics.st_lpips is None


def test_st_lpips_image(image_sample):
    from ayase.modules.st_lpips import STLPIPSModule

    image_sample.quality_metrics = QualityMetrics()
    m = STLPIPSModule()
    result = m.process(image_sample)
    assert result.quality_metrics.st_lpips is None


def test_kvq_dispatches_to_real_model_when_loaded():
    """KVQ module dispatches to real model when backend=='kvq'."""
    from ayase.modules.kvq import KVQModule

    module = KVQModule()
    module._backend = "kvq"
    module._ml_available = True
    module._device = "cpu"

    # Mock the model to return a known score
    mock_model = MagicMock()
    mock_model.return_value = MagicMock(item=MagicMock(return_value=0.75))
    module._model = mock_model

    # Create a small test image
    frame = np.full((64, 64, 3), 128, dtype=np.uint8)

    score = module._process_kvq_model(
        Sample(path=Path("test.png"), is_video=False), [frame]
    )
    assert score is not None
    mock_model.assert_called_once()


def test_rqvqa_dispatches_to_real_ensemble_when_loaded():
    """RQ-VQA stores the published unbounded raw score without clipping."""
    from ayase.modules.rqvqa import RQVQAModule

    module = RQVQAModule({"test_mode": True})
    module._backend = "rqvqa"
    module._ml_available = True
    module._score_video = MagicMock(return_value=8.8)
    sample = Sample(path=Path("test.mp4"), is_video=True)

    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics.rqvqa_score == 8.8
    module._score_video.assert_called_once_with(sample)

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
    assert result.quality_metrics.finevq_score is not None
    assert 0.0 <= result.quality_metrics.finevq_score <= 1.0


def test_finevq_image(image_sample):
    from ayase.modules.finevq import FineVQModule

    image_sample.quality_metrics = QualityMetrics()
    m = FineVQModule()
    result = m.process(image_sample)
    assert result.quality_metrics.finevq_score is not None


def test_kvq_basics():
    from ayase.modules.kvq import KVQModule
    from .conftest import _test_module_basics

    _test_module_basics(KVQModule, "kvq")


def test_kvq_video(video_sample):
    from ayase.modules.kvq import KVQModule

    video_sample.quality_metrics = QualityMetrics()
    m = KVQModule()
    result = m.process(video_sample)
    assert result.quality_metrics.kvq_score is not None
    assert 0.0 <= result.quality_metrics.kvq_score <= 1.0


def test_kvq_image(image_sample):
    from ayase.modules.kvq import KVQModule

    image_sample.quality_metrics = QualityMetrics()
    m = KVQModule()
    result = m.process(image_sample)
    assert result.quality_metrics.kvq_score is not None


def test_rqvqa_basics():
    from ayase.modules.rqvqa import RQVQAModule
    from .conftest import _test_module_basics

    _test_module_basics(RQVQAModule, "rqvqa")


def test_rqvqa_video(video_sample):
    from ayase.modules.rqvqa import RQVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = RQVQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics.rqvqa_score is not None
    assert 0.0 <= result.quality_metrics.rqvqa_score <= 1.0


def test_videval_basics():
    from ayase.modules.videval import VIDEVALModule
    from .conftest import _test_module_basics

    _test_module_basics(VIDEVALModule, "videval")


def test_videval_video(video_sample):
    from ayase.modules.videval import VIDEVALModule

    video_sample.quality_metrics = QualityMetrics()
    m = VIDEVALModule()
    result = m.process(video_sample)
    assert result.quality_metrics.videval_score is not None
    assert 0.0 <= result.quality_metrics.videval_score <= 1.0


def test_videval_image(image_sample):
    from ayase.modules.videval import VIDEVALModule

    image_sample.quality_metrics = QualityMetrics()
    m = VIDEVALModule()
    result = m.process(image_sample)
    assert result.quality_metrics.videval_score is not None


def test_tlvqm_basics():
    from ayase.modules.tlvqm import TLVQMModule
    from .conftest import _test_module_basics

    _test_module_basics(TLVQMModule, "tlvqm")


def test_tlvqm_video(video_sample):
    from ayase.modules.tlvqm import TLVQMModule

    video_sample.quality_metrics = QualityMetrics()
    m = TLVQMModule()
    result = m.process(video_sample)
    assert result.quality_metrics.tlvqm_score is not None
    assert 0.0 <= result.quality_metrics.tlvqm_score <= 1.0


def test_funque_basics():
    from ayase.modules.funque import FUNQUEModule
    from .conftest import _test_module_basics

    _test_module_basics(FUNQUEModule, "funque")


def test_funque_video(video_sample):
    from ayase.modules.funque import FUNQUEModule

    video_sample.quality_metrics = QualityMetrics()
    m = FUNQUEModule()
    result = m.process(video_sample)
    assert result.quality_metrics.funque_score is not None
    assert 0.0 <= result.quality_metrics.funque_score <= 1.0


def test_movie_basics():
    from ayase.modules.movie import MOVIEModule
    from .conftest import _test_module_basics

    _test_module_basics(MOVIEModule, "movie")


def test_movie_video(video_sample):
    from ayase.modules.movie import MOVIEModule

    video_sample.quality_metrics = QualityMetrics()
    m = MOVIEModule()
    result = m.process(video_sample)
    assert result.quality_metrics.movie_score is not None
    assert 0.0 <= result.quality_metrics.movie_score <= 1.0


def test_st_greed_basics():
    from ayase.modules.st_greed import STGREEDModule
    from .conftest import _test_module_basics

    _test_module_basics(STGREEDModule, "st_greed")


def test_st_greed_video(video_sample):
    from ayase.modules.st_greed import STGREEDModule

    video_sample.quality_metrics = QualityMetrics()
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
    from ayase.modules.c3dvqa import C3DVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = C3DVQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics.c3dvqa_score is not None
    assert 0.0 <= result.quality_metrics.c3dvqa_score <= 1.0


def test_flolpips_basics():
    from ayase.modules.flolpips import FloLPIPSModule
    from .conftest import _test_module_basics

    _test_module_basics(FloLPIPSModule, "flolpips")


def test_flolpips_video(video_sample):
    from ayase.modules.flolpips import FloLPIPSModule

    video_sample.quality_metrics = QualityMetrics()
    m = FloLPIPSModule()
    result = m.process(video_sample)
    assert result.quality_metrics.flolpips is not None
    assert 0.0 <= result.quality_metrics.flolpips <= 1.0


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
    from ayase.modules.hdr_vqm import HDRVQMModule

    video_sample.quality_metrics = QualityMetrics()
    m = HDRVQMModule()
    result = m.process(video_sample)
    assert result.quality_metrics.hdr_vqm is not None
    assert 0.0 <= result.quality_metrics.hdr_vqm <= 1.0


def test_hdr_vqm_image(image_sample):
    from ayase.modules.hdr_vqm import HDRVQMModule

    image_sample.quality_metrics = QualityMetrics()
    m = HDRVQMModule()
    result = m.process(image_sample)
    assert result.quality_metrics.hdr_vqm is not None


def test_st_lpips_basics():
    from ayase.modules.st_lpips import STLPIPSModule
    from .conftest import _test_module_basics

    _test_module_basics(STLPIPSModule, "st_lpips")


def test_st_lpips_video(video_sample):
    from ayase.modules.st_lpips import STLPIPSModule

    video_sample.quality_metrics = QualityMetrics()
    m = STLPIPSModule()
    result = m.process(video_sample)
    assert result.quality_metrics.st_lpips is not None
    assert 0.0 <= result.quality_metrics.st_lpips <= 1.0


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


def test_rqvqa_dispatches_to_real_model_when_loaded():
    """RQ-VQA module dispatches to real model when backend=='rqvqa'."""
    from ayase.modules.rqvqa import RQVQAModule

    module = RQVQAModule()
    module._backend = "rqvqa"
    module._ml_available = True
    module._device = "cpu"

    mock_model = MagicMock()
    mock_model.return_value = MagicMock(item=MagicMock(return_value=0.8))
    module._model = mock_model

    frame = np.full((64, 64, 3), 128, dtype=np.uint8)

    score = module._process_rqvqa_model(
        Sample(path=Path("test.png"), is_video=False), [frame]
    )
    assert score is not None
    mock_model.assert_called_once()

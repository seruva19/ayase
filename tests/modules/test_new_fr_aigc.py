"""Tests for new FR (full-reference) and AIGC video quality modules."""

import numpy as np
import pytest

from ayase.models import QualityMetrics, Sample, VideoMetadata

from .conftest import _test_module_basics


# ===================================================================== #
# FR modules (need reference)                                           #
# ===================================================================== #


# --- RankDVQA -------------------------------------------------------- #


def test_rankdvqa_basics():
    from ayase.modules.rankdvqa import RankDVQAModule

    _test_module_basics(RankDVQAModule, "rankdvqa")


def test_rankdvqa_no_reference(image_sample):
    from ayase.modules.rankdvqa import RankDVQAModule

    m = RankDVQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.rankdvqa_score is None


def test_rankdvqa_with_reference(synthetic_image):
    from ayase.modules.rankdvqa import RankDVQAModule

    sample = Sample(path=synthetic_image, is_video=False, reference_path=synthetic_image)
    sample.quality_metrics = QualityMetrics()
    m = RankDVQAModule()
    result = m.process(sample)
    assert result.quality_metrics.rankdvqa_score is not None


# --- CompressedVQAHDR ------------------------------------------------ #


def test_compressed_vqa_hdr_basics():
    from ayase.modules.compressed_vqa_hdr import CompressedVQAHDRModule

    _test_module_basics(CompressedVQAHDRModule, "compressed_vqa_hdr")


def test_compressed_vqa_hdr_no_reference(image_sample):
    from ayase.modules.compressed_vqa_hdr import CompressedVQAHDRModule

    m = CompressedVQAHDRModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.compressed_vqa_hdr is None


def test_compressed_vqa_hdr_with_reference(synthetic_image):
    from ayase.modules.compressed_vqa_hdr import CompressedVQAHDRModule

    sample = Sample(path=synthetic_image, is_video=False, reference_path=synthetic_image)
    sample.quality_metrics = QualityMetrics()
    m = CompressedVQAHDRModule()
    result = m.process(sample)
    assert result.quality_metrics.compressed_vqa_hdr is not None


# --- ST-MAD ---------------------------------------------------------- #


def test_st_mad_basics():
    from ayase.modules.st_mad import STMADModule

    _test_module_basics(STMADModule, "st_mad")


def test_st_mad_no_reference(image_sample):
    from ayase.modules.st_mad import STMADModule

    m = STMADModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.st_mad is None


def test_st_mad_with_reference(synthetic_image):
    from ayase.modules.st_mad import STMADModule

    sample = Sample(path=synthetic_image, is_video=False, reference_path=synthetic_image)
    sample.quality_metrics = QualityMetrics()
    m = STMADModule()
    result = m.process(sample)
    assert result.quality_metrics.st_mad is not None


# --- Spherical PSNR -------------------------------------------------- #


def test_spherical_psnr_basics():
    from ayase.modules.spherical_psnr import SphericalPSNRModule

    _test_module_basics(SphericalPSNRModule, "spherical_psnr")


def test_spherical_psnr_no_reference(image_sample):
    from ayase.modules.spherical_psnr import SphericalPSNRModule

    m = SphericalPSNRModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.s_psnr is None


def test_spherical_psnr_with_reference(synthetic_image):
    from ayase.modules.spherical_psnr import SphericalPSNRModule

    sample = Sample(path=synthetic_image, is_video=False, reference_path=synthetic_image)
    sample.quality_metrics = QualityMetrics()
    m = SphericalPSNRModule()
    result = m.process(sample)
    assert result.quality_metrics.s_psnr is not None
    assert result.quality_metrics.ws_psnr is not None
    assert result.quality_metrics.cpp_psnr is not None


# --- WS-SSIM --------------------------------------------------------- #


def test_ws_ssim_basics():
    from ayase.modules.ws_ssim import WSSSIMModule

    _test_module_basics(WSSSIMModule, "ws_ssim")


def test_ws_ssim_no_reference(image_sample):
    from ayase.modules.ws_ssim import WSSSIMModule

    m = WSSSIMModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.ws_ssim is None


def test_ws_ssim_with_reference(synthetic_image):
    from ayase.modules.ws_ssim import WSSSIMModule

    sample = Sample(path=synthetic_image, is_video=False, reference_path=synthetic_image)
    sample.quality_metrics = QualityMetrics()
    m = WSSSIMModule()
    result = m.process(sample)
    assert result.quality_metrics.ws_ssim is not None


# ===================================================================== #
# AIGC / NR video quality modules                                       #
# ===================================================================== #


# --- CRAVE (video only) ---------------------------------------------- #


def test_crave_basics():
    from ayase.modules.crave import CRAVEModule

    _test_module_basics(CRAVEModule, "crave")


def test_crave_image_skipped(image_sample):
    from ayase.modules.crave import CRAVEModule

    m = CRAVEModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.crave_score is None


def test_crave_video(video_sample):
    from ayase.modules.crave import CRAVEModule

    video_sample.quality_metrics = QualityMetrics()
    m = CRAVEModule()
    result = m.process(video_sample)
    assert result.quality_metrics.crave_score is not None


# --- AIGCVQA --------------------------------------------------------- #


def test_aigcvqa_basics():
    from ayase.modules.aigcvqa import AIGCVQAModule

    _test_module_basics(AIGCVQAModule, "aigcvqa")


def test_aigcvqa_video(video_sample):
    from ayase.modules.aigcvqa import AIGCVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = AIGCVQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics.aigcvqa_technical is not None
    assert result.quality_metrics.aigcvqa_aesthetic is not None
    assert result.quality_metrics.aigcvqa_alignment is not None


def test_aigcvqa_image(image_sample):
    from ayase.modules.aigcvqa import AIGCVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = AIGCVQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics.aigcvqa_technical is not None


# --- UGVQ ------------------------------------------------------------ #


def test_ugvq_basics():
    from ayase.modules.ugvq import UGVQModule

    _test_module_basics(UGVQModule, "ugvq")


def test_ugvq_video(video_sample):
    from ayase.modules.ugvq import UGVQModule

    video_sample.quality_metrics = QualityMetrics()
    m = UGVQModule()
    result = m.process(video_sample)
    assert result.quality_metrics.ugvq_score is not None


def test_ugvq_image(image_sample):
    from ayase.modules.ugvq import UGVQModule

    image_sample.quality_metrics = QualityMetrics()
    m = UGVQModule()
    result = m.process(image_sample)
    assert result.quality_metrics.ugvq_score is not None


# --- AIGVQA ---------------------------------------------------------- #


def test_aigvqa_basics():
    from ayase.modules.aigvqa import AIGVQAModule

    _test_module_basics(AIGVQAModule, "aigvqa")


def test_aigvqa_video(video_sample):
    from ayase.modules.aigvqa import AIGVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = AIGVQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics.aigvqa_score is not None


def test_aigvqa_image(image_sample):
    from ayase.modules.aigvqa import AIGVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = AIGVQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics.aigvqa_score is not None


# --- WorldConsistency (video only) ----------------------------------- #


def test_world_consistency_basics():
    from ayase.modules.world_consistency import WorldConsistencyModule

    _test_module_basics(WorldConsistencyModule, "world_consistency")


def test_world_consistency_image_skipped(image_sample):
    from ayase.modules.world_consistency import WorldConsistencyModule

    m = WorldConsistencyModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.world_consistency_score is None


def test_world_consistency_video(video_sample):
    from ayase.modules.world_consistency import WorldConsistencyModule

    video_sample.quality_metrics = QualityMetrics()
    m = WorldConsistencyModule()
    result = m.process(video_sample)
    assert result.quality_metrics.world_consistency_score is not None


# --- VideoReward (video only) ---------------------------------------- #


def test_videoreward_basics():
    from ayase.modules.videoreward import VideoRewardModule

    _test_module_basics(VideoRewardModule, "videoreward")


def test_videoreward_image_skipped(image_sample):
    from ayase.modules.videoreward import VideoRewardModule

    m = VideoRewardModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.videoreward_vq is None


def test_videoreward_video(video_sample):
    from ayase.modules.videoreward import VideoRewardModule

    video_sample.quality_metrics = QualityMetrics()
    m = VideoRewardModule()
    result = m.process(video_sample)
    assert result.quality_metrics.videoreward_vq is not None
    assert result.quality_metrics.videoreward_mq is not None
    assert result.quality_metrics.videoreward_ta is not None


# --- VADER ----------------------------------------------------------- #


def test_vader_basics():
    from ayase.modules.vader import VADERModule

    _test_module_basics(VADERModule, "vader")


def test_vader_video(video_sample):
    from ayase.modules.vader import VADERModule

    video_sample.quality_metrics = QualityMetrics()
    m = VADERModule()
    result = m.process(video_sample)
    assert result.quality_metrics.vader_score is not None


def test_vader_image(image_sample):
    from ayase.modules.vader import VADERModule

    image_sample.quality_metrics = QualityMetrics()
    m = VADERModule()
    result = m.process(image_sample)
    assert result.quality_metrics.vader_score is not None


# --- VQA2 ------------------------------------------------------------ #


def test_vqa2_basics():
    from ayase.modules.vqa2 import VQA2Module

    _test_module_basics(VQA2Module, "vqa2")


def test_vqa2_video(video_sample):
    from ayase.modules.vqa2 import VQA2Module

    video_sample.quality_metrics = QualityMetrics()
    m = VQA2Module()
    result = m.process(video_sample)
    assert result.quality_metrics.vqa2_score is not None


def test_vqa2_image(image_sample):
    from ayase.modules.vqa2 import VQA2Module

    image_sample.quality_metrics = QualityMetrics()
    m = VQA2Module()
    result = m.process(image_sample)
    assert result.quality_metrics.vqa2_score is not None


# --- T2VEval --------------------------------------------------------- #


def test_t2veval_basics():
    from ayase.modules.t2veval import T2VEvalModule

    _test_module_basics(T2VEvalModule, "t2veval")


def test_t2veval_video(video_sample):
    from ayase.modules.t2veval import T2VEvalModule

    video_sample.quality_metrics = QualityMetrics()
    m = T2VEvalModule()
    result = m.process(video_sample)
    assert result.quality_metrics.t2veval_score is not None


def test_t2veval_image(image_sample):
    from ayase.modules.t2veval import T2VEvalModule

    image_sample.quality_metrics = QualityMetrics()
    m = T2VEvalModule()
    result = m.process(image_sample)
    assert result.quality_metrics.t2veval_score is not None


# --- SR4KVQA --------------------------------------------------------- #


def test_sr4kvqa_basics():
    from ayase.modules.sr4kvqa import SR4KVQAModule

    _test_module_basics(SR4KVQAModule, "sr4kvqa")


def test_sr4kvqa_video(video_sample):
    from ayase.modules.sr4kvqa import SR4KVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = SR4KVQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics.sr4kvqa_score is not None


def test_sr4kvqa_image(image_sample):
    from ayase.modules.sr4kvqa import SR4KVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = SR4KVQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics.sr4kvqa_score is not None


# --- OAVQA ----------------------------------------------------------- #


def test_oavqa_basics():
    from ayase.modules.oavqa import OAVQAModule

    _test_module_basics(OAVQAModule, "oavqa")


def test_oavqa_video(video_sample):
    from ayase.modules.oavqa import OAVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = OAVQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics.oavqa_score is not None


def test_oavqa_image(image_sample):
    from ayase.modules.oavqa import OAVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = OAVQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics.oavqa_score is not None


# --- SQI (needs VideoMetadata) --------------------------------------- #


def test_sqi_basics():
    from ayase.modules.sqi import SQIModule

    _test_module_basics(SQIModule, "sqi")


def test_sqi_no_metadata(video_sample):
    from ayase.modules.sqi import SQIModule

    m = SQIModule()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.sqi_score is None


def test_sqi_with_metadata(synthetic_video):
    from ayase.modules.sqi import SQIModule

    sample = Sample(
        path=synthetic_video,
        is_video=True,
        video_metadata=VideoMetadata(
            width=1920,
            height=1080,
            frame_count=300,
            fps=30.0,
            duration=10.0,
            file_size=5000000,
        ),
    )
    sample.quality_metrics = QualityMetrics()
    m = SQIModule()
    result = m.process(sample)
    assert result.quality_metrics.sqi_score is not None


# ===================================================================== #
# Field existence checks                                                #
# ===================================================================== #


def test_new_fr_aigc_fields_exist():
    """Verify all expected fields exist on QualityMetrics."""
    qm = QualityMetrics()
    for field in [
        "rankdvqa_score",
        "compressed_vqa_hdr",
        "st_mad",
        "s_psnr",
        "ws_psnr",
        "cpp_psnr",
        "ws_ssim",
    ]:
        assert hasattr(qm, field), f"QualityMetrics missing FR field: {field}"
    for field in [
        "crave_score",
        "aigcvqa_technical",
        "aigcvqa_aesthetic",
        "aigcvqa_alignment",
        "ugvq_score",
        "aigvqa_score",
        "world_consistency_score",
        "videoreward_vq",
        "videoreward_mq",
        "videoreward_ta",
        "vader_score",
        "vqa2_score",
        "t2veval_score",
        "sr4kvqa_score",
        "oavqa_score",
        "sqi_score",
    ]:
        assert hasattr(qm, field), f"QualityMetrics missing AIGC field: {field}"

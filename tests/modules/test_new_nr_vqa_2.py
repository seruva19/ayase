"""Tests for new NR-VQA modules batch 2: FAVER, SimpleVQA, SiamVQA, MemoryVQA,
SAMA, CLiFVQA, AdaDQA, MDVQA, UIQM, UCIQE."""

import math

from ayase.models import QualityMetrics

from .conftest import _test_module_basics


# ===================================================================== #
# FAVER (video only)                                                    #
# ===================================================================== #


def test_faver_basics():
    from ayase.modules.faver import FAVERModule

    _test_module_basics(FAVERModule, "faver")


def test_faver_video(video_sample):
    from ayase.modules.faver import FAVERModule

    video_sample.quality_metrics = QualityMetrics()
    m = FAVERModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.faver_score is not None
    assert 0.0 <= result.quality_metrics.faver_score <= 1.0


def test_faver_image(image_sample):
    from ayase.modules.faver import FAVERModule

    image_sample.quality_metrics = QualityMetrics()
    m = FAVERModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.faver_score is None


# ===================================================================== #
# SimpleVQA                                                             #
# ===================================================================== #


def test_simplevqa_basics():
    from ayase.modules.simplevqa import SimpleVQAModule

    _test_module_basics(SimpleVQAModule, "simplevqa")


def test_simplevqa_video(video_sample):
    from ayase.modules.simplevqa import SimpleVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = SimpleVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.simplevqa_score is not None
    assert 0.0 <= result.quality_metrics.simplevqa_score <= 1.0


def test_simplevqa_image(image_sample):
    from ayase.modules.simplevqa import SimpleVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = SimpleVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.simplevqa_score is not None


# ===================================================================== #
# SiamVQA                                                               #
# ===================================================================== #


def test_siamvqa_basics():
    from ayase.modules.siamvqa import SiamVQAModule

    _test_module_basics(SiamVQAModule, "siamvqa")


def test_siamvqa_video(video_sample):
    from ayase.modules.siamvqa import SiamVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = SiamVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.siamvqa_score is not None
    assert 0.0 <= result.quality_metrics.siamvqa_score <= 1.0


def test_siamvqa_image(image_sample):
    from ayase.modules.siamvqa import SiamVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = SiamVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.siamvqa_score is not None


# ===================================================================== #
# MemoryVQA                                                             #
# ===================================================================== #


def test_memoryvqa_basics():
    from ayase.modules.memoryvqa import MemoryVQAModule

    _test_module_basics(MemoryVQAModule, "memoryvqa")


def test_memoryvqa_video(video_sample):
    from ayase.modules.memoryvqa import MemoryVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = MemoryVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.memoryvqa_score is not None
    assert 0.0 <= result.quality_metrics.memoryvqa_score <= 1.0


def test_memoryvqa_image(image_sample):
    from ayase.modules.memoryvqa import MemoryVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = MemoryVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.memoryvqa_score is not None


# ===================================================================== #
# SAMA                                                                  #
# ===================================================================== #


def test_sama_basics():
    from ayase.modules.sama import SAMAModule

    _test_module_basics(SAMAModule, "sama")


def test_sama_video(video_sample):
    from ayase.modules.sama import SAMAModule

    video_sample.quality_metrics = QualityMetrics()
    m = SAMAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.sama_score is not None
    assert 0.0 <= result.quality_metrics.sama_score <= 1.0


def test_sama_image(image_sample):
    from ayase.modules.sama import SAMAModule

    image_sample.quality_metrics = QualityMetrics()
    m = SAMAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.sama_score is not None


# ===================================================================== #
# CLiFVQA                                                               #
# ===================================================================== #


def test_clifvqa_basics():
    from ayase.modules.clifvqa import CLiFVQAModule

    _test_module_basics(CLiFVQAModule, "clifvqa")


def test_clifvqa_video(video_sample):
    from ayase.modules.clifvqa import CLiFVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = CLiFVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.clifvqa_score is not None
    assert 0.0 <= result.quality_metrics.clifvqa_score <= 1.0


def test_clifvqa_image(image_sample):
    from ayase.modules.clifvqa import CLiFVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = CLiFVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.clifvqa_score is not None


# ===================================================================== #
# AdaDQA                                                                #
# ===================================================================== #


def test_adadqa_basics():
    from ayase.modules.adadqa import AdaDQAModule

    _test_module_basics(AdaDQAModule, "adadqa")


def test_adadqa_video(video_sample):
    from ayase.modules.adadqa import AdaDQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = AdaDQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.adadqa_score is not None
    assert 0.0 <= result.quality_metrics.adadqa_score <= 1.0


def test_adadqa_image(image_sample):
    from ayase.modules.adadqa import AdaDQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = AdaDQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.adadqa_score is not None


# ===================================================================== #
# MDVQA (3 output fields)                                               #
# ===================================================================== #


def test_mdvqa_basics():
    from ayase.modules.mdvqa import MDVQAModule

    _test_module_basics(MDVQAModule, "mdvqa")


def test_mdvqa_video(video_sample):
    from ayase.modules.mdvqa import MDVQAModule

    video_sample.quality_metrics = QualityMetrics()
    m = MDVQAModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.mdvqa_semantic is not None
    assert 0.0 <= result.quality_metrics.mdvqa_semantic <= 1.0
    assert result.quality_metrics.mdvqa_distortion is not None
    assert 0.0 <= result.quality_metrics.mdvqa_distortion <= 1.0
    assert result.quality_metrics.mdvqa_motion is not None
    assert 0.0 <= result.quality_metrics.mdvqa_motion <= 1.0


def test_mdvqa_image(image_sample):
    from ayase.modules.mdvqa import MDVQAModule

    image_sample.quality_metrics = QualityMetrics()
    m = MDVQAModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.mdvqa_semantic is not None
    assert result.quality_metrics.mdvqa_distortion is not None
    assert result.quality_metrics.mdvqa_motion is not None


# ===================================================================== #
# UIQM (unbounded score)                                                #
# ===================================================================== #


def test_uiqm_basics():
    from ayase.modules.uiqm import UIQMModule

    _test_module_basics(UIQMModule, "uiqm")


def test_uiqm_video(video_sample):
    from ayase.modules.uiqm import UIQMModule

    video_sample.quality_metrics = QualityMetrics()
    m = UIQMModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.uiqm_score is not None
    assert math.isfinite(result.quality_metrics.uiqm_score)


def test_uiqm_image(image_sample):
    from ayase.modules.uiqm import UIQMModule

    image_sample.quality_metrics = QualityMetrics()
    m = UIQMModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.uiqm_score is not None
    assert math.isfinite(result.quality_metrics.uiqm_score)


# ===================================================================== #
# UCIQE (unbounded score)                                               #
# ===================================================================== #


def test_uciqe_basics():
    from ayase.modules.uciqe import UCIQEModule

    _test_module_basics(UCIQEModule, "uciqe")


def test_uciqe_video(video_sample):
    from ayase.modules.uciqe import UCIQEModule

    video_sample.quality_metrics = QualityMetrics()
    m = UCIQEModule()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics.uciqe_score is not None
    assert math.isfinite(result.quality_metrics.uciqe_score)


def test_uciqe_image(image_sample):
    from ayase.modules.uciqe import UCIQEModule

    image_sample.quality_metrics = QualityMetrics()
    m = UCIQEModule()
    m.on_mount()
    result = m.process(image_sample)
    assert result.quality_metrics.uciqe_score is not None
    assert math.isfinite(result.quality_metrics.uciqe_score)

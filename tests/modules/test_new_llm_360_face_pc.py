"""Tests for LLM/VLM, 360/VR, Face Quality, Streaming QoE, Point Cloud,
and Distribution modules (20 modules)."""

import numpy as np

from ayase.models import QualityMetrics, Sample


# ---------------------------------------------------------------------------
# LLM/VLM modules (5)
# ---------------------------------------------------------------------------


def test_lmmvqa_basics():
    from ayase.modules.lmmvqa import LMMVQAModule
    from .conftest import _test_module_basics

    _test_module_basics(LMMVQAModule, "lmmvqa")


def test_lmmvqa_image(image_sample):
    from ayase.modules.lmmvqa import LMMVQAModule

    m = LMMVQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.lmmvqa_score is not None
    assert 0.0 <= result.quality_metrics.lmmvqa_score <= 1.0


def test_lmmvqa_video(video_sample):
    from ayase.modules.lmmvqa import LMMVQAModule

    m = LMMVQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.lmmvqa_score is not None
    assert 0.0 <= result.quality_metrics.lmmvqa_score <= 1.0


def test_vqinsight_basics():
    from ayase.modules.vqinsight import VQInsightModule
    from .conftest import _test_module_basics

    _test_module_basics(VQInsightModule, "vqinsight")


def test_vqinsight_image(image_sample):
    from ayase.modules.vqinsight import VQInsightModule

    m = VQInsightModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.vqinsight_score is not None
    assert 0.0 <= result.quality_metrics.vqinsight_score <= 1.0


def test_vqinsight_video(video_sample):
    from ayase.modules.vqinsight import VQInsightModule

    m = VQInsightModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.vqinsight_score is not None
    assert 0.0 <= result.quality_metrics.vqinsight_score <= 1.0


def test_vqathinker_basics():
    from ayase.modules.vqathinker import VQAThinkerModule
    from .conftest import _test_module_basics

    _test_module_basics(VQAThinkerModule, "vqathinker")


def test_vqathinker_image(image_sample):
    from ayase.modules.vqathinker import VQAThinkerModule

    m = VQAThinkerModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.vqathinker_score is not None
    assert 0.0 <= result.quality_metrics.vqathinker_score <= 1.0


def test_vqathinker_video(video_sample):
    from ayase.modules.vqathinker import VQAThinkerModule

    m = VQAThinkerModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.vqathinker_score is not None
    assert 0.0 <= result.quality_metrics.vqathinker_score <= 1.0


def test_qclip_basics():
    from ayase.modules.qclip import QCLIPModule
    from .conftest import _test_module_basics

    _test_module_basics(QCLIPModule, "qclip")


def test_qclip_image(image_sample):
    from ayase.modules.qclip import QCLIPModule

    m = QCLIPModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.qclip_score is not None
    assert 0.0 <= result.quality_metrics.qclip_score <= 1.0


def test_qclip_video(video_sample):
    from ayase.modules.qclip import QCLIPModule

    m = QCLIPModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.qclip_score is not None
    assert 0.0 <= result.quality_metrics.qclip_score <= 1.0


def test_presresq_basics():
    from ayase.modules.presresq import PreResQModule
    from .conftest import _test_module_basics

    _test_module_basics(PreResQModule, "presresq")


def test_presresq_image(image_sample):
    from ayase.modules.presresq import PreResQModule

    m = PreResQModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.presresq_score is not None
    assert 0.0 <= result.quality_metrics.presresq_score <= 1.0


def test_presresq_video(video_sample):
    from ayase.modules.presresq import PreResQModule

    m = PreResQModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.presresq_score is not None
    assert 0.0 <= result.quality_metrics.presresq_score <= 1.0


# ---------------------------------------------------------------------------
# 360/VR modules (2)
# ---------------------------------------------------------------------------


def test_mc360iqa_basics():
    from ayase.modules.mc360iqa import MC360IQAModule
    from .conftest import _test_module_basics

    _test_module_basics(MC360IQAModule, "mc360iqa")


def test_mc360iqa_image(image_sample):
    from ayase.modules.mc360iqa import MC360IQAModule

    m = MC360IQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.mc360iqa_score is not None
    assert 0.0 <= result.quality_metrics.mc360iqa_score <= 1.0


def test_mc360iqa_video(video_sample):
    from ayase.modules.mc360iqa import MC360IQAModule

    m = MC360IQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.mc360iqa_score is not None
    assert 0.0 <= result.quality_metrics.mc360iqa_score <= 1.0


def test_provqa_basics():
    from ayase.modules.provqa import ProVQAModule
    from .conftest import _test_module_basics

    _test_module_basics(ProVQAModule, "provqa")


def test_provqa_image(image_sample):
    from ayase.modules.provqa import ProVQAModule

    m = ProVQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.provqa_score is not None
    assert 0.0 <= result.quality_metrics.provqa_score <= 1.0


def test_provqa_video(video_sample):
    from ayase.modules.provqa import ProVQAModule

    m = ProVQAModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.provqa_score is not None
    assert 0.0 <= result.quality_metrics.provqa_score <= 1.0


# ---------------------------------------------------------------------------
# Face Quality modules (4)
# ---------------------------------------------------------------------------


def test_serfiq_basics():
    from ayase.modules.serfiq import SERFIQModule
    from .conftest import _test_module_basics

    _test_module_basics(SERFIQModule, "serfiq")


def test_serfiq_image(image_sample):
    from ayase.modules.serfiq import SERFIQModule

    m = SERFIQModule()
    m.on_mount()
    result = m.process(image_sample)
    # Synthetic gradient image may not have detectable faces
    if result.quality_metrics is not None and result.quality_metrics.serfiq_score is not None:
        assert 0.0 <= result.quality_metrics.serfiq_score <= 1.0


def test_serfiq_video(video_sample):
    from ayase.modules.serfiq import SERFIQModule

    m = SERFIQModule()
    m.on_mount()
    result = m.process(video_sample)
    if result.quality_metrics is not None and result.quality_metrics.serfiq_score is not None:
        assert 0.0 <= result.quality_metrics.serfiq_score <= 1.0


def test_crfiqa_basics():
    from ayase.modules.crfiqa import CRFIQAModule
    from .conftest import _test_module_basics

    _test_module_basics(CRFIQAModule, "crfiqa")


def test_crfiqa_image(image_sample):
    from ayase.modules.crfiqa import CRFIQAModule

    m = CRFIQAModule()
    m.on_mount()
    result = m.process(image_sample)
    if result.quality_metrics is not None and result.quality_metrics.crfiqa_score is not None:
        assert 0.0 <= result.quality_metrics.crfiqa_score <= 1.0


def test_crfiqa_video(video_sample):
    from ayase.modules.crfiqa import CRFIQAModule

    m = CRFIQAModule()
    m.on_mount()
    result = m.process(video_sample)
    if result.quality_metrics is not None and result.quality_metrics.crfiqa_score is not None:
        assert 0.0 <= result.quality_metrics.crfiqa_score <= 1.0


def test_magface_basics():
    from ayase.modules.magface import MagFaceModule
    from .conftest import _test_module_basics

    _test_module_basics(MagFaceModule, "magface")


def test_magface_image(image_sample):
    from ayase.modules.magface import MagFaceModule

    m = MagFaceModule()
    m.on_mount()
    result = m.process(image_sample)
    if result.quality_metrics is not None and result.quality_metrics.magface_score is not None:
        assert 0.0 <= result.quality_metrics.magface_score <= 1.0


def test_magface_video(video_sample):
    from ayase.modules.magface import MagFaceModule

    m = MagFaceModule()
    m.on_mount()
    result = m.process(video_sample)
    if result.quality_metrics is not None and result.quality_metrics.magface_score is not None:
        assert 0.0 <= result.quality_metrics.magface_score <= 1.0


def test_grafiqs_basics():
    from ayase.modules.grafiqs import GraFIQsModule
    from .conftest import _test_module_basics

    _test_module_basics(GraFIQsModule, "grafiqs")


def test_grafiqs_image(image_sample):
    from ayase.modules.grafiqs import GraFIQsModule

    m = GraFIQsModule()
    m.on_mount()
    result = m.process(image_sample)
    if result.quality_metrics is not None and result.quality_metrics.grafiqs_score is not None:
        assert 0.0 <= result.quality_metrics.grafiqs_score <= 1.0


def test_grafiqs_video(video_sample):
    from ayase.modules.grafiqs import GraFIQsModule

    m = GraFIQsModule()
    m.on_mount()
    result = m.process(video_sample)
    if result.quality_metrics is not None and result.quality_metrics.grafiqs_score is not None:
        assert 0.0 <= result.quality_metrics.grafiqs_score <= 1.0


# ---------------------------------------------------------------------------
# Streaming QoE modules (2)
# ---------------------------------------------------------------------------


def test_p1204_basics():
    from ayase.modules.p1204 import P1204Module
    from .conftest import _test_module_basics

    _test_module_basics(P1204Module, "p1204")


def test_p1204_image(image_sample):
    from ayase.modules.p1204 import P1204Module

    m = P1204Module()
    m.on_mount()
    result = m.process(image_sample)
    # P.1204 is video-only; image should produce no score
    assert result.quality_metrics is None or result.quality_metrics.p1204_mos is None


def test_p1204_no_metadata(video_sample):
    from ayase.modules.p1204 import P1204Module

    m = P1204Module()
    m.on_mount()
    result = m.process(video_sample)
    assert result.quality_metrics is None or result.quality_metrics.p1204_mos is None


def test_p1204_with_metadata(synthetic_video):
    from ayase.models import VideoMetadata
    from ayase.modules.p1204 import P1204Module

    m = P1204Module()
    m.on_mount()
    vm = VideoMetadata(
        width=1920,
        height=1080,
        frame_count=900,
        fps=30.0,
        duration=30.0,
        file_size=10_000_000,
        bitrate=5_000_000,
        codec="h264",
    )
    sample = Sample(path=synthetic_video, is_video=True, video_metadata=vm)
    result = m.process(sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.p1204_mos is not None
    assert 1.0 <= result.quality_metrics.p1204_mos <= 5.0


def test_video_atlas_basics():
    from ayase.modules.video_atlas import VideoATLASModule
    from .conftest import _test_module_basics

    _test_module_basics(VideoATLASModule, "video_atlas")


def test_video_atlas_image(image_sample):
    from ayase.modules.video_atlas import VideoATLASModule

    m = VideoATLASModule()
    result = m.process(image_sample)
    # Video-only module; image should produce no score
    assert result.quality_metrics is None or result.quality_metrics.video_atlas_score is None


def test_video_atlas_video(video_sample):
    from ayase.modules.video_atlas import VideoATLASModule

    m = VideoATLASModule()
    result = m.process(video_sample)
    assert result.quality_metrics is not None
    assert result.quality_metrics.video_atlas_score is not None
    assert 0.0 <= result.quality_metrics.video_atlas_score <= 1.0


# ---------------------------------------------------------------------------
# Point Cloud modules (5) — basics only (no .ply fixtures)
# ---------------------------------------------------------------------------


def test_pc_psnr_basics():
    from ayase.modules.pc_psnr import PCPSNRModule
    from .conftest import _test_module_basics

    _test_module_basics(PCPSNRModule, "pc_psnr")


def test_pc_psnr_no_reference(image_sample):
    from ayase.modules.pc_psnr import PCPSNRModule

    m = PCPSNRModule()
    result = m.process(image_sample)
    # No reference and not a .ply file — should produce no score
    assert result.quality_metrics is None or result.quality_metrics.pc_d1_psnr is None


def test_pcqm_basics():
    from ayase.modules.pcqm import PCQMModule
    from .conftest import _test_module_basics

    _test_module_basics(PCQMModule, "pcqm")


def test_pcqm_no_reference(image_sample):
    from ayase.modules.pcqm import PCQMModule

    m = PCQMModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.pcqm_score is None


def test_graphsim_basics():
    from ayase.modules.graphsim import GraphSIMModule
    from .conftest import _test_module_basics

    _test_module_basics(GraphSIMModule, "graphsim")


def test_graphsim_no_reference(image_sample):
    from ayase.modules.graphsim import GraphSIMModule

    m = GraphSIMModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.graphsim_score is None


def test_pointssim_basics():
    from ayase.modules.pointssim import PointSSIMModule
    from .conftest import _test_module_basics

    _test_module_basics(PointSSIMModule, "pointssim")


def test_pointssim_no_reference(image_sample):
    from ayase.modules.pointssim import PointSSIMModule

    m = PointSSIMModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.pointssim_score is None


def test_mm_pcqa_basics():
    from ayase.modules.mm_pcqa import MMPCQAModule
    from .conftest import _test_module_basics

    _test_module_basics(MMPCQAModule, "mm_pcqa")


def test_mm_pcqa_no_pointcloud(image_sample):
    from ayase.modules.mm_pcqa import MMPCQAModule

    m = MMPCQAModule()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.mm_pcqa_score is None


# ---------------------------------------------------------------------------
# Distribution / BatchMetricModule modules (2)
# ---------------------------------------------------------------------------


def test_stream_is_batch():
    from ayase.base_modules import BatchMetricModule
    from ayase.modules.stream_metric import STREAMModule

    assert issubclass(STREAMModule, BatchMetricModule)


def test_stream_basics():
    from ayase.modules.stream_metric import STREAMModule
    from .conftest import _test_module_basics

    _test_module_basics(STREAMModule, "stream_metric")


def test_stream_extract(video_sample):
    from ayase.modules.stream_metric import STREAMModule

    m = STREAMModule()
    feat = m.extract_features(video_sample)
    assert feat is not None
    assert len(feat) > 0


def test_worldscore_is_batch():
    from ayase.base_modules import BatchMetricModule
    from ayase.modules.worldscore import WorldScoreModule

    assert issubclass(WorldScoreModule, BatchMetricModule)


def test_worldscore_basics():
    from ayase.modules.worldscore import WorldScoreModule
    from .conftest import _test_module_basics

    _test_module_basics(WorldScoreModule, "worldscore")


def test_worldscore_extract(video_sample):
    from ayase.modules.worldscore import WorldScoreModule

    m = WorldScoreModule()
    feat = m.extract_features(video_sample)
    assert feat is not None
    assert len(feat) > 0


# ---------------------------------------------------------------------------
# QualityMetrics field existence checks
# ---------------------------------------------------------------------------


def test_llm_vlm_fields():
    qm = QualityMetrics()
    for field in ["lmmvqa_score", "vqinsight_score", "vqathinker_score",
                  "qclip_score", "presresq_score"]:
        assert hasattr(qm, field)


def test_360vr_fields():
    qm = QualityMetrics()
    for field in ["mc360iqa_score", "provqa_score"]:
        assert hasattr(qm, field)


def test_face_quality_fields():
    qm = QualityMetrics()
    for field in ["serfiq_score", "crfiqa_score", "magface_score", "grafiqs_score"]:
        assert hasattr(qm, field)


def test_streaming_qoe_fields():
    qm = QualityMetrics()
    for field in ["p1204_mos", "video_atlas_score"]:
        assert hasattr(qm, field)


def test_point_cloud_fields():
    qm = QualityMetrics()
    for field in ["pc_d1_psnr", "pc_d2_psnr", "pcqm_score", "graphsim_score",
                  "pointssim_score", "mm_pcqa_score"]:
        assert hasattr(qm, field)

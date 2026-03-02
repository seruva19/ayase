from ayase.models import QualityMetrics


def test_scene_detection_basics():
    from ayase.modules.scene_detection import SceneDetectionModule
    from .conftest import _test_module_basics

    _test_module_basics(SceneDetectionModule, "scene_detection")


def test_scene_detection_video(video_sample):
    from ayase.modules.scene_detection import SceneDetectionModule

    video_sample.quality_metrics = QualityMetrics()
    m = SceneDetectionModule()
    m.setup()
    result = m.process(video_sample)
    qm = result.quality_metrics
    assert qm is not None
    assert qm.scene_stability is not None
    assert 0 <= qm.scene_stability <= 1
    assert qm.avg_scene_duration is not None
    assert qm.avg_scene_duration > 0


def test_raft_motion_basics():
    from ayase.modules.raft_motion import RAFTMotionModule
    from .conftest import _test_module_basics

    _test_module_basics(RAFTMotionModule, "raft_motion")


def test_raft_motion_disabled(video_sample):
    from ayase.modules.raft_motion import RAFTMotionModule

    m = RAFTMotionModule({})
    result = m.process(video_sample)
    assert result is video_sample


def test_ram_tagging_basics():
    from ayase.modules.ram_tagging import RAMTaggingModule
    from .conftest import _test_module_basics

    _test_module_basics(RAMTaggingModule, "ram_tagging")


def test_ram_tagging_disabled(image_sample):
    from ayase.modules.ram_tagging import RAMTaggingModule

    m = RAMTaggingModule({})
    result = m.process(image_sample)
    assert result is image_sample


def test_depth_anything_basics():
    from ayase.modules.depth_anything import DepthAnythingModule
    from .conftest import _test_module_basics

    _test_module_basics(DepthAnythingModule, "depth_anything")


def test_depth_anything_disabled(video_sample):
    from ayase.modules.depth_anything import DepthAnythingModule

    m = DepthAnythingModule({})
    result = m.process(video_sample)
    assert result is video_sample


def test_video_type_classifier_basics():
    from ayase.modules.video_type_classifier import VideoTypeClassifierModule
    from .conftest import _test_module_basics

    _test_module_basics(VideoTypeClassifierModule, "video_type_classifier")


def test_video_type_classifier_disabled(video_sample):
    from ayase.modules.video_type_classifier import VideoTypeClassifierModule

    m = VideoTypeClassifierModule({})
    result = m.process(video_sample)
    assert result is video_sample


def test_jedi_basics():
    from ayase.modules.jedi_metric import JEDiModule
    from .conftest import _test_module_basics

    _test_module_basics(JEDiModule, "jedi")


def test_jedi_disabled(video_sample):
    from ayase.modules.jedi_metric import JEDiModule

    m = JEDiModule({})
    result = m.process(video_sample)
    assert result is video_sample


def test_trajan_basics():
    from ayase.modules.trajan import TRAJANModule
    from .conftest import _test_module_basics

    _test_module_basics(TRAJANModule, "trajan")


def test_trajan_video(video_sample):
    from ayase.modules.trajan import TRAJANModule

    video_sample.quality_metrics = QualityMetrics()
    m = TRAJANModule()
    m.setup()
    result = m.process(video_sample)
    qm = result.quality_metrics
    assert qm is not None
    assert qm.trajan_score is not None
    assert 0 <= qm.trajan_score <= 1


def test_trajan_image(image_sample):
    from ayase.modules.trajan import TRAJANModule

    m = TRAJANModule()
    m.setup()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.trajan_score is None


def test_promptiqa_basics():
    from ayase.modules.promptiqa import PromptIQAModule
    from .conftest import _test_module_basics

    _test_module_basics(PromptIQAModule, "promptiqa")


def test_promptiqa_disabled(image_sample):
    from ayase.modules.promptiqa import PromptIQAModule

    m = PromptIQAModule({})
    result = m.process(image_sample)
    assert result is image_sample


def test_aigv_assessor_basics():
    from ayase.modules.aigv_assessor import AIGVAssessorModule
    from .conftest import _test_module_basics

    _test_module_basics(AIGVAssessorModule, "aigv_assessor")


def test_aigv_assessor_video(video_sample):
    from ayase.modules.aigv_assessor import AIGVAssessorModule

    video_sample.quality_metrics = QualityMetrics()
    m = AIGVAssessorModule()
    m.setup()
    result = m.process(video_sample)
    qm = result.quality_metrics
    assert qm is not None
    assert qm.aigv_static is not None
    assert 0 <= qm.aigv_static <= 1
    assert qm.aigv_temporal is not None
    assert qm.aigv_dynamic is not None


def test_aigv_assessor_image(image_sample):
    from ayase.modules.aigv_assessor import AIGVAssessorModule

    m = AIGVAssessorModule()
    m.setup()
    result = m.process(image_sample)
    assert result.quality_metrics is None or result.quality_metrics.aigv_static is None


def test_video_reward_basics():
    from ayase.modules.video_reward import VideoRewardModule
    from .conftest import _test_module_basics

    _test_module_basics(VideoRewardModule, "video_reward")


def test_video_reward_disabled(video_sample):
    from ayase.modules.video_reward import VideoRewardModule

    m = VideoRewardModule({})
    result = m.process(video_sample)
    assert result is video_sample


def test_text_overlay_basics():
    from ayase.modules.text_overlay import TextOverlayModule
    from .conftest import _test_module_basics

    _test_module_basics(TextOverlayModule, "text_overlay")


def test_text_overlay_image(image_sample):
    from ayase.modules.text_overlay import TextOverlayModule

    image_sample.quality_metrics = QualityMetrics()
    m = TextOverlayModule()
    m.setup()
    result = m.process(image_sample)
    qm = result.quality_metrics
    assert qm is not None
    assert qm.text_overlay_score is not None
    assert 0 <= qm.text_overlay_score <= 1


def test_text_overlay_video(video_sample):
    from ayase.modules.text_overlay import TextOverlayModule

    video_sample.quality_metrics = QualityMetrics()
    m = TextOverlayModule()
    m.setup()
    result = m.process(video_sample)
    qm = result.quality_metrics
    assert qm is not None
    assert qm.text_overlay_score is not None
    assert 0 <= qm.text_overlay_score <= 1


def test_ptlflow_motion_basics():
    from ayase.modules.ptlflow_motion import PtlflowMotionModule
    from .conftest import _test_module_basics

    _test_module_basics(PtlflowMotionModule, "ptlflow_motion")


def test_ptlflow_motion_disabled(video_sample):
    from ayase.modules.ptlflow_motion import PtlflowMotionModule

    m = PtlflowMotionModule({})
    result = m.process(video_sample)
    assert result is video_sample


def test_qcn_basics():
    from ayase.modules.qcn import QCNModule
    from .conftest import _test_module_basics

    _test_module_basics(QCNModule, "qcn")


def test_qcn_disabled(image_sample):
    from ayase.modules.qcn import QCNModule

    m = QCNModule({})
    result = m.process(image_sample)
    assert result is image_sample


def test_motion_scene_qualitymetrics_fields():
    qm = QualityMetrics()
    fields = [
        "scene_stability",
        "avg_scene_duration",
        "raft_motion_score",
        "ptlflow_motion_score",
        "ram_tags",
        "depth_anything_score",
        "depth_anything_consistency",
        "video_type",
        "video_type_confidence",
        "jedi",
        "trajan_score",
        "promptiqa_score",
        "qcn_score",
        "aigv_static",
        "aigv_temporal",
        "aigv_dynamic",
        "aigv_alignment",
        "video_reward_score",
        "text_overlay_score",
    ]
    for field in fields:
        assert hasattr(qm, field)


def test_jedi_dataset_stats_field():
    from ayase.models import DatasetStats

    stats = DatasetStats(
        total_samples=10,
        valid_samples=10,
        invalid_samples=0,
        total_size=1000)
    assert hasattr(stats, "jedi")
    assert stats.jedi is None

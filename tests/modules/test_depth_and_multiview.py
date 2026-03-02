def test_depth_consistency_basics():
    from ayase.modules.depth_consistency import DepthConsistencyModule
    from .conftest import _test_module_basics

    _test_module_basics(DepthConsistencyModule, "depth_consistency")


def test_depth_map_quality_basics():
    from ayase.modules.depth_map_quality import DepthMapQualityModule
    from .conftest import _test_module_basics

    _test_module_basics(DepthMapQualityModule, "depth_map_quality")


def test_semantic_segmentation_consistency_basics():
    from ayase.modules.semantic_segmentation_consistency import (
        SemanticSegmentationConsistencyModule,
    )
    from .conftest import _test_module_basics

    _test_module_basics(
        SemanticSegmentationConsistencyModule,
        "semantic_segmentation_consistency",
    )


def test_multi_view_consistency_basics():
    from ayase.modules.multi_view_consistency import MultiViewConsistencyModule
    from .conftest import _test_module_basics

    _test_module_basics(MultiViewConsistencyModule, "multi_view_consistency")


def test_multi_view_consistency_video(video_sample):
    from ayase.modules.multi_view_consistency import MultiViewConsistencyModule

    m = MultiViewConsistencyModule()
    m.setup()
    result = m.process(video_sample)
    if result.quality_metrics is not None:
        mv = result.quality_metrics.multiview_consistency
        if mv is not None:
            assert 0 <= mv <= 1


def test_stereoscopic_quality_basics():
    from ayase.modules.stereoscopic_quality import StereoscopicQualityModule
    from .conftest import _test_module_basics

    _test_module_basics(StereoscopicQualityModule, "stereoscopic_quality")

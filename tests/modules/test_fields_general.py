import numpy as np

from ayase.models import DatasetStats, QualityMetrics, Sample, VideoMetadata


def test_quality_metrics_fields():
    qm = QualityMetrics()
    basic_fields = [
        "blur_score",
        "noise_score",
        "aesthetic_score",
        "technical_score",
        "motion_score",
        "camera_motion_score",
        "temporal_consistency",
        "contrast",
        "brightness",
        "saturation",
        "compression_score",
    ]
    for field in basic_fields:
        assert hasattr(qm, field)

    classic_metrics = [
        "fid_score",
        "kid_score",
        "inception_score",
        "ssim_score",
        "psnr_score",
        "lpips_score",
        "clip_score",
        "alignment_score",
    ]
    for field in classic_metrics:
        assert hasattr(qm, field)

    for field in ["human_preference_score", "engagement_score", "usability_score"]:
        assert hasattr(qm, field)


def test_quality_metrics_default_values():
    qm = QualityMetrics()
    assert qm.blur_score is None
    assert qm.noise_score is None
    assert qm.aesthetic_score is None
    assert qm.technical_score is None
    assert qm.motion_score is None


def test_dataset_stats_fields():
    stats = DatasetStats(
        total_samples=100, valid_samples=90, invalid_samples=10, total_size=1000000
    )
    assert stats.avg_technical_score is None
    assert stats.avg_aesthetic_score is None
    assert stats.avg_motion_score is None
    assert stats.usability_ratio is None
    assert stats.size_distribution is None
    assert stats.duration_distribution is None


def test_quality_metrics_new_fields_exist():
    qm = QualityMetrics()
    new_fields = [
        "fvd",
        "kvd",
        "fvmd",
        "vmaf",
        "ms_ssim",
        "vif",
        "niqe",
        "t2v_score",
        "t2v_alignment",
        "t2v_quality",
        "dynamics_range",
        "dynamics_controllability",
        "scene_complexity",
        "compression_artifacts",
        "naturalness_score",
        "video_memorability",
        "usability_rate",
        "confidence_score",
        "hdr_quality",
        "sdr_quality",
        "temporal_information",
        "spatial_information",
        "flicker_score",
        "judder_score",
        "stutter_score",
        "dists",
        "fsim",
        "gmsd",
        "vsi_score",
        "brisque",
        "pesq_score",
        "av_sync_offset",
        "dover_score",
        "dover_technical",
        "dover_aesthetic",
        "topiq_score",
        "liqe_score",
        "clip_iqa_score",
        "color_grading_score",
        "white_balance_score",
        "exposure_consistency",
        "focus_quality",
        "banding_severity",
        "qalign_quality",
        "qalign_aesthetic",
        "face_quality_score",
        "face_identity_consistency",
        "face_expression_smoothness",
        "face_landmark_jitter",
        "object_permanence_score",
        "semantic_consistency",
        "depth_temporal_consistency",
        "codec_efficiency",
        "gop_quality",
        "codec_artifacts",
        "deepfake_probability",
        "harmful_content_score",
        "watermark_strength",
        "bias_score",
        "depth_quality",
        "multiview_consistency",
        "stereo_comfort_score",
        "musiq_score",
        "contrique_score",
        "mdtvsfa_score",
    ]
    for field in new_fields:
        assert hasattr(qm, field)
        assert getattr(qm, field) is None


def test_quality_metrics_accept_values():
    qm = QualityMetrics(
        flicker_score=25.0,
        brisque=42.0,
        deepfake_probability=0.1,
        musiq_score=75.0,
    )
    assert qm.flicker_score == 25.0
    assert qm.brisque == 42.0
    assert qm.deepfake_probability == 0.1
    assert qm.musiq_score == 75.0


def test_dataset_stats_new_fields():
    stats = DatasetStats(
        total_samples=100,
        valid_samples=95,
        invalid_samples=5,
        total_size=1_000_000,
    )
    new_fields = [
        "fvd",
        "kvd",
        "fvmd",
        "fid",
        "precision",
        "recall",
        "coverage",
        "density",
        "diversity_score",
        "semantic_coverage",
        "outlier_count",
        "class_balance_score",
        "duplicate_pairs",
    ]
    for field in new_fields:
        assert hasattr(stats, field)


def test_video_metadata_fields():
    metadata = VideoMetadata(
        width=1920,
        height=1080,
        fps=30.0,
        duration=10.0,
        frame_count=300,
        codec="h264",
        bitrate=5000000,
        file_size=6250000,
    )
    assert metadata.width == 1920
    assert metadata.height == 1080
    assert metadata.fps == 30.0
    assert metadata.duration == 10.0
    assert metadata.frame_count == 300
    assert metadata.codec == "h264"
    assert metadata.bitrate == 5000000
    assert metadata.file_size == 6250000


def test_sample_metadata_path_resolution(tmp_dir):
    sample = Sample(path=tmp_dir / "test.mp4", is_video=True)
    assert sample.path == tmp_dir / "test.mp4"


def test_sample_image_loading(tmp_dir):
    import cv2

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    path = tmp_dir / "test.png"
    cv2.imwrite(str(path), img)
    sample = Sample(path=path, is_video=False)
    loaded = sample.load_image()
    assert loaded is not None
    assert loaded.shape == img.shape

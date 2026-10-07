from ayase.models import QualityMetrics


def test_video_native_fields():
    qm = QualityMetrics()
    fields = [
        "finevq_raw_mean",
        "kvq_score",
        "rqvqa_score",
        "svr60_score",
        "resnet_svr_score",
        "funque_score",
        "gabor_flow_score",
        "mscn_entropy_score",
        "c3dvqa_score",
        "flolpips",
        "hdr_subband_flicker_score",
        "stlpips_selfdist",
        "camera_jitter_score",
        "jump_cut_score",
        "playback_speed_score",
        "flow_coherence",
        "letterbox_ratio",
    ]
    for field in fields:
        assert hasattr(qm, field)

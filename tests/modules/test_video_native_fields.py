from ayase.models import QualityMetrics


def test_video_native_fields():
    qm = QualityMetrics()
    fields = [
        "finevq_score",
        "kvq_score",
        "rqvqa_score",
        "videval_score",
        "tlvqm_score",
        "funque_score",
        "movie_score",
        "st_greed_score",
        "c3dvqa_score",
        "flolpips",
        "hdr_vqm",
        "st_lpips",
        "camera_jitter_score",
        "jump_cut_score",
        "playback_speed_score",
        "flow_coherence",
        "letterbox_ratio",
    ]
    for field in fields:
        assert hasattr(qm, field)

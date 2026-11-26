def test_avbench_alignment_sample(video_sample, monkeypatch):
    from ayase.modules.avbench_alignment import AVBenchAlignmentModule

    module = AVBenchAlignmentModule({"test_mode": True})
    module.on_mount()
    assert module._model is None
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None

    module._model = object()
    monkeypatch.setattr(module, "_score", lambda path: None)
    assert module.process(video_sample).quality_metrics is None
    monkeypatch.setattr(module, "_score", lambda path: 0.75)
    assert module.process(video_sample).quality_metrics.avbench_alignment_score == 0.75
    module.on_dispose()
    assert module._model is None

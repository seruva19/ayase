def test_wmreward_sample(video_sample, monkeypatch):
    from ayase.modules.wmreward import WMRewardModule

    module = WMRewardModule({"test_mode": True})
    module.on_mount()
    assert module._encoder is None
    assert module.process(video_sample) is video_sample
    assert video_sample.quality_metrics is None

    module._encoder = object()
    monkeypatch.setattr(module, "_score", lambda path: None)
    assert module.process(video_sample).quality_metrics is None
    monkeypatch.setattr(module, "_score", lambda path: 0.42)
    assert module.process(video_sample).quality_metrics.latent_prediction_reward == 0.42
    module.on_dispose()
    assert module._encoder is None

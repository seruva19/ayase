"""Tests for videophy module."""

from ..conftest import _test_module_basics


def test_videophy_basics():
    from ayase.modules.vlm_phy import VideoPhyModule
    _test_module_basics(VideoPhyModule, "vlm_phy")


def test_videophy_image_is_noop(image_sample):
    from ayase.modules.vlm_phy import VideoPhyModule
    m = VideoPhyModule()
    out = m.process(image_sample)
    assert out is image_sample
    if out.quality_metrics is not None:
        assert out.quality_metrics.vlm_pc_likert is None


def test_videophy_parse_likert():
    from ayase.modules.vlm_phy import _parse_likert
    assert _parse_likert("3") == 0.5
    assert _parse_likert("The answer is 5.") == 1.0
    assert _parse_likert("score: 1, terrible") == 0.0
    # No fabricated neutral: unparseable responses yield None
    assert _parse_likert("no digit here") is None


def test_videophy_decodes_generated_answer_without_prompt_digits():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from ayase.modules.vlm_phy import PHYSICS_PROMPT, VideoPhyModule, _parse_likert

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def apply_chat_template(self, conversation, add_generation_prompt):
            return PHYSICS_PROMPT

        def __call__(self, **kwargs):
            return Inputs(input_ids=torch.tensor([[1, 20, 21]]))

        def batch_decode(self, tokens, skip_special_tokens):
            # A decoder that exposes the prompt's literal 1-5 if it leaks.
            assert tokens.tolist() == [[5]]
            return ["5"]

    module = VideoPhyModule()
    module._vlm_processor = Processor()
    module._vlm_model = SimpleNamespace(
        generate=lambda **kwargs: torch.tensor([[1, 20, 21, 5]])
    )
    answer = module._ask_vlm(np.zeros((8, 4, 4, 3)), PHYSICS_PROMPT)
    assert _parse_likert(answer) == 1.0


def test_videophy_no_backend_leaves_fields_unset(video_sample):
    """Without a loaded backend the module must not fabricate neutral scores."""
    from ayase.modules.vlm_phy import VideoPhyModule
    m = VideoPhyModule(config={"backend": "trajectory"})
    # Without setup there is no VLM and no trajectory model — the module
    # must not crash and must leave both fields unset (no neutral 0.5).
    out = m.process(video_sample)
    assert out is video_sample
    if out.quality_metrics is not None:
        assert out.quality_metrics.vlm_pc_likert is None
        assert out.quality_metrics.vlm_sa_likert is None


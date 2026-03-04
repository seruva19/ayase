def test_vlm_presets_structure():
    from ayase.modules.vlm_judge import VLM_PRESETS

    assert isinstance(VLM_PRESETS, dict)
    expected = {"shot_scale", "time_of_day", "clothing_style", "mood", "expression"}
    assert expected == set(VLM_PRESETS.keys())
    for name, labels in VLM_PRESETS.items():
        assert isinstance(labels, list)
        assert len(labels) >= 4
        # All labels should be non-empty strings
        for label in labels:
            assert isinstance(label, str) and len(label) > 0


def test_vlm_judge_presets_config():
    from ayase.modules.vlm_judge import VLMJudgeModule
    from .conftest import _test_module_basics

    _test_module_basics(VLMJudgeModule, "vlm_judge")
    m = VLMJudgeModule()
    assert "presets" in m.default_config
    assert "mode" in m.default_config


def test_vlm_match_label_exact():
    from ayase.modules.vlm_judge import VLMJudgeModule

    labels = ["close_up", "medium", "wide", "extreme_wide"]
    assert VLMJudgeModule._match_label("close_up", labels) == "close_up"
    assert VLMJudgeModule._match_label("medium", labels) == "medium"


def test_vlm_match_label_substring():
    from ayase.modules.vlm_judge import VLMJudgeModule

    labels = ["close_up", "medium", "wide", "extreme_wide"]
    assert VLMJudgeModule._match_label("I think it's a medium shot", labels) == "medium"
    assert VLMJudgeModule._match_label("this is wide angle", labels) == "wide"


def test_vlm_match_label_case_insensitive():
    from ayase.modules.vlm_judge import VLMJudgeModule

    labels = ["close_up", "medium", "wide", "extreme_wide"]
    assert VLMJudgeModule._match_label("WIDE", labels) == "wide"
    assert VLMJudgeModule._match_label("Close_Up", labels) == "close_up"


def test_vlm_match_label_fallback():
    from ayase.modules.vlm_judge import VLMJudgeModule

    labels = ["happy", "sad", "neutral"]
    # Completely unrelated response → falls back to last label
    result = VLMJudgeModule._match_label("xyzzy gibberish", labels)
    assert result in labels


def test_vlm_heuristic_presets(image_sample):
    from ayase.modules.vlm_judge import VLMJudgeModule

    m = VLMJudgeModule({"mode": "presets", "presets": ["time_of_day", "shot_scale"]})
    result = m.process(image_sample)

    preset_dets = [d for d in result.detections if d.get("type") == "vlm_preset"]
    assert len(preset_dets) == 2
    preset_names = {d["preset"] for d in preset_dets}
    assert preset_names == {"time_of_day", "shot_scale"}
    for d in preset_dets:
        assert "label" in d
        assert d["method"] == "heuristic"


def test_vlm_heuristic_all_presets(image_sample):
    from ayase.modules.vlm_judge import VLM_PRESETS, VLMJudgeModule

    m = VLMJudgeModule({"mode": "presets"})  # empty presets list → all presets
    result = m.process(image_sample)

    preset_dets = [d for d in result.detections if d.get("type") == "vlm_preset"]
    assert len(preset_dets) == len(VLM_PRESETS)
    for d in preset_dets:
        assert d["label"] in VLM_PRESETS[d["preset"]]


def test_vlm_heuristic_no_duplicate_detections(image_sample):
    from ayase.modules.vlm_judge import VLMJudgeModule

    m = VLMJudgeModule({"mode": "presets", "presets": ["mood", "expression"]})
    result = m.process(image_sample)

    preset_dets = [d for d in result.detections if d.get("type") == "vlm_preset"]
    # Exactly one detection per preset, no duplicates
    presets_seen = [d["preset"] for d in preset_dets]
    assert len(presets_seen) == len(set(presets_seen))

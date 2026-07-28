"""Focused tests for the DICE instruction-guided edit evaluator."""

import pytest
from PIL import Image

from ayase.models import CaptionMetadata, Sample
from tests.modules.conftest import _test_module_basics


def test_dice_edit_basics():
    from ayase.modules.dice_edit import DICEEditModule

    _test_module_basics(DICEEditModule, "dice_edit")


def test_parse_dice_changes():
    from ayase.modules.dice_edit import parse_dice_changes

    text = (
        'Assistant: ["EDIT: green vase changed to flowerpot, '
        'BOUNDING_BOX: [0.09, 0.35, 0.32, 0.63]", '
        '"ADD: yellow flower, [0.4, 0.2, 0.6, 0.7]"]'
    )
    assert parse_dice_changes(text) == [
        {
            "operation": "EDIT",
            "subject": "green vase changed to flowerpot",
            "bbox": [0.09, 0.35, 0.32, 0.63],
        },
        {
            "operation": "ADD",
            "subject": "yellow flower",
            "bbox": [0.4, 0.2, 0.6, 0.7],
        },
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning: requested change. Answer: YES", True),
        ("Decision: NO", False),
        ("YES is discussed but no final label exists", None),
    ],
)
def test_parse_dice_decision(text, expected):
    from ayase.modules.dice_edit import parse_dice_decision

    assert parse_dice_decision(text) is expected


def test_render_dice_change_marks_only_target_image():
    from ayase.modules.dice_edit import render_dice_change

    source = Image.new("RGB", (640, 480), "white")
    edited = Image.new("RGB", (640, 480), "white")
    marked_source, marked_edited = render_dice_change(
        source,
        edited,
        {"operation": "ADD", "subject": "object", "bbox": [0.1, 0.1, 0.5, 0.5]},
    )
    assert marked_source.size == (576, 576)
    assert marked_edited.size == (576, 576)
    assert marked_edited.getpixel((32 + 51, 32 + 51)) == (255, 0, 0)
    assert marked_source.getpixel((32 + 51, 32 + 51)) == (255, 255, 255)


def test_process_without_backend_is_graceful(image_sample):
    from ayase.modules.dice_edit import DICEEditModule

    sample = Sample(
        path=image_sample.path,
        is_video=False,
        reference_path=image_sample.path,
        caption=CaptionMetadata(text="change the vase", length=15),
    )
    module = DICEEditModule()
    assert module.process(sample) is sample
    assert sample.quality_metrics is None


def test_process_aggregates_object_level_decisions(monkeypatch, tmp_path):
    from ayase.modules.dice_edit import DICEEditModule

    source_path = tmp_path / "source.png"
    edited_path = tmp_path / "edited.png"
    Image.new("RGB", (64, 64), "white").save(source_path)
    Image.new("RGB", (64, 64), "black").save(edited_path)

    module = DICEEditModule()
    module._backend = "dice"
    module._difference_root = tmp_path
    module._coherence_root = tmp_path
    module._base_root = tmp_path
    outputs = iter(
        [
            (
                '["EDIT: white square to black, BOUNDING_BOX: '
                '[0.0, 0.0, 1.0, 1.0]"]'
            ),
            "Reasoning: requested change. Answer: YES",
        ]
    )
    monkeypatch.setattr(module, "_load_model", lambda *args: object())
    monkeypatch.setattr(module, "_generate", lambda *args, **kwargs: next(outputs))
    monkeypatch.setattr(module, "_release_model", lambda model: None)

    sample = Sample(
        path=edited_path,
        is_video=False,
        reference_path=source_path,
        caption=CaptionMetadata(text="make the square black", length=21),
    )
    result = module.process(sample)

    assert result is sample
    assert result.quality_metrics.dice_edit_coherence_score == 1.0
    assert result.detections[-1]["changes"][0]["coherent"] is True

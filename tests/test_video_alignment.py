import json

import pytest
from pydantic import ValidationError

from ayase.models import Sample
from ayase.pipeline import Pipeline
from ayase.video_alignment import VideoAlignment


def test_aligns_different_fps_and_durations():
    output = [12, 12.25, 12.5, 12.75, 13]
    source = [4, 4.125, 4.5]
    assert VideoAlignment().source_indices(output, source) == [0, 1, 2, None, None]
    assert VideoAlignment(mode="normalized").source_indices(output, source) == [0, 1, 1, 2, 2]
    assert VideoAlignment().source_indices(output, []) == [None] * 5


def test_edit_retiming_can_hold_frames_and_limit_coverage():
    policy = VideoAlignment(mode="anchors", anchors=[(0.5, 0), (1, 0.5), (1.5, 0.5)])
    assert policy.source_indices([0, 0.5, 0.75, 1, 1.5, 2], [0, 0.25, 0.5, 0.75]) == [
        None,
        0,
        1,
        2,
        2,
        None,
    ]
    with pytest.raises(ValueError, match="strictly increasing"):
        policy.source_indices([0, 0], [0, 1])
    with pytest.raises(ValidationError, match="must not decrease"):
        VideoAlignment(mode="anchors", anchors=[(0, 1), (1, 0)])


def test_edit_inputs_round_trip_and_invalidate_resume(tmp_path):
    media = tmp_path / "output.mp4"
    source = tmp_path / "source.mp4"
    media.write_bytes(b"edited")
    source.write_bytes(b"original")
    sample = Sample(
        path=media,
        is_video=True,
        source_video=source,
        edit_instruction="Change the sky",
        video_alignment=VideoAlignment(mode="normalized"),
    )
    original = Pipeline([])
    original.process_sample(sample)
    checkpoint = tmp_path / "state.json"
    original.save_state(checkpoint)

    resumed = Pipeline([])
    resumed.load_state(checkpoint)
    assert resumed.results[str(media)] == sample
    changed = sample.model_copy(update={"edit_instruction": "Change the lighting"})
    assert resumed.process_sample(changed) is changed
    altered_policy = changed.model_copy(update={"video_alignment": VideoAlignment()})
    assert resumed.process_sample(altered_policy) is altered_policy

    source.write_bytes(b"replaced source")
    resumed.load_state(checkpoint)
    assert not resumed.results


def test_ordinary_samples_keep_existing_json_shape():
    sample = Sample(path="output.mp4", is_video=True)
    document = json.loads(sample.model_dump_json())
    assert document == sample.canonical_model_dump(mode="json")
    assert all(
        key not in document for key in ("source_video", "edit_instruction", "video_alignment")
    )
    assert Sample.model_validate(document) == sample

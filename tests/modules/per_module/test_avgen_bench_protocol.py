"""Pure protocol conformance tests for the AVGen-Bench foundation."""

import hashlib
import json
import math

import pytest


def test_protocol_identity_and_dimension_contract():
    from ayase.third_party.avgen_bench import (
        DIMENSION_DEFINITIONS,
        DIMENSION_ORDER,
        GROUP_DIMENSIONS,
        GROUP_WEIGHTS,
        LAUNCHER_PROFILE,
        SAMPLE_SCORE_FIELDS,
        SOURCE_DIMENSION_KEYS,
        SOURCE_REVISION,
        SPEECH_BACKEND,
        SYNCNET_LAUNCHER_CONFIDENCE_THRESHOLD,
    )

    assert SOURCE_REVISION == "1049eabac472d479fe5feeb1ee202961f8e0982a"
    assert LAUNCHER_PROFILE == "run_full_evaluation.sh:full-suite"
    assert SPEECH_BACKEND == "faster-whisper"
    assert SYNCNET_LAUNCHER_CONFIDENCE_THRESHOLD == 1.0
    assert len(DIMENSION_ORDER) == len(DIMENSION_DEFINITIONS) == 11
    assert tuple(SOURCE_DIMENSION_KEYS.values()) == (
        "Vis",
        "Aud",
        "AV",
        "Lip",
        "Text",
        "Face",
        "Music",
        "Speech",
        "Lo-Phy",
        "Hi-Phy",
        "Holistic",
    )
    assert GROUP_DIMENSIONS["basic"] == ("visual", "audio")
    assert GROUP_DIMENSIONS["cross"] == ("av_sync", "lip_sync")
    assert len(GROUP_DIMENSIONS["fine"]) == 7
    assert GROUP_WEIGHTS == {"basic": 0.2, "cross": 0.2, "fine": 0.6}
    assert SAMPLE_SCORE_FIELDS["high_physics"] == "avgen_bench_high_physics_score"
    assert DIMENSION_DEFINITIONS["lip_sync"].reducer == "pooled_passing_track_mean"


def test_normalization_matches_all_55_independent_oracle_boundaries():
    from ayase.third_party.avgen_bench import normalize_raw_mean

    upper = {
        "visual": 1.0,
        "audio": 10.0,
        "av_sync": 0.5,
        "lip_sync": 8.0,
        "text": 100.0,
        "face": 100.0,
        "music": 100.0,
        "speech": 100.0,
        "low_physics": 5.0,
        "high_physics": 100.0,
        "holistic": 100.0,
    }
    higher_better = {
        "visual",
        "audio",
        "text",
        "face",
        "music",
        "speech",
        "low_physics",
        "high_physics",
        "holistic",
    }
    for dimension, end in upper.items():
        raw_values = (-end, 0.0, end / 2.0, end, 2.0 * end)
        expected = (0.0, 0.0, 50.0, 100.0, 100.0)
        if dimension not in higher_better:
            expected = (100.0, 100.0, 50.0, 0.0, 0.0)
        assert [normalize_raw_mean(dimension, raw) for raw in raw_values] == list(expected)


def test_all_2048_presence_masks_match_independent_source_oracle_digest():
    from ayase.third_party.avgen_bench import DIMENSION_ORDER, aggregate_raw_means

    fixture = {
        "visual": 0.37,
        "audio": 6.3,
        "av_sync": 0.13,
        "lip_sync": 3.0,
        "text": 91.0,
        "face": 47.0,
        "music": 62.0,
        "speech": 81.0,
        "low_physics": 3.0,
        "high_physics": 74.0,
        "holistic": 55.0,
    }
    rows = []
    for mask in range(1 << len(DIMENSION_ORDER)):
        selected = {
            key: fixture[key] for index, key in enumerate(DIMENSION_ORDER) if mask & (1 << index)
        }
        result = aggregate_raw_means(selected, {key: 1 for key in selected})
        rows.append([mask, result["score"], result["coverage"], result["group_scores"]])

    digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    assert digest == "5a33377af79da18371fc3888c8fd8110a5a3816cbd940abb03952c967c478059"


def test_full_fixture_group_scores_weights_and_missing_order():
    from ayase.third_party.avgen_bench import aggregate_raw_means

    raw = {
        "visual": 0.37,
        "audio": 6.3,
        "av_sync": 0.13,
        "lip_sync": 3.0,
        "text": 91.0,
        "face": 47.0,
        "music": 62.0,
        "speech": 81.0,
        "low_physics": 3.0,
        "high_physics": 74.0,
        "holistic": 55.0,
    }
    result = aggregate_raw_means(raw, {key: 1 for key in raw})

    assert result["score"] == pytest.approx(63.93571428571428)
    assert result["coverage"] == 1.0
    assert result["group_scores"] == pytest.approx(
        {"basic": 50.0, "cross": 68.25, "fine": 67.14285714285714}
    )
    assert result["missing_dimensions"] == []

    empty = aggregate_raw_means({}, {})
    assert empty["score"] is None
    assert empty["coverage"] == 0.0
    assert empty["group_scores"] == {"basic": None, "cross": None, "fine": None}
    assert empty["missing_source_dimensions"] == [
        "Vis",
        "Aud",
        "AV",
        "Lip",
        "Text",
        "Face",
        "Music",
        "Speech",
        "Lo-Phy",
        "Hi-Phy",
        "Holistic",
    ]


@pytest.mark.parametrize(
    ("dimension", "observations", "expected_raw_mean"),
    [
        ("visual", [-1.0, 1.0], 0.0),
        ("av_sync", [0.0, 1.0], 0.5),
        ("lip_sync", [0.0, 16.0], 8.0),
        ("low_physics", [-5.0, 5.0], 0.0),
    ],
)
def test_raw_dataset_mean_is_normalized_after_pooling(dimension, observations, expected_raw_mean):
    from ayase.third_party.avgen_bench import aggregate_observations

    result = aggregate_observations({dimension: observations})
    assert result["score"] == 0.0
    assert result["dimensions"][dimension]["raw_mean"] == expected_raw_mean


def test_lip_sync_pools_absolute_passing_tracks_not_per_video_means():
    from ayase.third_party.avgen_bench import aggregate_observations

    result = aggregate_observations({"lip_sync": [0, 8, 8, 8]})
    lip = result["dimensions"]["lip_sync"]
    assert lip["evidence_count"] == 4
    assert lip["raw_mean"] == 6.0
    assert lip["normalized_score"] == 25.0
    assert result["score"] == 25.0


def test_strict_observations_exclude_bool_strings_nonfinite_and_negative_offsets():
    from ayase.third_party.avgen_bench import aggregate_observations, is_strict_finite_number

    assert is_strict_finite_number(0)
    assert is_strict_finite_number(0.5)
    assert not is_strict_finite_number(True)
    assert not is_strict_finite_number("0.5")
    assert not is_strict_finite_number(float("nan"))
    assert not is_strict_finite_number(float("inf"))

    result = aggregate_observations(
        {
            "visual": [0.5, True, "0.25", float("nan"), float("inf")],
            "av_sync": [-0.1],
            "lip_sync": [-1],
        }
    )
    assert result["dimensions"]["visual"]["evidence_count"] == 1
    assert result["dimensions"]["visual"]["invalid_observation_count"] == 4
    assert result["dimensions"]["visual"]["normalized_score"] == 50.0
    assert result["dimensions"]["av_sync"]["invalid_observation_count"] == 1
    assert result["dimensions"]["lip_sync"]["invalid_observation_count"] == 1


def test_zero_evidence_never_fabricates_music_or_ocr_metric():
    from ayase.third_party.avgen_bench import aggregate_observations, aggregate_raw_means

    diagnostics = {
        "Music": {"mean_score": 0.0, "num_midi_evaluable": 0},
        "Text": {"fallback_raw_csv_mean": 100.0, "count_eligible_for_summary": 0},
    }
    result = aggregate_observations({"music": [], "text": []}, source_raw_summaries=diagnostics)
    assert result["score"] is None
    assert result["coverage"] == 0.0
    assert result["dimensions"]["music"]["evidence_count"] == 0
    assert result["dimensions"]["text"]["evidence_count"] == 0
    assert result["raw_source_summaries"] == diagnostics

    with pytest.raises(ValueError, match="zero observations"):
        aggregate_raw_means({"music": 0.0}, {"music": 0})


def test_raw_mean_validation_rejects_ambiguous_or_invalid_input():
    from ayase.third_party.avgen_bench import aggregate_raw_means, normalize_raw_mean

    with pytest.raises(ValueError, match="finite int or float"):
        normalize_raw_mean("visual", True)
    with pytest.raises(ValueError, match="valid finite raw mean"):
        aggregate_raw_means({"visual": math.nan}, {"visual": 1})
    with pytest.raises(ValueError, match="absolute non-negative"):
        aggregate_raw_means({"lip_sync": -1.0}, {"lip_sync": 1})
    with pytest.raises(KeyError, match="unknown"):
        aggregate_raw_means({"Vis": 0.5}, {"Vis": 1})


def test_asset_bundle_digests_and_canonical_universe():
    from ayase.third_party.avgen_bench import (
        asset_manifest,
        canonical_data_summary,
        canonical_face_annotations,
        canonical_prompts,
        verify_asset_bundle,
    )

    verification = verify_asset_bundle()
    assert verification == {"valid": True, "checked": 23, "failures": []}
    manifest = asset_manifest()
    assert manifest["generated_from"] == "immutable Git blobs"
    assert len(manifest["files"]) == 23
    with pytest.raises(TypeError):
        manifest["schema"] = 2

    prompts = canonical_prompts()
    annotations = canonical_face_annotations()
    summary = canonical_data_summary()
    assert len(prompts) == summary["total_prompts"] == 235
    assert len(annotations) == summary["total_face_annotations"] == 220
    assert summary["face_max_ids_positive"] == 103
    assert len(summary["missing_face_annotations"]) == 15
    assert {item["category"] for item in summary["missing_face_annotations"]} == {
        "musical_instrument_tutorial"
    }
    assert summary["categories"]["musical_instrument_tutorial"] == {
        "prompts": 35,
        "face_annotations": 20,
        "face_max_ids_positive": 1,
    }


def test_lookup_is_exact_and_preserves_original_unicode_identity_and_prompt():
    from ayase.third_party.avgen_bench import (
        lookup_face_annotation,
        lookup_prompt,
        missing_face_annotation_identities,
    )

    title = "Piano: I–V–vi–IV in C (C–G–Am–F) Block Chords"
    prompt = lookup_prompt("musical_instrument_tutorial", title)
    assert prompt.content == title
    assert "I–V–vi–IV" in prompt.content
    assert prompt.prompt
    assert (prompt.category, prompt.content) in missing_face_annotation_identities()
    with pytest.raises(KeyError, match="face-annotation identity"):
        lookup_face_annotation(prompt.category, prompt.content)
    with pytest.raises(KeyError, match="prompt identity"):
        lookup_prompt("cooking", title)

    labeled = lookup_prompt(
        "ads", "Leave With A Cannonballer | Live There With Your Family | Vacation Rental Platform"
    )
    face = lookup_face_annotation(labeled.category, labeled.content)
    assert face.prompt == labeled.prompt
    assert face.min_ids == 2
    assert face.max_ids == 6
    with pytest.raises(TypeError):
        face.annotation["primary_faces"]["max_ids"] = 99


def test_safe_content_token_is_hash_based_windows_safe_and_identity_specific():
    from ayase.third_party.avgen_bench import safe_content_token

    content = "Acoustic Guitar: C Major Triad Arpeggio (C–E–G)"
    token = safe_content_token("musical_instrument_tutorial", content)
    assert token == safe_content_token("musical_instrument_tutorial", content)
    assert token != safe_content_token("sports", content)
    assert token.startswith("avgen-")
    assert len(token) == 70
    assert not set('<>:"/\\|?*').intersection(token)

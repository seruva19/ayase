"""Backward-compatibility contracts for the 0.1.80 provenance cleanup.

Renamed modules/fields kept resolving through deprecated aliases, removed
stub fields read as ``None`` instead of raising, and explicitly selected
modules count as the provenance opt-in so consumers that name a module
directly are not silently gated.
"""

import warnings
from collections import UserDict
from types import MappingProxyType

import pytest

from ayase.cli import _instantiate_modules, _parse_pipeline_str
from ayase.config import AyaseConfig
from ayase.models import DatasetStats, QualityMetrics
from ayase.pipeline import (
    MODULE_ALIASES,
    AyasePipeline,
    ModuleRegistry,
    Pipeline,
)
from ayase.profile import instantiate_profile_modules
from ayase.runtime import opt_in_all_provenance


@pytest.fixture(scope="module")
def discovered():
    ModuleRegistry.discover_modules()
    yield


class TestModuleAliases:
    def test_all_aliases_resolve(self, discovered):
        for old, new in MODULE_ALIASES.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                cls = ModuleRegistry.get_module(old)
            assert cls is not None, f"alias {old} -> {new} did not resolve"
            assert cls.name == new, f"alias {old} resolved to {cls.name}, want {new}"

    def test_alias_warns(self, discovered):
        with pytest.warns(DeprecationWarning, match="renamed"):
            ModuleRegistry.get_module("ttsds2")

    def test_canonical_names_do_not_warn(self, discovered):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ModuleRegistry.get_module("metadata")
        assert not [w for w in caught if w.category is DeprecationWarning]

    def test_aliases_not_listed(self, discovered):
        listed = ModuleRegistry.list_modules()
        for old in MODULE_ALIASES:
            assert old not in listed

    def test_no_alias_shadows_live_module(self, discovered):
        listed = ModuleRegistry.list_modules()
        for old in MODULE_ALIASES:
            assert old not in listed


class TestQualityMetricsAliases:
    def test_alias_targets_exist(self):
        missing = [
            n
            for n in QualityMetrics._DEPRECATED_FIELD_ALIASES.values()
            if n not in QualityMetrics.model_fields
        ]
        assert missing == []

    def test_read_through_alias(self):
        qm = QualityMetrics(aesthetic_v25_dup=0.7)
        with pytest.warns(DeprecationWarning, match="renamed"):
            assert qm.vqa_a_score == 0.7

    def test_construct_with_legacy_key(self):
        qm = QualityMetrics(vqa_a_score=0.5, ttsds2_score=0.9)
        assert qm.aesthetic_v25_dup == 0.5
        assert qm.tts_system_dist_score == 0.9

    def test_legacy_key_does_not_override_new(self):
        qm = QualityMetrics(vqa_a_score=0.5, aesthetic_v25_dup=0.8)
        assert qm.aesthetic_v25_dup == 0.8

    def test_removed_field_reads_none(self):
        qm = QualityMetrics()
        with pytest.warns(DeprecationWarning, match="non-functional stub|no published definition"):
            assert qm.qclip_score is None

    def test_removed_field_dropped_on_construct(self):
        qm = QualityMetrics(confidence_score=0.9)
        assert "confidence_score" not in qm.model_dump()

    def test_unknown_field_still_forbidden(self):
        with pytest.raises(Exception):
            QualityMetrics(bogus_field=1.0)

    def test_unknown_attr_still_raises(self):
        with pytest.raises(AttributeError):
            QualityMetrics().bogus_field

    def test_field_never_on_model_still_raises(self):
        # ``fgd`` was a DatasetStats field, never a QualityMetrics one.
        with pytest.raises(AttributeError):
            QualityMetrics().fgd
        with pytest.raises(Exception):
            QualityMetrics(fgd=0.5)

    def test_dump_has_only_new_names(self):
        qm = QualityMetrics(vqa_a_score=0.5)
        keys = set(qm.model_dump())
        assert "aesthetic_v25_dup" in keys
        assert "vqa_a_score" not in keys

    def test_legacy_input_is_not_mutated(self):
        payload = {"vqa_a_score": 0.5}
        QualityMetrics.model_validate(payload)
        assert payload == {"vqa_a_score": 0.5}

    @pytest.mark.parametrize("mapping", [UserDict, MappingProxyType])
    def test_legacy_mapping_inputs(self, mapping):
        qm = QualityMetrics.model_validate(mapping({"vqa_a_score": 0.5}))
        assert qm.aesthetic_v25_dup == 0.5

    def test_legacy_provenance_keys_are_migrated(self):
        qm = QualityMetrics.model_validate(
            {
                "vqa_a_score": 0.5,
                "metric_provenance": {"vqa_a_score": "adapted"},
            }
        )
        assert qm.metric_provenance == {"aesthetic_v25_dup": "adapted"}

    def test_read_only_legacy_provenance_mapping_is_migrated(self):
        qm = QualityMetrics.model_validate(
            {
                "vqa_a_score": 0.5,
                "metric_provenance": MappingProxyType(
                    {"vqa_a_score": "adapted"}
                ),
            }
        )
        assert qm.metric_provenance == {"aesthetic_v25_dup": "adapted"}

    def test_non_null_counts_new_name_once(self):
        qm = QualityMetrics(vqa_a_score=0.5)
        assert qm.non_null_metrics()["aesthetic_v25_dup"] == 0.5


class TestDatasetStatsAliases:
    def test_read_through_alias(self):
        stats = DatasetStats(
            total_samples=1, valid_samples=1, invalid_samples=0, total_size=0,
            mmd_selfsplit=0.4,
        )
        with pytest.warns(DeprecationWarning, match="renamed"):
            assert stats.jedi == 0.4

    def test_construct_with_legacy_key(self):
        stats = DatasetStats(
            total_samples=1, valid_samples=1, invalid_samples=0, total_size=0,
            verse_bench_overall=0.8,
        )
        assert stats.verse_bench_overall_est == 0.8

    def test_removed_field_reads_none(self):
        stats = DatasetStats(
            total_samples=1, valid_samples=1, invalid_samples=0, total_size=0
        )
        with pytest.warns(DeprecationWarning, match="non-functional stub|no published definition"):
            assert stats.fmd is None


class TestExplicitSelectionOptIn:
    """Naming a module directly is itself the provenance opt-in."""

    def _cfg(self) -> AyaseConfig:
        return AyaseConfig(general={"device": "cpu"})

    def test_instantiate_modules_opted_in(self, discovered):
        modules = _instantiate_modules(
            ["expression_similarity"], self._cfg(), provenance_opt_in=True
        )
        assert "own" in modules[0].config["allow_provenance"]
        assert "adapted" in modules[0].config["allow_provenance"]
        pipe = Pipeline(modules)
        assert "expression_similarity" not in pipe._provenance_excluded

    def test_instantiate_modules_default_gated(self, discovered):
        modules = _instantiate_modules(["expression_similarity"], self._cfg())
        pipe = Pipeline(modules)
        assert "expression_similarity" in pipe._provenance_excluded

    def test_pipeline_string_opted_in(self, discovered):
        modules = _parse_pipeline_str("expression_similarity", self._cfg())
        assert "own" in modules[0].config["allow_provenance"]

    def test_profile_opted_in(self, discovered):
        modules = instantiate_profile_modules(
            {"name": "t", "modules": ["expression_similarity"]}, self._cfg()
        )
        assert "own" in modules[0].config["allow_provenance"]
        pipe = Pipeline(modules)
        assert "expression_similarity" not in pipe._provenance_excluded

    def test_ayase_pipeline_modules_opted_in(self, discovered):
        ap = AyasePipeline(config=self._cfg(), modules=["expression_similarity"])
        assert "expression_similarity" not in ap.pipeline._provenance_excluded

    def test_ayase_pipeline_config_modules_opted_in(self, discovered):
        cfg = self._cfg()
        cfg.pipeline.modules = ["expression_similarity"]
        ap = AyasePipeline(config=cfg)
        assert "expression_similarity" not in ap.pipeline._provenance_excluded

    def test_bare_module_still_gated(self):
        """A module instantiated with no provenance opt-in stays excluded."""
        ModuleRegistry.discover_modules()
        cls = ModuleRegistry.get_module("expression_similarity")
        pipe = Pipeline([cls(config={})])
        assert "expression_similarity" in pipe._provenance_excluded

    def test_opt_in_merges_existing(self):
        params = opt_in_all_provenance({"allow_provenance": ["adapted"]})
        assert sorted(params["allow_provenance"]) == ["adapted", "own"]

    def test_opt_in_all_provenance_preserves_other_keys(self):
        params = opt_in_all_provenance({"models_dir": "m", "device": "cpu"})
        assert params["models_dir"] == "m"
        assert params["device"] == "cpu"

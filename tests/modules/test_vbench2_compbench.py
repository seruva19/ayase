"""Tests for VBench-2.0 upgraded modules, ChronoMagic-Bench, and T2V-CompBench."""

import numpy as np
import pytest

from ayase.models import CaptionMetadata, QualityMetrics, Sample

from .conftest import _test_module_basics


# ===================================================================== #
# Module basics                                                         #
# ===================================================================== #


class TestPhysicsModule:
    def test_basics(self):
        from ayase.modules.physics import PhysicsModule
        _test_module_basics(PhysicsModule, "physics")

    def test_image_skipped(self, image_sample):
        from ayase.modules.physics import PhysicsModule
        mod = PhysicsModule()
        result = mod.process(image_sample)
        assert result.quality_metrics is None or result.quality_metrics.physics_score is None

    def test_score_from_tracks(self):
        from ayase.modules.physics import PhysicsModule
        mod = PhysicsModule()
        # Smooth trajectory — high score
        t = np.linspace(0, 10, 20)
        tracks = np.stack([
            np.column_stack([t, np.sin(t)]),
        ] * 5, axis=1)  # [20, 5, 2] — 5 points tracked over 20 frames
        score = mod._score_from_tracks(tracks)
        assert 0.0 <= score <= 1.0
        assert score > 0.3  # smooth motion should score well


class TestHumanFidelityModule:
    def test_basics(self):
        from ayase.modules.human_fidelity import HumanFidelityModule
        _test_module_basics(HumanFidelityModule, "human_fidelity")



class TestCommonsenseModule:
    def test_basics(self):
        from ayase.modules.commonsense import CommonsenseModule
        _test_module_basics(CommonsenseModule, "commonsense")



class TestDynamicsControllabilityModule:
    def test_basics(self):
        from ayase.modules.dynamics_controllability import DynamicsControllabilityModule
        _test_module_basics(DynamicsControllabilityModule, "dynamics_controllability")

    def test_image_skipped(self, image_sample):
        from ayase.modules.dynamics_controllability import DynamicsControllabilityModule
        mod = DynamicsControllabilityModule()
        result = mod.process(image_sample)
        assert result.quality_metrics is None or result.quality_metrics.dynamics_controllability is None

    def test_no_caption_skipped(self, video_sample):
        from ayase.modules.dynamics_controllability import DynamicsControllabilityModule
        mod = DynamicsControllabilityModule()
        result = mod.process(video_sample)
        assert result.quality_metrics is None or result.quality_metrics.dynamics_controllability is None

    def test_keyword_extraction(self):
        from ayase.modules.dynamics_controllability import DynamicsControllabilityModule
        mod = DynamicsControllabilityModule()
        assert mod._extract_expected_motion("a person running fast") > 0.5
        assert mod._extract_expected_motion("a calm still lake") < 0.3
        assert mod._extract_expected_motion("a beautiful sunset") == 0.5  # no keywords

    def test_camera_keyword_extraction(self):
        from ayase.modules.dynamics_controllability import DynamicsControllabilityModule
        mod = DynamicsControllabilityModule()
        result = mod._extract_camera_keywords("camera pan left and zoom in slowly")
        assert "pan_left" in result
        assert "zoom_in" in result

    def test_video_with_caption(self, video_sample):
        from ayase.modules.dynamics_controllability import DynamicsControllabilityModule
        mod = DynamicsControllabilityModule()
        video_sample.caption = CaptionMetadata(text="a ball moving slowly in a circle", length=31)
        result = mod.process(video_sample)
        qm = result.quality_metrics
        assert qm is not None
        assert qm.dynamics_controllability is not None
        assert 0.0 <= qm.dynamics_controllability <= 1.0


class TestCreativityModule:
    def test_basics(self):
        from ayase.modules.creativity import CreativityModule
        _test_module_basics(CreativityModule, "creativity")



class TestChronoMagicModule:
    def test_basics(self):
        from ayase.modules.chronomagic import ChronoMagicModule
        _test_module_basics(ChronoMagicModule, "chronomagic")

    def test_image_skipped(self, image_sample):
        from ayase.modules.chronomagic import ChronoMagicModule
        mod = ChronoMagicModule()
        result = mod.process(image_sample)
        assert result.quality_metrics is None or result.quality_metrics.chronomagic_mt_score is None

    def test_heuristic_smooth_video(self):
        from ayase.modules.chronomagic import ChronoMagicModule
        mod = ChronoMagicModule()
        # Create smooth gradient frames
        frames = []
        for i in range(16):
            frame = np.ones((64, 64, 3), dtype=np.uint8) * min(255, i * 16)
            frames.append(frame)
        grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]

        diffs = []
        for i in range(len(grays) - 1):
            diff = np.abs(grays[i + 1] - grays[i]).mean()
            diffs.append(diff)
        diffs_arr = np.array(diffs)

        # Smooth progression should have low cv
        cv = diffs_arr.std() / (diffs_arr.mean() + 1e-6)
        smoothness = 1.0 / (1.0 + cv)
        assert smoothness > 0.5


class TestT2VCompBenchModule:
    def test_basics(self):
        from ayase.modules.t2v_compbench import T2VCompBenchModule
        _test_module_basics(T2VCompBenchModule, "t2v_compbench")

    def test_image_skipped(self, image_sample):
        from ayase.modules.t2v_compbench import T2VCompBenchModule
        mod = T2VCompBenchModule()
        result = mod.process(image_sample)
        assert result.quality_metrics is None or result.quality_metrics.compbench_overall is None

    def test_no_caption_skipped(self, video_sample):
        from ayase.modules.t2v_compbench import T2VCompBenchModule
        mod = T2VCompBenchModule()
        result = mod.process(video_sample)
        assert result.quality_metrics is None or result.quality_metrics.compbench_overall is None

    def test_parse_attributes(self):
        from ayase.modules.t2v_compbench import T2VCompBenchModule
        attrs = T2VCompBenchModule._parse_attributes("a red ball and a blue car")
        adj_list = [a[0] for a in attrs]
        assert "red" in adj_list
        assert "blue" in adj_list

    def test_parse_spatial(self):
        from ayase.modules.t2v_compbench import T2VCompBenchModule
        spatials = T2VCompBenchModule._parse_spatial("a cat above a table")
        assert len(spatials) > 0
        assert spatials[0][1] == "above"

    def test_parse_count(self):
        from ayase.modules.t2v_compbench import T2VCompBenchModule
        counts = T2VCompBenchModule._parse_count("three dogs and two cats")
        nums = [c[0] for c in counts]
        assert 3 in nums
        assert 2 in nums

    def test_parse_actions(self):
        from ayase.modules.t2v_compbench import T2VCompBenchModule
        actions = T2VCompBenchModule._parse_actions("a person running in a park")
        assert len(actions) > 0
        verbs = [a[1] for a in actions]
        assert "running" in verbs

    def test_parse_relations(self):
        from ayase.modules.t2v_compbench import T2VCompBenchModule
        rels = T2VCompBenchModule._parse_relations("a person holding a ball")
        assert len(rels) > 0
        rel_types = [r[1] for r in rels]
        assert "holding" in rel_types


# ===================================================================== #
# QualityMetrics field existence                                        #
# ===================================================================== #


class TestNewFields:
    def test_vbench2_fields_exist(self):
        qm = QualityMetrics()
        assert hasattr(qm, "human_fidelity_score")
        assert hasattr(qm, "physics_score")
        assert hasattr(qm, "commonsense_score")
        assert hasattr(qm, "creativity_score")

    def test_chronomagic_fields_exist(self):
        qm = QualityMetrics()
        assert hasattr(qm, "chronomagic_mt_score")
        assert hasattr(qm, "chronomagic_ch_score")

    def test_compbench_fields_exist(self):
        qm = QualityMetrics()
        for field in [
            "compbench_attribute", "compbench_object_rel", "compbench_action",
            "compbench_spatial", "compbench_numeracy", "compbench_scene",
            "compbench_overall",
        ]:
            assert hasattr(qm, field), f"Missing field: {field}"

    def test_field_groups(self):
        qm = QualityMetrics()
        # Set all new fields so they show up in to_grouped_dict
        qm.human_fidelity_score = 0.5
        qm.physics_score = 0.5
        qm.commonsense_score = 0.5
        qm.creativity_score = 0.5
        qm.chronomagic_mt_score = 0.5
        qm.chronomagic_ch_score = 0.5
        qm.compbench_attribute = 0.5
        qm.compbench_overall = 0.5
        grouped = qm.to_grouped_dict()
        assert "human_fidelity_score" in grouped.get("scene", {})
        assert "physics_score" in grouped.get("motion", {})
        assert "commonsense_score" in grouped.get("scene", {})
        assert "creativity_score" in grouped.get("aesthetic", {})
        assert "chronomagic_mt_score" in grouped.get("temporal", {})
        assert "chronomagic_ch_score" in grouped.get("temporal", {})
        assert "compbench_attribute" in grouped.get("alignment", {})
        assert "compbench_overall" in grouped.get("alignment", {})

    def test_all_new_fields_default_none(self):
        qm = QualityMetrics()
        new_fields = [
            "human_fidelity_score", "physics_score", "commonsense_score",
            "creativity_score", "chronomagic_mt_score", "chronomagic_ch_score",
            "compbench_attribute", "compbench_object_rel", "compbench_action",
            "compbench_spatial", "compbench_numeracy", "compbench_scene",
            "compbench_overall",
        ]
        for field in new_fields:
            assert getattr(qm, field) is None, f"{field} should default to None"


# Need cv2 import for the chronomagic heuristic test
import cv2

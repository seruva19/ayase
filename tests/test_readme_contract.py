"""Tests that verify every public API claim made in README.md.

Each test class maps to a README section. If a test here breaks, either the
code regressed or the README needs updating.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from ayase.config import (
    AyaseConfig,
    FilterConfig,
    GeneralConfig,
    OutputConfig,
    PipelineConfig,
    QualityConfig)
from ayase.models import (
    CaptionMetadata,
    QualityMetrics,
    Sample,
    ValidationIssue,
    ValidationSeverity)
from ayase.pipeline import AyasePipeline, ModuleRegistry, Pipeline, PipelineModule
from ayase.profile import PipelineProfile, instantiate_profile_modules, load_profile
from ayase.scanner import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, scan_dataset


# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def synthetic_video(tmp_dir):
    path = tmp_dir / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (64, 64))
    for i in range(12):
        frame = np.full((64, 64, 3), (i * 20) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def synthetic_image(tmp_dir):
    path = tmp_dir / "test_image.png"
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 1] = 128
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def dataset_dir(tmp_dir, synthetic_video, synthetic_image):
    """A minimal dataset directory with one video, one image, and a caption."""
    vid = tmp_dir / "dataset" / "clip.mp4"
    img = tmp_dir / "dataset" / "photo.png"
    vid.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(vid), fourcc, 24.0, (64, 64))
    for i in range(8):
        writer.write(np.full((64, 64, 3), 100, dtype=np.uint8))
    writer.release()

    cv2.imwrite(str(img), np.zeros((64, 64, 3), dtype=np.uint8))

    caption = tmp_dir / "dataset" / "clip.txt"
    caption.write_text("a short clip", encoding="utf-8")
    return tmp_dir / "dataset"


# ── Stub module for testing ──────────────────────────────────────


class _StubModule(PipelineModule):
    name = "readme_stub"
    description = "Stub for README tests"

    def process(self, sample: Sample) -> Sample:
        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.technical_score = 75.0
        sample.quality_metrics.aesthetic_score = 6.5
        return sample


# =====================================================================
# README §Overview – "225 quality metrics"
# =====================================================================


class TestMetricCount:
    """Verify the metric count claimed in the overview."""

    def test_quality_metrics_has_251_fields(self):
        """README: '251 quality metrics across visual, temporal, audio, …'"""
        qm = QualityMetrics()
        # Pydantic model_fields gives declared fields (excludes _FIELD_GROUPS etc.)
        field_count = len(QualityMetrics.model_fields)
        assert field_count == 251, f"Expected 251, got {field_count}"

    def test_all_metric_fields_default_to_none(self):
        qm = QualityMetrics()
        for name in QualityMetrics.model_fields:
            assert getattr(qm, name) is None, f"{name} should default to None"


class TestQualityMetricsValidation:
    """Verify extra='forbid' catches typos and declared fields are accepted."""

    def test_quality_metrics_forbids_extra_fields(self):
        """QualityMetrics rejects undeclared field names (catches typos)."""
        with pytest.raises(ValidationError):
            QualityMetrics(blur_scroe=0.5)  # typo

    def test_quality_metrics_allows_declared_fields(self):
        """QualityMetrics accepts any known field."""
        qm = QualityMetrics(blur_score=0.5, psnr=42.0, aesthetic_score=7.5)
        assert qm.blur_score == 0.5
        assert qm.psnr == 42.0


# =====================================================================
# README §Metrics table – every row corresponds to a real field
# =====================================================================


# The 231 metric names from the README table, in order.
README_METRICS = [
    "blur_score", "compression_score", "aesthetic_score", "clip_score",
    "brightness", "contrast", "saturation", "fast_vqa_score",
    "motion_score", "camera_motion_score", "temporal_consistency",
    "technical_score", "noise_score", "artifacts_score",
    "watermark_probability", "ocr_area_ratio", "face_count", "nsfw_score",
    "audio_quality_score", "perceptual_hash", "depth_score", "auto_caption",
    "vqa_a_score", "vqa_t_score", "is_score", "sd_score",
    "gradient_detail", "blip_bleu", "detection_score", "count_score",
    "color_score", "celebrity_id_score", "identity_loss",
    "face_recognition_score", "ocr_score", "ocr_fidelity", "ocr_cer", "ocr_wer",
    "i2v_clip", "i2v_dino", "i2v_lpips", "i2v_quality",
    "action_score", "action_confidence", "flow_score", "motion_ac_score",
    "warping_error", "clip_temp", "face_consistency", "psnr", "ssim",
    "lpips", "spectral_entropy", "spectral_rank", "fvd", "kvd", "fvmd",
    "vmaf", "ms_ssim", "vif", "niqe", "t2v_score", "t2v_alignment",
    "t2v_quality", "dynamics_range", "dynamics_controllability",
    "scene_complexity", "compression_artifacts", "naturalness_score",
    "video_memorability", "usability_rate", "confidence_score",
    "human_preference_score", "engagement_score", "usability_score",
    "hdr_quality", "sdr_quality", "temporal_information",
    "spatial_information", "flicker_score", "judder_score",
    "stutter_score", "dists", "fsim", "gmsd", "vsi_score", "brisque",
    "pesq_score", "estoi_score", "mcd_score", "si_sdr_score",
    "lpdist_score", "utmos_score",
    "av_sync_offset", "dover_score", "dover_technical",
    "dover_aesthetic", "topiq_score", "liqe_score", "clip_iqa_score",
    "color_grading_score", "white_balance_score", "exposure_consistency",
    "focus_quality", "banding_severity", "qalign_quality",
    "qalign_aesthetic", "face_quality_score", "face_identity_consistency",
    "face_expression_smoothness", "face_landmark_jitter",
    "object_permanence_score", "semantic_consistency",
    "depth_temporal_consistency", "subject_consistency",
    "background_consistency", "motion_smoothness", "codec_efficiency",
    "gop_quality", "codec_artifacts", "deepfake_probability",
    "ai_generated_probability", "harmful_content_score",
    "watermark_strength", "bias_score", "depth_quality",
    "multiview_consistency", "stereo_comfort_score", "musiq_score",
    "contrique_score", "mdtvsfa_score", "nima_score", "dbcnn_score",
    "wadiqam_score", "maniqa_score", "arniqa_score", "qualiclip_score",
    "pieapp", "cw_ssim", "nlpd", "mad", "ahiq", "topiq_fr",
    "dreamsim", "cover_score", "cover_technical", "cover_aesthetic",
    "cover_semantic", "vqa_score_alignment", "videoscore_visual",
    "videoscore_temporal", "videoscore_dynamic", "videoscore_alignment",
    "videoscore_factual", "face_iqa_score", "scene_stability",
    "avg_scene_duration", "raft_motion_score", "ram_tags",
    "depth_anything_score", "depth_anything_consistency", "video_type",
    "video_type_confidence", "jedi", "trajan_score", "promptiqa_score",
    "aigv_static", "aigv_temporal", "aigv_dynamic", "aigv_alignment",
    "video_reward_score", "tifa_score", "text_overlay_score", "ptlflow_motion_score",
    "qcn_score", "finevq_score", "kvq_score", "rqvqa_score",
    "videval_score", "tlvqm_score", "funque_score", "movie_score",
    "st_greed_score", "c3dvqa_score", "flolpips", "hdr_vqm", "st_lpips",
    "camera_jitter_score", "jump_cut_score", "playback_speed_score",
    "flow_coherence", "letterbox_ratio", "tonal_dynamic_range", "vtss", "cnniqa_score",
    "hyperiqa_score", "paq2piq_score", "tres_score", "unique_score",
    "laion_aesthetic", "compare2score", "afine_score", "ckdn_score",
    "deepwsd_score", "ssimulacra2", "butteraugli", "flip_score",
    "vmaf_neg", "ilniqe", "nrqm", "pi_score", "piqe", "maclip_score",
    "dmm", "wadiqam_fr", "ssimc", "cambi", "xpsnr", "vmaf_phone",
    "vmaf_4k", "visqol", "dnsmos_overall", "dnsmos_sig", "dnsmos_bak",
    "pu_psnr", "pu_ssim", "max_fall", "max_cll", "hdr_vdp",
    "delta_ictcp", "ciede2000", "psnr_hvs", "psnr_hvs_m", "cgvqm",
    "strred", "p1203_mos", "nemo_quality_score", "nemo_quality_label",
    "human_fidelity_score", "physics_score", "commonsense_score",
    "creativity_score", "chronomagic_mt_score", "chronomagic_ch_score",
    "compbench_attribute", "compbench_object_rel", "compbench_action",
    "compbench_spatial", "compbench_numeracy", "compbench_scene",
    "compbench_overall",
]


class TestMetricsTable:
    def test_readme_table_count(self):
        assert len(README_METRICS) == 251

    @pytest.mark.parametrize("field_name", README_METRICS)
    def test_readme_metric_exists_in_model(self, field_name):
        """Every metric listed in the README table must be a QualityMetrics field."""
        assert field_name in QualityMetrics.model_fields, (
            f"README lists '{field_name}' but it is not a QualityMetrics field"
        )

    def test_no_unlisted_fields(self):
        """Every QualityMetrics field must appear in the README table."""
        readme_set = set(README_METRICS)
        model_set = set(QualityMetrics.model_fields.keys())
        unlisted = model_set - readme_set
        assert not unlisted, f"Fields in QualityMetrics but not in README: {unlisted}"


# =====================================================================
# README §Python API – scan_dataset, sample.quality_metrics.summary()
# =====================================================================


class TestPythonAPI:
    def test_scan_dataset_returns_samples(self, dataset_dir):
        """README: scan_dataset(Path(...), recursive=True) returns samples."""
        samples = scan_dataset(dataset_dir, recursive=True)
        assert isinstance(samples, list)
        assert len(samples) >= 1
        assert all(isinstance(s, Sample) for s in samples)

    def test_scan_dataset_recursive_param(self, dataset_dir):
        """recursive=True is an accepted parameter."""
        samples = scan_dataset(dataset_dir, recursive=True)
        assert isinstance(samples, list)

    def test_scan_dataset_finds_videos_and_images(self, dataset_dir):
        samples = scan_dataset(dataset_dir, include_videos=True, include_images=True)
        extensions = {s.path.suffix.lower() for s in samples}
        # Our dataset_dir has .mp4 and .png
        assert ".mp4" in extensions
        assert ".png" in extensions

    def test_scan_dataset_attaches_caption(self, dataset_dir):
        """Scanner discovers sidecar .txt captions."""
        samples = scan_dataset(dataset_dir, include_videos=True, include_images=True)
        video_samples = [s for s in samples if s.is_video]
        assert len(video_samples) == 1
        assert video_samples[0].caption is not None
        assert video_samples[0].caption.text == "a short clip"

    def test_quality_metrics_summary(self):
        """README: sample.quality_metrics.summary() returns a string."""
        qm = QualityMetrics(blur_score=120.0, aesthetic_score=7.5, clip_score=0.8)
        summary = qm.summary()
        assert isinstance(summary, str)
        assert "3 metrics" in summary

    def test_quality_metrics_non_null_metrics(self):
        qm = QualityMetrics(technical_score=50.0, motion_score=3.2)
        result = qm.non_null_metrics()
        assert result == {"technical_score": 50.0, "motion_score": 3.2}

    def test_quality_metrics_to_grouped_dict(self):
        qm = QualityMetrics(clip_score=0.8, flow_score=2.0)
        grouped = qm.to_grouped_dict()
        assert "alignment" in grouped
        assert "clip_score" in grouped["alignment"]
        assert "motion" in grouped
        assert "flow_score" in grouped["motion"]


# =====================================================================
# README §Pipeline API – ModuleRegistry, Pipeline, process, export
# =====================================================================


class TestPipelineAPI:
    def test_module_registry_discover(self):
        """README: ModuleRegistry.discover_modules() populates registry."""
        ModuleRegistry.discover_modules()
        modules = ModuleRegistry.list_modules()
        assert isinstance(modules, dict)
        assert len(modules) > 0

    def test_module_registry_get_module(self):
        """README: ModuleRegistry.get_module(name) returns a class."""
        ModuleRegistry.discover_modules()
        cls = ModuleRegistry.get_module("metadata")
        assert cls is not None
        assert issubclass(cls, PipelineModule)

    def test_module_registry_get_module_returns_none_for_unknown(self):
        assert ModuleRegistry.get_module("nonexistent_module_xyz") is None

    def test_pipeline_init_accepts_module_list(self):
        """README: Pipeline(modules) where modules is a list of instances."""
        modules = [_StubModule()]
        pipeline = Pipeline(modules)
        assert len(pipeline.modules) == 1

    def test_pipeline_start_stop_lifecycle(self):
        """README: pipeline.start() then pipeline.stop()."""
        pipeline = Pipeline([_StubModule()])
        pipeline.start()
        pipeline.stop()

    def test_pipeline_process_sample_is_async(self, synthetic_video):
        """README: asyncio.run(pipeline.process_sample(sample))."""
        pipeline = Pipeline([_StubModule()])
        pipeline.start()
        sample = Sample(path=synthetic_video, is_video=True)
        result = asyncio.run(pipeline.process_sample(sample))
        assert isinstance(result, Sample)
        assert result.quality_metrics is not None
        assert result.quality_metrics.technical_score == 75.0
        pipeline.stop()

    def test_pipeline_results_populated(self, synthetic_video):
        """After processing, pipeline.results contains the sample."""
        pipeline = Pipeline([_StubModule()])
        pipeline.start()
        sample = Sample(path=synthetic_video, is_video=True)
        asyncio.run(pipeline.process_sample(sample))
        assert str(synthetic_video) in pipeline.results
        pipeline.stop()

    def test_pipeline_stats_updated(self, synthetic_video):
        """Pipeline.stats tracks counts after processing."""
        pipeline = Pipeline([_StubModule()])
        pipeline.start()
        sample = Sample(path=synthetic_video, is_video=True)
        asyncio.run(pipeline.process_sample(sample))
        assert pipeline.stats.total_samples == 1
        assert pipeline.stats.valid_samples == 1
        pipeline.stop()

    def test_pipeline_running_averages(self):
        """Pipeline computes running averages for key metrics."""
        pipeline = Pipeline([_StubModule()])
        pipeline.start()
        asyncio.run(pipeline.process_sample(Sample(path=Path("a.mp4"), is_video=True)))
        asyncio.run(pipeline.process_sample(Sample(path=Path("b.mp4"), is_video=True)))
        assert pipeline.stats.avg_technical_score == pytest.approx(75.0)
        assert pipeline.stats.avg_aesthetic_score == pytest.approx(6.5)
        pipeline.stop()

    def test_pipeline_export_report_json(self, synthetic_video, tmp_dir):
        """README: pipeline.export_report(path, format='json')."""
        pipeline = Pipeline([_StubModule()])
        pipeline.start()
        asyncio.run(
            pipeline.process_sample(Sample(path=synthetic_video, is_video=True))
        )
        pipeline.stop()

        out = tmp_dir / "report.json"
        pipeline.export_report(out, format="json")
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "stats" in data
        assert "samples" in data
        assert data["stats"]["total_samples"] == 1

    def test_pipeline_export_report_csv(self, synthetic_video, tmp_dir):
        pipeline = Pipeline([_StubModule()])
        pipeline.start()
        asyncio.run(
            pipeline.process_sample(Sample(path=synthetic_video, is_video=True))
        )
        pipeline.stop()

        out = tmp_dir / "report.csv"
        pipeline.export_report(out, format="csv")
        assert out.exists()
        content = out.read_text(encoding="utf-8")
        assert "Path" in content

    def test_pipeline_export_report_html(self, synthetic_video, tmp_dir):
        pipeline = Pipeline([_StubModule()])
        pipeline.start()
        asyncio.run(
            pipeline.process_sample(Sample(path=synthetic_video, is_video=True))
        )
        pipeline.stop()

        out = tmp_dir / "report.html"
        pipeline.export_report(out, format="html")
        assert out.exists()
        assert "<html>" in out.read_text(encoding="utf-8")


# =====================================================================
# README §Pipeline API – instantiate from registry (exact README pattern)
# =====================================================================


class TestPipelineReadmePattern:
    def test_readme_pipeline_pattern(self, synthetic_video):
        """Reproduce the exact Pipeline API example from the README."""
        ModuleRegistry.discover_modules()

        module_names = ["metadata", "basic_quality"]
        modules = [ModuleRegistry.get_module(n)() for n in module_names]
        assert len(modules) == 2

        pipeline = Pipeline(modules)
        pipeline.start()

        sample = Sample(
            path=synthetic_video, is_video=True, quality_metrics=QualityMetrics()
        )
        processed = asyncio.run(pipeline.process_sample(sample))
        assert isinstance(processed, Sample)

        pipeline.stop()

        # Verify something was actually computed
        assert processed.quality_metrics is not None
        assert processed.quality_metrics.technical_score is not None


# =====================================================================
# README §Profile-based pipelines – load_profile, instantiate
# =====================================================================


class TestProfileAPI:
    def test_load_profile_from_toml(self, tmp_dir):
        """README: load_profile('my_profile.toml')."""
        profile_path = tmp_dir / "my_profile.toml"
        profile_path.write_text(
            'name = "readme_test"\n'
            'modules = ["metadata", "basic_quality"]\n',
            encoding="utf-8")
        profile = load_profile(profile_path)
        assert isinstance(profile, PipelineProfile)
        assert profile.modules == ["metadata", "basic_quality"]

    def test_load_profile_from_json(self, tmp_dir):
        profile_path = tmp_dir / "profile.json"
        profile_path.write_text(
            json.dumps(
                {"name": "json_test", "modules": ["metadata"]}
            ),
            encoding="utf-8")
        profile = load_profile(profile_path)
        assert profile.name == "json_test"

    def test_load_profile_from_dict(self):
        profile = load_profile({"modules": ["metadata"]})
        assert isinstance(profile, PipelineProfile)

    def test_load_profile_passthrough(self):
        original = PipelineProfile(modules=["metadata"])
        assert load_profile(original) is original

    def test_instantiate_profile_modules(self):
        """README: instantiate_profile_modules(profile) -> list of PipelineModule."""
        config = AyaseConfig()
        modules = instantiate_profile_modules(
            {"modules": ["metadata", "basic_quality"]},
            config=config)
        assert isinstance(modules, list)
        assert len(modules) == 2
        assert all(isinstance(m, PipelineModule) for m in modules)
        assert modules[0].name == "metadata"

    def test_instantiate_unknown_module_raises(self):
        config = AyaseConfig()
        with pytest.raises(ValueError, match="Unknown module"):
            instantiate_profile_modules(
                {"modules": ["does_not_exist"]}, config=config
            )

    def test_profile_modules_into_pipeline(self, synthetic_video):
        """Full round-trip: profile -> modules -> Pipeline -> process."""
        config = AyaseConfig()
        modules = instantiate_profile_modules(
            {"modules": ["metadata", "basic_quality"]},
            config=config)
        pipeline = Pipeline(modules)
        pipeline.start()

        sample = Sample(
            path=synthetic_video, is_video=True, quality_metrics=QualityMetrics()
        )
        result = asyncio.run(pipeline.process_sample(sample))
        pipeline.stop()

        assert result.quality_metrics is not None


# =====================================================================
# README §Configuration – TOML fields map to actual config classes
# =====================================================================


class TestConfiguration:
    def test_general_config_fields(self):
        """README [general] section fields exist."""
        cfg = GeneralConfig()
        assert isinstance(cfg.parallel_jobs, int)
        assert isinstance(cfg.cache_enabled, bool)

    def test_general_config_has_models_dir(self):
        cfg = GeneralConfig()
        assert hasattr(cfg, "models_dir")

    def test_quality_config_fields(self):
        """README [quality] section fields exist."""
        cfg = QualityConfig()
        assert isinstance(cfg.enable_blur_detection, bool)
        assert isinstance(cfg.blur_threshold, float)

    def test_pipeline_config_fields(self):
        """README [pipeline] section fields exist."""
        cfg = PipelineConfig()
        assert hasattr(cfg, "dataset_path")
        assert isinstance(cfg.modules, list)
        assert isinstance(cfg.plugin_folders, list)

    def test_output_config_fields(self):
        """README [output] section fields exist."""
        cfg = OutputConfig()
        assert cfg.default_format == "markdown"
        assert isinstance(cfg.artifacts_dir, Path)
        assert isinstance(cfg.artifacts_format, str)

    def test_filter_config_fields(self):
        """README [filter] section fields exist."""
        cfg = FilterConfig()
        assert cfg.default_mode == "list"
        assert isinstance(cfg.min_score_threshold, int)

    def test_ayase_config_has_all_sections(self):
        """AyaseConfig composes all sub-configs shown in README."""
        cfg = AyaseConfig()
        assert isinstance(cfg.general, GeneralConfig)
        assert isinstance(cfg.quality, QualityConfig)
        assert isinstance(cfg.pipeline, PipelineConfig)
        assert isinstance(cfg.output, OutputConfig)
        assert isinstance(cfg.filter, FilterConfig)

    def test_ayase_config_no_removed_sections(self):
        """Model-specific sections were removed in the refactoring."""
        cfg = AyaseConfig()
        assert not hasattr(cfg, "wan")
        assert not hasattr(cfg, "hunyuan")
        assert not hasattr(cfg, "kandinsky5")
        assert not hasattr(cfg, "scoring")

    def test_config_load_returns_defaults(self):
        """README: AyaseConfig.load() falls back to defaults."""
        cfg = AyaseConfig.load()
        assert isinstance(cfg, AyaseConfig)

    def test_config_load_from_toml(self, tmp_dir):
        toml_path = tmp_dir / "ayase.toml"
        toml_path.write_text(
            "[general]\nparallel_jobs = 16\n",
            encoding="utf-8")
        cfg = AyaseConfig.load(toml_path)
        assert cfg.general.parallel_jobs == 16


# =====================================================================
# README §Writing Plugins – custom PipelineModule subclass
# =====================================================================


class TestPluginSystem:
    def test_custom_module_auto_registers(self):
        """README: subclassing PipelineModule with a name auto-registers it."""

        class _ReadmePlugin(PipelineModule):
            name = "readme_plugin_test"
            description = "Custom quality check"
            default_config = {"threshold": 0.5}

            def process(self, sample: Sample) -> Sample:
                return sample

        cls = ModuleRegistry.get_module("readme_plugin_test")
        assert cls is _ReadmePlugin

    def test_custom_module_receives_config(self):
        class _ConfigPlugin(PipelineModule):
            name = "readme_config_test"
            description = "Config test"
            default_config = {"threshold": 0.5}

            def process(self, sample: Sample) -> Sample:
                return sample

        m = _ConfigPlugin(config={"threshold": 0.9})
        assert m.config["threshold"] == 0.9

    def test_custom_module_can_append_issues(self):
        """README plugin example appends ValidationIssue to sample."""

        class _IssuePlugin(PipelineModule):
            name = "readme_issue_test"
            description = "Issue test"
            default_config = {"threshold": 0.5}

            def process(self, sample: Sample) -> Sample:
                sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message="Quality below threshold")
                )
                return sample

        sample = Sample(path=Path("dummy.mp4"), is_video=True)
        result = _IssuePlugin().process(sample)
        assert len(result.validation_issues) == 1
        assert result.validation_issues[0].severity == ValidationSeverity.WARNING
        assert result.validation_issues[0].message == "Quality below threshold"

    def test_custom_module_in_pipeline(self, synthetic_video):
        """A custom plugin can be used in a Pipeline end-to-end."""

        class _E2EPlugin(PipelineModule):
            name = "readme_e2e_test"
            description = "E2E test"

            def process(self, sample: Sample) -> Sample:
                if sample.quality_metrics is None:
                    sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.aesthetic_score = 9.0
                return sample

        pipeline = Pipeline([_E2EPlugin()])
        pipeline.start()
        sample = Sample(path=synthetic_video, is_video=True)
        result = asyncio.run(pipeline.process_sample(sample))
        pipeline.stop()

        assert result.quality_metrics.aesthetic_score == 9.0


# =====================================================================
# README §Overview – Sample model fields and properties
# =====================================================================


class TestSampleModel:
    def test_sample_required_fields(self):
        s = Sample(path=Path("video.mp4"), is_video=True)
        assert s.path == Path("video.mp4")
        assert s.is_video is True

    def test_sample_optional_fields_default_none(self):
        s = Sample(path=Path("a.mp4"), is_video=True)
        assert s.quality_metrics is None
        assert s.caption is None
        assert s.video_metadata is None
        assert s.embedding is None

    def test_sample_is_valid_with_no_issues(self):
        s = Sample(path=Path("a.mp4"), is_video=True)
        assert s.is_valid is True

    def test_sample_is_valid_with_error(self):
        s = Sample(path=Path("a.mp4"), is_video=True)
        s.validation_issues.append(
            ValidationIssue(severity=ValidationSeverity.ERROR, message="bad")
        )
        assert s.is_valid is False

    def test_sample_is_valid_with_warning_only(self):
        s = Sample(path=Path("a.mp4"), is_video=True)
        s.validation_issues.append(
            ValidationIssue(severity=ValidationSeverity.WARNING, message="meh")
        )
        assert s.is_valid is True

    def test_sample_no_composite_score(self):
        """composite_score was removed from Sample in the refactoring."""
        s = Sample(path=Path("a.mp4"), is_video=True)
        assert not hasattr(s, "composite_score") or "composite_score" not in Sample.model_fields


# =====================================================================
# README §Overview – removed model-specific types
# =====================================================================


class TestRemovedTypes:
    def test_no_model_type_in_models(self):
        from ayase import models
        assert not hasattr(models, "ModelType")

    def test_no_validation_result_in_models(self):
        from ayase import models
        assert not hasattr(models, "ValidationResult")

    def test_no_model_requirements_in_models(self):
        from ayase import models
        assert not hasattr(models, "ModelRequirements")

    def test_no_validator_module(self):
        with pytest.raises(ImportError):
            import ayase.validator  # noqa: F401

    def test_no_quality_module(self):
        with pytest.raises(ImportError):
            import ayase.quality  # noqa: F401


# =====================================================================
# README – __init__.py public API
# =====================================================================


class TestPublicAPI:
    def test_version(self):
        import ayase
        assert hasattr(ayase, "__version__")
        assert isinstance(ayase.__version__, str)

    def test_description(self):
        """README: 'Modular media quality metrics toolkit'."""
        import ayase
        assert ayase.__description__ == "Modular media quality metrics toolkit"

    def test_exports(self):
        from ayase import PipelineProfile, instantiate_profile_modules, load_profile
        assert callable(load_profile)
        assert callable(instantiate_profile_modules)


# =====================================================================
# README – scanner constants
# =====================================================================


class TestScannerConstants:
    def test_video_extensions(self):
        for ext in [".mp4", ".webm", ".avi", ".mov", ".mkv"]:
            assert ext in VIDEO_EXTENSIONS

    def test_image_extensions(self):
        for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
            assert ext in IMAGE_EXTENSIONS

    def test_scan_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            scan_dataset(Path("/nonexistent/dataset/path"))


# =====================================================================
# README – Pipeline state save/load
# =====================================================================


class TestPipelineStatePersistence:
    def test_save_and_load_state(self, synthetic_video, tmp_dir):
        pipeline = Pipeline([_StubModule()])
        pipeline.start()
        asyncio.run(
            pipeline.process_sample(Sample(path=synthetic_video, is_video=True))
        )
        pipeline.stop()

        state_path = tmp_dir / "state.json"
        pipeline.save_state(state_path)
        assert state_path.exists()

        pipeline2 = Pipeline([_StubModule()])
        pipeline2.load_state(state_path)
        assert pipeline2.stats.total_samples == 1
        assert str(synthetic_video) in pipeline2.results


# =====================================================================
# README – CLI commands exist (import-level check, not subprocess)
# =====================================================================


class TestCLICommands:
    """Verify CLI commands referenced in README are registered."""

    def test_cli_app_exists(self):
        from ayase.cli import app
        assert app is not None

    def test_scan_command_exists(self):
        from ayase.cli import scan
        assert callable(scan)

    def test_run_command_exists(self):
        from ayase.cli import run
        assert callable(run)

    def test_filter_command_exists(self):
        from ayase.cli import filter
        assert callable(filter)

    def test_stats_command_exists(self):
        from ayase.cli import stats
        assert callable(stats)

    def test_tui_command_exists(self):
        from ayase.cli import tui
        assert callable(tui)

    def test_modules_list_command_exists(self):
        from ayase.cli import modules_list
        assert callable(modules_list)

    def test_modules_check_command_exists(self):
        from ayase.cli import modules_check
        assert callable(modules_check)

    def test_config_subcommands_exist(self):
        from ayase.cli import config_init, config_show, config_edit, config_validate
        assert callable(config_init)
        assert callable(config_show)
        assert callable(config_validate)

    def test_no_validate_command(self):
        """The 'validate' command was removed in the refactoring."""
        from ayase import cli
        assert not hasattr(cli, "validate") or not callable(getattr(cli, "validate", None))


# =====================================================================
# AyasePipeline high-level facade
# =====================================================================


def _make_dataset(tmp_path: Path) -> Path:
    """Create a minimal dataset with one image."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(dataset / "photo.png"), img)
    return dataset


class _AlwaysFailsMount(PipelineModule):
    """Module whose setup() always raises, simulating missing ML dep."""
    name = "always_fails_mount_test"
    description = "Always fails to mount"

    def __init__(self, config=None):
        super().__init__(config)
        self.process_calls = 0

    def setup(self) -> None:
        raise ImportError("Simulated missing package")

    def process(self, sample: Sample) -> Sample:
        self.process_calls += 1
        return sample


class TestAyasePipeline:
    """Tests for the AyasePipeline entry point."""

    def test_import_from_ayase(self):
        from ayase import AyasePipeline
        assert AyasePipeline is not None

    def test_init_no_modules(self):
        ayase = AyasePipeline()
        assert ayase.pipeline is not None
        assert len(ayase._modules) == 0

    def test_init_with_modules(self):
        ayase = AyasePipeline(modules=["basic"])
        assert len(ayase._modules) == 1
        assert ayase._modules[0].name == "basic"

    def test_init_with_config_path(self, tmp_dir):
        toml_path = tmp_dir / "ayase.toml"
        toml_path.write_text("[general]\nparallel_jobs = 42\n", encoding="utf-8")
        ayase = AyasePipeline(config=toml_path)
        assert ayase.config.general.parallel_jobs == 42

    def test_init_with_config_object(self):
        cfg = AyaseConfig()
        ayase = AyasePipeline(config=cfg)
        assert ayase.config is cfg

    def test_init_with_profile_dict(self):
        ayase = AyasePipeline(profile={"modules": ["basic"]})
        assert len(ayase._modules) == 1

    def test_init_unknown_module_raises(self):
        with pytest.raises(ValueError, match="Unknown module"):
            AyasePipeline(modules=["nonexistent_module_xyz"])

    def test_run_returns_results(self, dataset_dir):
        ayase = AyasePipeline(modules=["basic"])
        results = ayase.run(dataset_dir)
        assert isinstance(results, dict)
        assert len(results) >= 1

    def test_run_populates_stats(self, dataset_dir):
        ayase = AyasePipeline(modules=["basic"])
        ayase.run(dataset_dir)
        assert ayase.stats.total_samples >= 1

    def test_run_samples_have_metrics(self, dataset_dir):
        ayase = AyasePipeline(modules=["basic"])
        results = ayase.run(dataset_dir)
        for sample in results.values():
            assert sample.quality_metrics is not None

    def test_results_property(self, dataset_dir):
        ayase = AyasePipeline(modules=["basic"])
        ayase.run(dataset_dir)
        assert ayase.results is ayase.pipeline.results

    def test_stats_property(self, dataset_dir):
        ayase = AyasePipeline(modules=["basic"])
        ayase.run(dataset_dir)
        assert ayase.stats is ayase.pipeline.stats

    def test_export_json(self, dataset_dir, tmp_dir):
        ayase = AyasePipeline(modules=["basic"])
        ayase.run(dataset_dir)
        out = tmp_dir / "report.json"
        ayase.export(out)
        assert out.exists()
        import json
        data = json.loads(out.read_text(encoding="utf-8"))
        assert "stats" in data
        assert "samples" in data

    def test_export_csv(self, dataset_dir, tmp_dir):
        ayase = AyasePipeline(modules=["basic"])
        ayase.run(dataset_dir)
        out = tmp_dir / "report.csv"
        ayase.export(out, format="csv")
        assert out.exists()

    def test_run_with_prebuilt_samples(self, dataset_dir):
        samples = scan_dataset(dataset_dir)
        ayase = AyasePipeline(modules=["basic"])
        results = ayase.run(dataset_dir, samples=samples)
        assert len(results) == len(samples)

    def test_ayase_pipeline_inline_models_dir(self, tmp_dir):
        """AyasePipeline accepts inline config with custom models_dir."""
        models_dir = tmp_dir / "my_models"
        models_dir.mkdir()

        cfg = AyaseConfig(general=GeneralConfig(models_dir=models_dir))
        ayase = AyasePipeline(config=cfg, modules=["basic"])

        assert ayase.config.general.models_dir == models_dir
        assert len(ayase._modules) == 1
        assert ayase._modules[0].config["models_dir"] == str(models_dir)

    def test_ayase_pipeline_e2e_with_tiered_module(self, tmp_dir):
        """End-to-end: AyasePipeline with a module that has fallback tiers."""
        dataset = _make_dataset(tmp_dir)

        # hdr_sdr_vqa is a good test — no ML deps, always mounts, produces a metric
        ayase = AyasePipeline(modules=["hdr_sdr_vqa"])
        results = ayase.run(dataset)

        assert len(results) == 1
        sample = list(results.values())[0]
        assert sample.quality_metrics is not None
        # Should produce sdr_quality for a uint8 image
        assert sample.quality_metrics.sdr_quality is not None

    def test_ayase_pipeline_e2e_multiple_modules(self, tmp_dir):
        """End-to-end with multiple modules on an image dataset."""
        dataset = _make_dataset(tmp_dir)

        ayase = AyasePipeline(modules=["basic", "hdr_sdr_vqa"])
        results = ayase.run(dataset)

        assert len(results) == 1
        sample = list(results.values())[0]
        qm = sample.quality_metrics
        assert qm is not None
        # basic module produces blur_score, hdr_sdr_vqa produces sdr_quality
        assert qm.blur_score is not None
        assert qm.sdr_quality is not None

    def test_ayase_pipeline_skips_failed_mount_module(self, tmp_dir):
        """Modules that fail on_mount are skipped during processing (no crash)."""
        dataset = _make_dataset(tmp_dir)

        # Build pipeline with one good module and one that always fails
        ModuleRegistry.discover_modules()
        good_cls = ModuleRegistry.get_module("basic")
        fail_module = _AlwaysFailsMount()
        good_module = good_cls()

        pipeline = Pipeline([fail_module, good_module])
        pipeline.start()

        # fail_module should NOT be mounted
        assert not fail_module._mounted
        # good_module should be mounted
        assert good_module._mounted

        sample = Sample(path=list(Path(dataset).glob("*"))[0], is_video=False)
        asyncio.run(pipeline.process_sample(sample))

        # fail_module was skipped, good_module ran
        assert fail_module.process_calls == 0
        assert sample.quality_metrics is not None

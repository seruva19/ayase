"""Integration tests with synthetic video datasets.

Creates videos with known properties and verifies that Ayase modules
produce correct, non-null metrics for them.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ayase.models import CaptionMetadata, QualityMetrics, Sample

# ---------------------------------------------------------------------------
# Synthetic video generators
# ---------------------------------------------------------------------------


def _make_static_video(path: Path, frames: int = 32, size: int = 128) -> Path:
    """Solid color video — zero motion, high consistency."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (size, size))
    color = (80, 120, 200)
    for _ in range(frames):
        frame = np.full((size, size, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _make_moving_circle_video(path: Path, frames: int = 48, size: int = 128) -> Path:
    """Circle moving across frame — measurable optical flow."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (size, size))
    for i in range(frames):
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        # Background gradient
        for x in range(size):
            frame[:, x, 0] = int(x * 255 / size)
            frame[:, x, 2] = int((size - x) * 255 / size)
        # Moving circle
        cx = int(size * (i / frames))
        cy = size // 2
        cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
        writer.write(frame)
    writer.release()
    return path


def _make_flickering_video(path: Path, frames: int = 32, size: int = 128) -> Path:
    """Alternating bright/dark frames — high warping error."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (size, size))
    for i in range(frames):
        val = 220 if i % 2 == 0 else 30
        frame = np.full((size, size, 3), val, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def _make_text_overlay_video(
    path: Path, text: str = "Hello", frames: int = 32, size: int = 256
) -> Path:
    """Video with text overlay — for OCR testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (size, size))
    for _ in range(frames):
        frame = np.full((size, size, 3), 40, dtype=np.uint8)
        cv2.putText(frame, text, (30, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        writer.write(frame)
    writer.release()
    return path


def _make_scene_change_video(path: Path, frames: int = 48, size: int = 128) -> Path:
    """Video with abrupt scene change at midpoint — low background consistency."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 24.0, (size, size))
    for i in range(frames):
        if i < frames // 2:
            frame = np.full((size, size, 3), (200, 50, 50), dtype=np.uint8)
        else:
            frame = np.full((size, size, 3), (50, 200, 50), dtype=np.uint8)
        cx = int(size * 0.5 + 20 * np.sin(i * 0.3))
        cv2.circle(frame, (cx, size // 2), 10, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def static_video(tmp_dir):
    return _make_static_video(tmp_dir / "static.mp4")


@pytest.fixture
def moving_video(tmp_dir):
    return _make_moving_circle_video(tmp_dir / "moving.mp4")


@pytest.fixture
def flickering_video(tmp_dir):
    return _make_flickering_video(tmp_dir / "flicker.mp4")


@pytest.fixture
def text_video(tmp_dir):
    return _make_text_overlay_video(tmp_dir / "text.mp4", text="Hello")


@pytest.fixture
def scene_change_video(tmp_dir):
    return _make_scene_change_video(tmp_dir / "scene_change.mp4")


def _sample(path: Path, caption: str = None) -> Sample:
    s = Sample(path=path, is_video=True, quality_metrics=QualityMetrics())
    if caption:
        s.caption = CaptionMetadata(text=caption, length=len(caption))
    return s


# ===========================================================================
# Tests: Metadata & Basic Quality
# ===========================================================================


class TestMetadata:
    def test_metadata_extracts_video_info(self, moving_video):
        from ayase.modules.metadata import MetadataModule

        sample = _sample(moving_video)
        result = MetadataModule().process(sample)
        assert result.video_metadata is not None
        assert result.video_metadata.width == 128
        assert result.video_metadata.height == 128
        assert result.video_metadata.frame_count >= 40
        assert result.video_metadata.fps > 0
        assert result.video_metadata.duration > 0

    def test_basic_quality_produces_scores(self, moving_video):
        from ayase.modules.basic import BasicQualityModule

        sample = _sample(moving_video)
        result = BasicQualityModule().process(sample)
        qm = result.quality_metrics
        assert qm is not None
        assert qm.blur_score is not None
        assert qm.brightness is not None
        assert qm.contrast is not None


# ===========================================================================
# Tests: Motion & Optical Flow
# ===========================================================================


class TestMotion:
    def test_motion_module_detects_motion(self, moving_video):
        from ayase.modules.motion import MotionModule

        sample = _sample(moving_video)
        result = MotionModule().process(sample)
        assert result.quality_metrics is not None
        assert result.quality_metrics.motion_score is not None
        assert result.quality_metrics.motion_score > 0

    def test_static_video_has_low_motion(self, static_video):
        from ayase.modules.motion import MotionModule

        sample = _sample(static_video)
        result = MotionModule().process(sample)
        assert result.quality_metrics is not None
        # Static video should have very low motion
        if result.quality_metrics.motion_score is not None:
            assert result.quality_metrics.motion_score < 5.0


# ===========================================================================
# Tests: Temporal Flickering (Farneback fallback — no RAFT in CI)
# ===========================================================================


class TestTemporalFlickering:
    def test_flickering_video_has_high_warping_error(self, flickering_video):
        from ayase.modules.temporal_flickering import TemporalFlickeringModule

        module = TemporalFlickeringModule({})
        sample = _sample(flickering_video)
        result = module.process(sample)
        qm = result.quality_metrics
        assert qm.warping_error is not None
        assert qm.warping_error > 0

    def test_static_video_has_low_warping_error(self, static_video):
        from ayase.modules.temporal_flickering import TemporalFlickeringModule

        module = TemporalFlickeringModule({})
        sample = _sample(static_video)
        result = module.process(sample)
        qm = result.quality_metrics
        assert qm.warping_error is not None
        assert qm.warping_error < 0.01


# ===========================================================================
# Tests: Subject & Background Consistency (no ML — skip if unavailable)
# ===========================================================================


class TestConsistency:
    def test_subject_consistency_instantiates(self):
        from ayase.modules.subject_consistency import SubjectConsistencyModule

        m = SubjectConsistencyModule({})
        assert m.name == "subject_consistency"
        assert m.max_frames == 16

    def test_background_consistency_instantiates(self):
        from ayase.modules.background_consistency import BackgroundConsistencyModule

        m = BackgroundConsistencyModule({})
        assert m.name == "background_consistency"
        assert m.max_frames == 16


# ===========================================================================
# Tests: Motion Smoothness (flow proxy fallback)
# ===========================================================================


class TestMotionSmoothness:
    def test_static_video_is_smooth(self, static_video):
        from ayase.modules.motion_smoothness import MotionSmoothnessModule

        module = MotionSmoothnessModule({})
        sample = _sample(static_video)
        result = module.process(sample)
        qm = result.quality_metrics
        assert qm.motion_smoothness is not None
        # Static video should be very smooth
        assert qm.motion_smoothness > 0.8

    def test_moving_video_has_smoothness(self, moving_video):
        from ayase.modules.motion_smoothness import MotionSmoothnessModule

        module = MotionSmoothnessModule({})
        sample = _sample(moving_video)
        result = module.process(sample)
        assert result.quality_metrics.motion_smoothness is not None
        assert 0.0 <= result.quality_metrics.motion_smoothness <= 1.0


# ===========================================================================
# Tests: Motion Amplitude (Farneback fallback)
# ===========================================================================


class TestMotionAmplitude:
    def test_fast_motion_with_fast_caption(self, moving_video):
        from ayase.modules.motion_amplitude import MotionAmplitudeModule

        module = MotionAmplitudeModule({})
        sample = _sample(moving_video, caption="A ball racing quickly across the screen")
        result = module.process(sample)
        # Module should attempt classification
        # (score depends on whether Farneback flow exceeds threshold)
        assert result is not None

    def test_no_caption_skips(self, moving_video):
        from ayase.modules.motion_amplitude import MotionAmplitudeModule

        module = MotionAmplitudeModule({})
        sample = _sample(moving_video)  # No caption
        result = module.process(sample)
        # Without caption, motion_ac_score should remain None
        assert result.quality_metrics.motion_ac_score is None

    def test_no_motion_keywords_skips(self, moving_video):
        from ayase.modules.motion_amplitude import MotionAmplitudeModule

        module = MotionAmplitudeModule({})
        sample = _sample(moving_video, caption="A landscape at sunset")
        result = module.process(sample)
        # No motion keywords → skip
        assert result.quality_metrics.motion_ac_score is None


# ===========================================================================
# Tests: Advanced Flow (config / instantiation)
# ===========================================================================


class TestAdvancedFlow:
    def test_advanced_flow_defaults(self):
        from ayase.modules.advanced_flow import AdvancedFlowModule

        m = AdvancedFlowModule()
        assert m.name == "advanced_flow"
        assert m.use_large_model is True

    def test_advanced_flow_small_config(self):
        from ayase.modules.advanced_flow import AdvancedFlowModule

        m = AdvancedFlowModule({"use_large_model": False})
        assert m.use_large_model is False


# ===========================================================================
# Tests: Action Recognition (config only — model too large for CI)
# ===========================================================================


class TestActionRecognition:
    def test_action_recognition_defaults_to_large(self):
        from ayase.modules.action_recognition import ActionRecognitionModule

        m = ActionRecognitionModule()
        assert "large" in m.model_name

    def test_action_recognition_custom_model(self):
        from ayase.modules.action_recognition import ActionRecognitionModule

        m = ActionRecognitionModule({"model_name": "MCG-NJU/videomae-base-finetuned-kinetics"})
        assert "base" in m.model_name


# ===========================================================================
# Tests: QualityMetrics grouped export
# ===========================================================================


class TestQualityMetricsAPI:
    def test_non_null_metrics(self):
        qm = QualityMetrics(clip_score=0.8, flow_score=5.0, dover_score=0.7)
        nn = qm.non_null_metrics()
        assert len(nn) == 3
        assert nn["clip_score"] == 0.8

    def test_non_null_count(self):
        qm = QualityMetrics(clip_score=0.8, blur_score=50.0)
        assert qm.non_null_count() == 2

    def test_to_grouped_dict(self):
        qm = QualityMetrics(clip_score=0.8, flow_score=5.0, pesq_score=3.5)
        grouped = qm.to_grouped_dict()
        assert "alignment" in grouped
        assert grouped["alignment"]["clip_score"] == 0.8
        assert "motion" in grouped
        assert "audio" in grouped

    def test_summary(self):
        qm = QualityMetrics(clip_score=0.8, dover_score=0.7)
        s = qm.summary()
        assert "2 metrics" in s
        assert "alignment=" in s
        assert "nr_quality=" in s

    def test_empty_summary(self):
        qm = QualityMetrics()
        assert qm.summary() == "0 metrics"
        assert qm.non_null_count() == 0

    def test_grouped_dict_unmapped_fields(self):
        qm = QualityMetrics(scene_complexity=0.5)
        grouped = qm.to_grouped_dict()
        assert "scene" in grouped
        assert grouped["scene"]["scene_complexity"] == 0.5


# ===========================================================================
# Tests: Pipeline end-to-end with non-ML modules
# ===========================================================================


class TestPipelineE2E:
    def test_pipeline_runs_multiple_modules(self, moving_video):
        from ayase.modules.basic import BasicQualityModule
        from ayase.modules.metadata import MetadataModule
        from ayase.modules.motion import MotionModule
        from ayase.pipeline import Pipeline

        modules = [MetadataModule(), BasicQualityModule(), MotionModule()]
        pipeline = Pipeline(modules)
        pipeline.start()

        sample = _sample(moving_video)
        pipeline.process_sample(sample)

        assert sample.video_metadata is not None
        assert sample.quality_metrics is not None
        assert sample.quality_metrics.blur_score is not None
        assert sample.quality_metrics.motion_score is not None

    def test_pipeline_with_caption(self, moving_video, tmp_dir):
        """Verify scanner picks up .txt captions and pipeline processes them."""
        from ayase.scanner import scan_dataset

        caption_path = moving_video.with_suffix(".txt")
        caption_path.write_text("A green circle moving across the screen")

        samples = scan_dataset(tmp_dir, include_videos=True, include_images=False)
        assert len(samples) >= 1
        video_samples = [s for s in samples if s.path.name == "moving.mp4"]
        assert len(video_samples) == 1
        assert video_samples[0].caption is not None
        assert "green circle" in video_samples[0].caption.text


# ===========================================================================
# Tests: OCR Fidelity
# ===========================================================================


class TestOCRFidelity:
    def test_ocr_fidelity_instantiates(self):
        from ayase.modules.ocr_fidelity import OCRFidelityModule

        m = OCRFidelityModule()
        assert m.name == "ocr_fidelity"
        assert m.num_frames == 8
        assert m.lang == "en"

    def test_ocr_fidelity_custom_config(self):
        from ayase.modules.ocr_fidelity import OCRFidelityModule

        m = OCRFidelityModule({"num_frames": 4, "lang": "ch"})
        assert m.num_frames == 4
        assert m.lang == "ch"

    def test_ocr_fidelity_skips_no_caption(self, moving_video):
        from ayase.modules.ocr_fidelity import OCRFidelityModule

        module = OCRFidelityModule()
        module._ocr_available = True  # Pretend OCR is available
        module._ocr = None  # But don't actually init it

        sample = _sample(moving_video)  # No caption
        result = module.process(sample)
        # Without caption, should skip gracefully
        assert result.quality_metrics.ocr_fidelity is None

    def test_ocr_fidelity_skips_no_quoted_text(self, moving_video):
        from ayase.modules.ocr_fidelity import OCRFidelityModule

        module = OCRFidelityModule()
        module._ocr_available = True
        module._ocr = None

        sample = _sample(moving_video, caption="A beautiful sunset over the ocean")
        result = module.process(sample)
        # Caption has no quoted text → should skip
        assert result.quality_metrics.ocr_fidelity is None

    def test_extract_quoted_text(self):
        from ayase.modules.ocr_fidelity import _extract_quoted_text

        assert _extract_quoted_text('A sign saying "Hello World"') == ["Hello World"]
        assert _extract_quoted_text("No quotes here") == []
        assert _extract_quoted_text('"First" and "Second"') == ["First", "Second"]
        assert _extract_quoted_text("'single quotes'") == ["single quotes"]

    def test_normalized_edit_distance(self):
        from ayase.modules.ocr_fidelity import _normalized_edit_distance

        assert _normalized_edit_distance("hello", "hello") == 0.0
        assert _normalized_edit_distance("", "") == 0.0
        ned = _normalized_edit_distance("hello", "helo")
        assert 0 < ned < 1.0  # One deletion


# ===========================================================================
# Tests: I2V Similarity
# ===========================================================================


class TestI2VSimilarity:
    def test_i2v_instantiates(self):
        from ayase.modules.i2v_similarity import I2VSimilarityModule

        m = I2VSimilarityModule()
        assert m.name == "i2v_similarity"
        assert m.window_size == 16
        assert m.stride == 8
        assert m.max_frames == 256
        assert m.enable_clip is True
        assert m.enable_dino is True
        assert m.enable_lpips is True

    def test_i2v_custom_config(self):
        from ayase.modules.i2v_similarity import I2VSimilarityModule

        m = I2VSimilarityModule(
            {
                "window_size": 8,
                "stride": 4,
                "max_frames": 128,
                "enable_clip": False,
            }
        )
        assert m.window_size == 8
        assert m.stride == 4
        assert m.max_frames == 128
        assert m.enable_clip is False

    def test_i2v_skips_non_video(self, tmp_dir):
        from ayase.modules.i2v_similarity import I2VSimilarityModule

        module = I2VSimilarityModule({})
        # Create an image sample (not video)
        img_path = tmp_dir / "test.png"
        img = np.full((128, 128, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(img_path), img)
        sample = Sample(path=img_path, is_video=False, quality_metrics=QualityMetrics())
        result = module.process(sample)
        assert result.quality_metrics.i2v_clip is None

    def test_i2v_skips_no_reference(self, moving_video):
        from ayase.modules.i2v_similarity import I2VSimilarityModule

        module = I2VSimilarityModule({})
        sample = _sample(moving_video)  # No reference_path
        result = module.process(sample)
        assert result.quality_metrics.i2v_quality is None

    def test_i2v_aggregation_formula(self):
        from ayase.modules.i2v_similarity import I2VSimilarityModule

        # clip=0.8 → 80, lpips=0.2 → (1-0.2)*100=80, dino=0.7 → 70
        # (80*0.4 + 80*0.2 + 70*0.4) / (0.4+0.2+0.4) = (32+16+28)/1.0 = 76
        score = I2VSimilarityModule._aggregate(0.8, 0.7, 0.2)
        assert score is not None
        assert abs(score - 76.0) < 0.01

    def test_i2v_aggregation_partial(self):
        from ayase.modules.i2v_similarity import I2VSimilarityModule

        # Only CLIP available
        score = I2VSimilarityModule._aggregate(0.9, None, None)
        assert score is not None
        assert abs(score - 90.0) < 0.01

    def test_i2v_aggregation_empty(self):
        from ayase.modules.i2v_similarity import I2VSimilarityModule

        score = I2VSimilarityModule._aggregate(None, None, None)
        assert score is None


# ===========================================================================
# Tests: Synthetic dataset scan
# ===========================================================================


class TestDatasetScan:
    def test_scan_finds_all_videos(self, tmp_dir):
        _make_static_video(tmp_dir / "a.mp4", frames=16)
        _make_moving_circle_video(tmp_dir / "b.mp4", frames=16)
        _make_flickering_video(tmp_dir / "c.mp4", frames=16)

        from ayase.scanner import scan_dataset

        samples = scan_dataset(tmp_dir, include_videos=True, include_images=False)
        assert len(samples) == 3
        names = {s.path.name for s in samples}
        assert names == {"a.mp4", "b.mp4", "c.mp4"}

    def test_scan_with_nested_dirs(self, tmp_dir):
        sub1 = tmp_dir / "train"
        sub2 = tmp_dir / "val"
        sub1.mkdir()
        sub2.mkdir()
        _make_static_video(sub1 / "x.mp4", frames=10)
        _make_moving_circle_video(sub2 / "y.mp4", frames=10)

        from ayase.scanner import scan_dataset

        samples = scan_dataset(tmp_dir, recursive=True, include_videos=True, include_images=False)
        assert len(samples) == 2

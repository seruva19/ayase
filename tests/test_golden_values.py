"""Golden-value regression tests.

Fixed-seed synthetic fixtures + exact numeric golden values for 20 modules (~45 fields).
Catches logic bugs, broken formulas, wrong scale, silent regressions.

Tolerance: pytest.approx(expected, rel=0.02, abs=0.5) — 2% relative or 0.5 absolute.

Fixtures use np.random.RandomState(seed) for determinism (NOT conftest fixtures).
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from ayase.models import QualityMetrics, Sample

# ── pyiqa availability ───────────────────────────────────────────────────
try:
    import pyiqa

    HAS_PYIQA = True
except ImportError:
    HAS_PYIQA = False


# ── Golden values (discovered on deterministic fixtures) ─────────────────
# Re-generate with: python tests/discover_golden_values.py

GOLDEN = {
    "basic_image": {
        "blur_score": 630.852138,
        "brightness": 141.434341,
        "contrast": 73.203445,
        "saturation": 36.762192,
        "noise_score": 0.381820,
        "artifacts_score": 1.000000,
        "technical_score": 69.520353,
        "gradient_detail": 38.788473,
    },
    "basic_video": {
        "blur_score": 191.374161,
        "brightness": 24.051056,
        "contrast": 31.346585,
        "saturation": 254.843399,
        "noise_score": 0.628739,
        "artifacts_score": 0.953079,
        "technical_score": 48.149353,
        "gradient_detail": 6.885284,
    },
    "motion": {
        "motion_score": 3.959223,
    },
    "ti_si_video": {
        "spatial_information": 41.965851,
        "temporal_information": 20.549837,
    },
    "compression_image": {
        "compression_artifacts": 12.781250,
    },
    "compression_video": {
        "compression_artifacts": 16.071806,
    },
    "flicker": {
        "flicker_score": 0.035070,
    },
    "judder_stutter": {
        "judder_score": 5.095755,
        "stutter_score": 0.000000,
    },
    "camera_jitter": {
        "camera_jitter_score": 1.000000,
    },
    "scene_complexity_image": {
        "scene_complexity": 51.343504,
    },
    "scene_complexity_video": {
        "scene_complexity": 10.656110,
    },
    "letterbox": {
        "letterbox_ratio": 0.007812,
    },
    "hdr_metadata": {
        "max_fall": 49.780792,
        "max_cll": 1610.198120,
    },
    "scene_detection": {
        "scene_stability": 1.000000,
        "avg_scene_duration": 2.133333,
    },
    "dynamics_range": {
        "dynamics_range": 0.446293,
    },
    "flow_coherence": {
        "flow_coherence": 0.277577,
    },
    "naturalness": {
        "naturalness_score": 0.585167,
    },
    "ciede2000": {
        "ciede2000": 9.068089,
    },
    "psnr_hvs": {
        "psnr_hvs": 24.656172,
    },
    "pu_metrics": {
        "pu_psnr": 32.369150,
        "pu_ssim": 0.996600,
    },
    "brisque": {
        "brisque": 74.743347,
    },
    "niqe": {
        "niqe": 30.158907,
    },
}

# Default tolerance for golden comparisons
TOL = dict(rel=0.02, abs=0.5)


# ── Deterministic fixtures ───────────────────────────────────────────────


@pytest.fixture(scope="module")
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture(scope="module")
def golden_image(tmp_dir):
    """256x256 gradient + noise (RandomState(42))."""
    rng = np.random.RandomState(42)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for x in range(256):
        img[:, x, :] = x
    noise = rng.randint(0, 30, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    path = tmp_dir / "golden_image.png"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture(scope="module")
def golden_video(tmp_dir):
    """64 frames @ 30fps, 256x256, moving green circle."""
    path = tmp_dir / "golden_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (256, 256))
    for i in range(64):
        frame = np.zeros((256, 256, 3), dtype=np.uint8)
        cx = 128 + int(50 * np.sin(i * 0.2))
        cy = 128 + int(50 * np.cos(i * 0.2))
        cv2.circle(frame, (cx, cy), 30, (0, 255, 0), -1)
        for x in range(256):
            frame[:, x, 0] = min(255, x + i)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture(scope="module")
def degraded_image(golden_image, tmp_dir):
    """golden_image + GaussianBlur(5,5) + noise (RandomState(99))."""
    rng = np.random.RandomState(99)
    img = cv2.imread(str(golden_image))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    noise = rng.randint(0, 20, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    path = tmp_dir / "degraded_image.png"
    cv2.imwrite(str(path), img)
    return path


# ── Helpers ──────────────────────────────────────────────────────────────


def _run(module_cls, sample, config=None):
    """Instantiate, setup, process — return sample."""
    m = module_cls(config=config)
    if hasattr(m, "setup"):
        m.setup()
    return m.process(sample)


def _img_sample(path):
    return Sample(path=path, is_video=False, quality_metrics=QualityMetrics())


def _vid_sample(path):
    return Sample(path=path, is_video=True, quality_metrics=QualityMetrics())


def _ref_sample(path, ref_path):
    return Sample(
        path=path, is_video=False, reference_path=ref_path,
        quality_metrics=QualityMetrics())


# ═════════════════════════════════════════════════════════════════════════
# Tier 1: Pure OpenCV modules (15 modules)
# ═════════════════════════════════════════════════════════════════════════


class TestBasicImage:
    """BasicQualityModule on golden image — 8 fields."""

    def _result(self, golden_image):
        from ayase.modules.basic import BasicQualityModule

        return _run(BasicQualityModule, _img_sample(golden_image))

    @pytest.fixture(scope="class")
    def qm(self, golden_image):
        return self._result(golden_image).quality_metrics

    def test_blur_score(self, qm):
        assert qm.blur_score is not None
        assert isinstance(qm.blur_score, float)
        assert qm.blur_score == pytest.approx(GOLDEN["basic_image"]["blur_score"], **TOL)

    def test_brightness(self, qm):
        assert qm.brightness is not None
        assert 0 <= qm.brightness <= 255
        assert qm.brightness == pytest.approx(GOLDEN["basic_image"]["brightness"], **TOL)

    def test_contrast(self, qm):
        assert qm.contrast is not None
        assert 0 <= qm.contrast <= 255
        assert qm.contrast == pytest.approx(GOLDEN["basic_image"]["contrast"], **TOL)

    def test_saturation(self, qm):
        assert qm.saturation is not None
        assert 0 <= qm.saturation <= 255
        assert qm.saturation == pytest.approx(GOLDEN["basic_image"]["saturation"], **TOL)

    def test_noise_score(self, qm):
        assert qm.noise_score is not None
        assert 0 <= qm.noise_score <= 1
        assert qm.noise_score == pytest.approx(GOLDEN["basic_image"]["noise_score"], **TOL)

    def test_artifacts_score(self, qm):
        assert qm.artifacts_score is not None
        assert 0 <= qm.artifacts_score <= 1
        assert qm.artifacts_score == pytest.approx(GOLDEN["basic_image"]["artifacts_score"], **TOL)

    def test_technical_score(self, qm):
        assert qm.technical_score is not None
        assert 0 <= qm.technical_score <= 100
        assert qm.technical_score == pytest.approx(GOLDEN["basic_image"]["technical_score"], **TOL)

    def test_gradient_detail(self, qm):
        assert qm.gradient_detail is not None
        assert 0 <= qm.gradient_detail <= 100
        assert qm.gradient_detail == pytest.approx(GOLDEN["basic_image"]["gradient_detail"], **TOL)


class TestBasicVideo:
    """BasicQualityModule on golden video — 8 fields."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.basic import BasicQualityModule

        return _run(BasicQualityModule, _vid_sample(golden_video)).quality_metrics

    def test_blur_score(self, qm):
        assert qm.blur_score is not None
        assert qm.blur_score == pytest.approx(GOLDEN["basic_video"]["blur_score"], **TOL)

    def test_brightness(self, qm):
        assert qm.brightness is not None
        assert qm.brightness == pytest.approx(GOLDEN["basic_video"]["brightness"], **TOL)

    def test_contrast(self, qm):
        assert qm.contrast is not None
        assert qm.contrast == pytest.approx(GOLDEN["basic_video"]["contrast"], **TOL)

    def test_saturation(self, qm):
        assert qm.saturation is not None
        assert qm.saturation == pytest.approx(GOLDEN["basic_video"]["saturation"], **TOL)

    def test_noise_score(self, qm):
        assert qm.noise_score is not None
        assert qm.noise_score == pytest.approx(GOLDEN["basic_video"]["noise_score"], **TOL)

    def test_artifacts_score(self, qm):
        assert qm.artifacts_score is not None
        assert qm.artifacts_score == pytest.approx(GOLDEN["basic_video"]["artifacts_score"], **TOL)

    def test_technical_score(self, qm):
        assert qm.technical_score is not None
        assert qm.technical_score == pytest.approx(GOLDEN["basic_video"]["technical_score"], **TOL)

    def test_gradient_detail(self, qm):
        assert qm.gradient_detail is not None
        assert qm.gradient_detail == pytest.approx(GOLDEN["basic_video"]["gradient_detail"], **TOL)


class TestMotion:
    """MotionModule on golden video."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.motion import MotionModule

        return _run(MotionModule, _vid_sample(golden_video)).quality_metrics

    def test_motion_score(self, qm):
        assert qm.motion_score is not None
        assert isinstance(qm.motion_score, float)
        assert qm.motion_score >= 0
        assert qm.motion_score == pytest.approx(GOLDEN["motion"]["motion_score"], **TOL)


class TestTISI:
    """TISIModule on golden video — spatial + temporal information."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.ti_si import TISIModule

        return _run(TISIModule, _vid_sample(golden_video)).quality_metrics

    def test_spatial_information(self, qm):
        assert qm.spatial_information is not None
        assert qm.spatial_information >= 0
        assert qm.spatial_information == pytest.approx(
            GOLDEN["ti_si_video"]["spatial_information"], **TOL
        )

    def test_temporal_information(self, qm):
        assert qm.temporal_information is not None
        assert qm.temporal_information >= 0
        assert qm.temporal_information == pytest.approx(
            GOLDEN["ti_si_video"]["temporal_information"], **TOL
        )


class TestCompressionArtifactsImage:
    """CompressionArtifactsModule on golden image."""

    @pytest.fixture(scope="class")
    def qm(self, golden_image):
        from ayase.modules.compression_artifacts import CompressionArtifactsModule

        return _run(CompressionArtifactsModule, _img_sample(golden_image)).quality_metrics

    def test_compression_artifacts(self, qm):
        assert qm.compression_artifacts is not None
        assert 0 <= qm.compression_artifacts <= 100
        assert qm.compression_artifacts == pytest.approx(
            GOLDEN["compression_image"]["compression_artifacts"], **TOL
        )


class TestCompressionArtifactsVideo:
    """CompressionArtifactsModule on golden video."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.compression_artifacts import CompressionArtifactsModule

        return _run(CompressionArtifactsModule, _vid_sample(golden_video)).quality_metrics

    def test_compression_artifacts(self, qm):
        assert qm.compression_artifacts is not None
        assert 0 <= qm.compression_artifacts <= 100
        assert qm.compression_artifacts == pytest.approx(
            GOLDEN["compression_video"]["compression_artifacts"], **TOL
        )


class TestFlicker:
    """FlickerDetectionModule on golden video."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.flicker_detection import FlickerDetectionModule

        return _run(FlickerDetectionModule, _vid_sample(golden_video)).quality_metrics

    def test_flicker_score(self, qm):
        assert qm.flicker_score is not None
        assert qm.flicker_score >= 0
        assert qm.flicker_score == pytest.approx(GOLDEN["flicker"]["flicker_score"], **TOL)


class TestJudderStutter:
    """JudderStutterModule on golden video."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.judder_stutter import JudderStutterModule

        return _run(JudderStutterModule, _vid_sample(golden_video)).quality_metrics

    def test_judder_score(self, qm):
        assert qm.judder_score is not None
        assert qm.judder_score >= 0
        assert qm.judder_score == pytest.approx(
            GOLDEN["judder_stutter"]["judder_score"], **TOL
        )

    def test_stutter_score(self, qm):
        assert qm.stutter_score is not None
        assert 0 <= qm.stutter_score <= 100
        assert qm.stutter_score == pytest.approx(
            GOLDEN["judder_stutter"]["stutter_score"], **TOL
        )


class TestCameraJitter:
    """CameraJitterModule on golden video."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.camera_jitter import CameraJitterModule

        return _run(CameraJitterModule, _vid_sample(golden_video)).quality_metrics

    def test_camera_jitter_score(self, qm):
        assert qm.camera_jitter_score is not None
        assert 0 <= qm.camera_jitter_score <= 1
        assert qm.camera_jitter_score == pytest.approx(
            GOLDEN["camera_jitter"]["camera_jitter_score"], **TOL
        )


class TestSceneComplexityImage:
    """SceneComplexityModule on golden image."""

    @pytest.fixture(scope="class")
    def qm(self, golden_image):
        from ayase.modules.scene_complexity import SceneComplexityModule

        return _run(SceneComplexityModule, _img_sample(golden_image)).quality_metrics

    def test_scene_complexity(self, qm):
        assert qm.scene_complexity is not None
        assert 0 <= qm.scene_complexity <= 100
        assert qm.scene_complexity == pytest.approx(
            GOLDEN["scene_complexity_image"]["scene_complexity"], **TOL
        )


class TestSceneComplexityVideo:
    """SceneComplexityModule on golden video."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.scene_complexity import SceneComplexityModule

        return _run(SceneComplexityModule, _vid_sample(golden_video)).quality_metrics

    def test_scene_complexity(self, qm):
        assert qm.scene_complexity is not None
        assert 0 <= qm.scene_complexity <= 100
        assert qm.scene_complexity == pytest.approx(
            GOLDEN["scene_complexity_video"]["scene_complexity"], **TOL
        )


class TestLetterbox:
    """LetterboxModule on golden image."""

    @pytest.fixture(scope="class")
    def qm(self, golden_image):
        from ayase.modules.letterbox import LetterboxModule

        return _run(LetterboxModule, _img_sample(golden_image)).quality_metrics

    def test_letterbox_ratio(self, qm):
        assert qm.letterbox_ratio is not None
        assert 0 <= qm.letterbox_ratio <= 1
        assert qm.letterbox_ratio == pytest.approx(
            GOLDEN["letterbox"]["letterbox_ratio"], **TOL
        )


class TestHDRMetadata:
    """HDRMetadataModule on golden video."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.hdr_metadata import HDRMetadataModule

        return _run(HDRMetadataModule, _vid_sample(golden_video)).quality_metrics

    def test_max_fall(self, qm):
        assert qm.max_fall is not None
        assert qm.max_fall >= 0
        assert qm.max_fall == pytest.approx(GOLDEN["hdr_metadata"]["max_fall"], **TOL)

    def test_max_cll(self, qm):
        assert qm.max_cll is not None
        assert qm.max_cll >= 0
        assert qm.max_cll == pytest.approx(GOLDEN["hdr_metadata"]["max_cll"], **TOL)


class TestSceneDetection:
    """SceneDetectionModule on golden video (heuristic, no TransNetV2)."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.scene_detection import SceneDetectionModule

        return _run(
            SceneDetectionModule, _vid_sample(golden_video),
            config={}).quality_metrics

    def test_scene_stability(self, qm):
        assert qm.scene_stability is not None
        assert 0 <= qm.scene_stability <= 1
        assert qm.scene_stability == pytest.approx(
            GOLDEN["scene_detection"]["scene_stability"], **TOL
        )

    def test_avg_scene_duration(self, qm):
        assert qm.avg_scene_duration is not None
        assert qm.avg_scene_duration > 0
        assert qm.avg_scene_duration == pytest.approx(
            GOLDEN["scene_detection"]["avg_scene_duration"], **TOL
        )


class TestDynamicsRange:
    """DynamicsRangeModule on golden video."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.dynamics_range import DynamicsRangeModule

        return _run(DynamicsRangeModule, _vid_sample(golden_video)).quality_metrics

    def test_dynamics_range(self, qm):
        assert qm.dynamics_range is not None
        assert qm.dynamics_range >= 0
        assert qm.dynamics_range == pytest.approx(
            GOLDEN["dynamics_range"]["dynamics_range"], **TOL
        )


class TestFlowCoherence:
    """FlowCoherenceModule on golden video."""

    @pytest.fixture(scope="class")
    def qm(self, golden_video):
        from ayase.modules.flow_coherence import FlowCoherenceModule

        return _run(FlowCoherenceModule, _vid_sample(golden_video)).quality_metrics

    def test_flow_coherence(self, qm):
        assert qm.flow_coherence is not None
        assert 0 <= qm.flow_coherence <= 1
        assert qm.flow_coherence == pytest.approx(
            GOLDEN["flow_coherence"]["flow_coherence"], **TOL
        )


class TestNaturalness:
    """NaturalnessModule on golden image (manual NSS, no pyiqa)."""

    @pytest.fixture(scope="class")
    def qm(self, golden_image):
        from ayase.modules.naturalness import NaturalnessModule

        return _run(
            NaturalnessModule, _img_sample(golden_image),
            config={"use_pyiqa": False}).quality_metrics

    def test_naturalness_score(self, qm):
        assert qm.naturalness_score is not None
        assert 0 <= qm.naturalness_score <= 1
        assert qm.naturalness_score == pytest.approx(
            GOLDEN["naturalness"]["naturalness_score"], **TOL
        )


class TestExposure:
    """ExposureModule on golden image — validation issues only."""

    @pytest.fixture(scope="class")
    def result(self, golden_image):
        from ayase.modules.exposure import ExposureModule

        return _run(ExposureModule, _img_sample(golden_image))

    def test_no_exposure_issues(self, result):
        assert len(result.validation_issues) == 0


# ═════════════════════════════════════════════════════════════════════════
# Tier 2: Reference-based modules (3 modules, golden+degraded pair)
# ═════════════════════════════════════════════════════════════════════════


class TestCIEDE2000:
    """CIEDE2000Module on golden vs degraded image pair."""

    @pytest.fixture(scope="class")
    def qm(self, golden_image, degraded_image):
        from ayase.modules.ciede2000 import CIEDE2000Module

        return _run(
            CIEDE2000Module, _ref_sample(degraded_image, golden_image)).quality_metrics

    def test_ciede2000(self, qm):
        assert qm.ciede2000 is not None
        assert qm.ciede2000 >= 0
        assert qm.ciede2000 == pytest.approx(GOLDEN["ciede2000"]["ciede2000"], **TOL)


class TestPSNRHVS:
    """PSNRHVSModule on golden vs degraded image pair (DCT approximation)."""

    @pytest.fixture(scope="class")
    def qm(self, golden_image, degraded_image):
        from ayase.modules.psnr_hvs import PSNRHVSModule

        return _run(
            PSNRHVSModule, _ref_sample(degraded_image, golden_image)).quality_metrics

    def test_psnr_hvs(self, qm):
        assert qm.psnr_hvs is not None
        assert qm.psnr_hvs > 0
        assert qm.psnr_hvs == pytest.approx(GOLDEN["psnr_hvs"]["psnr_hvs"], **TOL)


class TestPUMetrics:
    """PUMetricsModule on golden vs degraded image pair."""

    @pytest.fixture(scope="class")
    def qm(self, golden_image, degraded_image):
        from ayase.modules.pu_metrics import PUMetricsModule

        return _run(
            PUMetricsModule, _ref_sample(degraded_image, golden_image)).quality_metrics

    def test_pu_psnr(self, qm):
        assert qm.pu_psnr is not None
        assert qm.pu_psnr > 0
        assert qm.pu_psnr == pytest.approx(GOLDEN["pu_metrics"]["pu_psnr"], **TOL)

    def test_pu_ssim(self, qm):
        assert qm.pu_ssim is not None
        assert 0 <= qm.pu_ssim <= 1
        assert qm.pu_ssim == pytest.approx(GOLDEN["pu_metrics"]["pu_ssim"], **TOL)


# ═════════════════════════════════════════════════════════════════════════
# Tier 3: pyiqa modules (2 modules, skip if unavailable)
# ═════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_PYIQA, reason="pyiqa not installed")
class TestBRISQUE:
    """BRISQUEModule on golden image (requires pyiqa)."""

    @pytest.fixture(scope="class")
    def qm(self, golden_image):
        from ayase.modules.brisque import BRISQUEModule

        return _run(BRISQUEModule, _img_sample(golden_image)).quality_metrics

    def test_brisque(self, qm):
        assert qm.brisque is not None
        assert qm.brisque >= 0
        assert qm.brisque == pytest.approx(GOLDEN["brisque"]["brisque"], **TOL)


@pytest.mark.skipif(not HAS_PYIQA, reason="pyiqa not installed")
class TestNIQE:
    """NIQEModule on golden image (requires pyiqa)."""

    @pytest.fixture(scope="class")
    def qm(self, golden_image):
        from ayase.modules.niqe import NIQEModule

        return _run(NIQEModule, _img_sample(golden_image)).quality_metrics

    def test_niqe(self, qm):
        assert qm.niqe is not None
        assert qm.niqe >= 0
        assert qm.niqe == pytest.approx(GOLDEN["niqe"]["niqe"], **TOL)

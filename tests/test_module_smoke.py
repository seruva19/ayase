import importlib
import pkgutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import ModuleRegistry


@pytest.fixture
def synthetic_image(tmp_path: Path) -> Path:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = (255, 255, 255)
    path = tmp_path / "sample.png"
    cv2.imwrite(str(path), image)
    return path


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    path = tmp_path / "sample.mp4"
    fourcc = getattr(cv2, "VideoWriter_fourcc")
    writer = cv2.VideoWriter(str(path), fourcc(*"mp4v"), 10.0, (64, 64))
    for i in range(10):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :] = (i * 20 % 255, i * 10 % 255, i * 5 % 255)
        writer.write(frame)
    writer.release()
    return path


def _discover_module_names() -> list[str]:
    modules_pkg = importlib.import_module("ayase.modules")
    return [name for _, name, _ in pkgutil.iter_modules(modules_pkg.__path__)]


MODULE_NAMES = sorted(_discover_module_names())


def test_readiness_covers_all_modules() -> None:
    ModuleRegistry.discover_modules()
    readiness = ModuleRegistry.readiness_report()
    missing = [name for name in MODULE_NAMES if name not in readiness]
    assert not missing


@pytest.mark.parametrize("module_name", MODULE_NAMES)
def test_module_smoke(
    module_name: str,
    synthetic_image: Path,
    synthetic_video: Path,
    tmp_path: Path) -> None:
    ModuleRegistry.discover_modules()
    readiness = ModuleRegistry.readiness_report()
    info = readiness.get(module_name)
    assert info is not None

    module_cls = ModuleRegistry.get_module(module_name)
    if info.get("status") == "missing":
        assert module_cls is None
        return

    assert module_cls is not None
    assert module_cls.name == module_name
    assert isinstance(module_cls.description, str)
    assert isinstance(module_cls.default_config, dict)

    module = module_cls(config={"models_dir": str(tmp_path)})
    module.on_mount()

    image_sample = Sample(
        path=synthetic_image,
        is_video=False,
        quality_metrics=QualityMetrics())
    video_sample = Sample(
        path=synthetic_video,
        is_video=True,
        quality_metrics=QualityMetrics())
    module.process(image_sample)
    module.process(video_sample)

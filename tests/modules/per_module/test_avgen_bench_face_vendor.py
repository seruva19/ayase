"""Contracts for the source-vendored InsightFace 0.7.3 Face backend."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import site
import subprocess
import sys
import textwrap

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_ROOT = _REPO_ROOT / "src" / "ayase" / "vendor"
_PACKAGE_ROOT = _VENDOR_ROOT / "insightface"
_UPSTREAM_PACKAGE_TREE_SHA256 = (
    "53f67248f1c5a3b0812678bee8faff2c717c7727a154377761cef585863a715f"
)


def _package_tree_sha256() -> tuple[str, int]:
    rows = []
    for path in _PACKAGE_ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.name in {"LICENSE", "NOTICE"}
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
        ):
            continue
        relative = path.relative_to(_PACKAGE_ROOT).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{relative}\0{digest}")
    payload = "\n".join(sorted(rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), len(rows)


def test_vendored_insightface_tree_is_complete_and_pinned():
    digest, file_count = _package_tree_sha256()

    assert file_count == 60
    assert digest == _UPSTREAM_PACKAGE_TREE_SHA256
    assert (_PACKAGE_ROOT / "data" / "objects" / "meanshape_68.pkl").is_file()
    assert (_PACKAGE_ROOT / "model_zoo" / "arcface_onnx.py").is_file()
    assert (_PACKAGE_ROOT / "model_zoo" / "attribute.py").is_file()
    assert (_PACKAGE_ROOT / "model_zoo" / "landmark.py").is_file()
    assert (_PACKAGE_ROOT / "model_zoo" / "retinaface.py").is_file()
    assert (_PACKAGE_ROOT / "model_zoo" / "scrfd.py").is_file()
    assert (_PACKAGE_ROOT / "thirdparty" / "face3d" / "mesh_numpy" / "render.py").is_file()

    assert not (_PACKAGE_ROOT / "README.md").exists()
    assert not (_PACKAGE_ROOT / "setup.py").exists()
    assert not (_PACKAGE_ROOT / "pyproject.toml").exists()
    assert not (_PACKAGE_ROOT / "insightface.egg-info").exists()

    license_text = (_PACKAGE_ROOT / "LICENSE").read_text(encoding="utf-8")
    notice_text = (_PACKAGE_ROOT / "NOTICE").read_text(encoding="utf-8")
    assert license_text.startswith("MIT License\n")
    assert "Copyright (c) 2022 Jiankang Deng and Jia Guo" in license_text
    assert "InsightFace 0.7.3" in notice_text
    assert "f191f719612ebb37018f41936814500544cd0f86e6fcd676c023f354c668ddf7" in notice_text
    assert "non-commercial research" in notice_text


def test_vendored_insightface_imports_standalone_and_routes_all_buffalo_l_heads(tmp_path):
    required = ("cv2", "easydict", "numpy", "onnx", "onnxruntime", "skimage", "sklearn")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        pytest.skip(f"standalone InsightFace import dependencies unavailable: {missing}")

    script = textwrap.dedent(
        """
        import json
        from pathlib import Path
        from types import SimpleNamespace
        import sys

        import insightface
        from insightface.app import FaceAnalysis
        import insightface.app as app
        from insightface.model_zoo import model_zoo
        import onnx

        assert insightface.__version__ == "0.7.3"
        assert "ayase" not in sys.modules
        assert "MaskRenderer" not in app.__dict__
        assert "insightface.app.mask_renderer" not in sys.modules
        assert "insightface.thirdparty.face3d" not in sys.modules

        signatures = {
            "det_10g.onnx": ([1, 3, 640, 640], [[1, 1]] * 9),
            "w600k_r50.onnx": ([1, 3, 112, 112], [[1, 512]]),
            "1k3d68.onnx": ([1, 3, 192, 192], [[1, 3309]]),
            "2d106det.onnx": ([1, 3, 192, 192], [[1, 212]]),
            "genderage.onnx": ([1, 3, 96, 96], [[1, 3]]),
        }

        class ValueInfo:
            def __init__(self, name, shape):
                self.name = name
                self.shape = shape

        class FakeSession:
            def __init__(self, model_path, **kwargs):
                self.model_path = model_path
                self._providers = kwargs.get("providers", ["CPUExecutionProvider"])
                self._provider_options = kwargs.get("provider_options")
                input_shape, output_shapes = signatures[Path(model_path).name]
                self._inputs = [ValueInfo("input", input_shape)]
                self._outputs = [
                    ValueInfo(f"output_{index}", shape)
                    for index, shape in enumerate(output_shapes)
                ]
                self.prepared_providers = None

            def get_inputs(self):
                return self._inputs

            def get_outputs(self):
                return self._outputs

            def set_providers(self, providers):
                self.prepared_providers = providers

        model_zoo.PickableInferenceSession = FakeSession
        onnx.load = lambda _path: SimpleNamespace(graph=SimpleNamespace(node=[]))

        root = Path(__import__("os").environ["AYASE_TEST_INSIGHTFACE_ROOT"])
        model_dir = root / "models" / "buffalo_l"
        model_dir.mkdir(parents=True)
        for filename in signatures:
            (model_dir / filename).touch()

        analysis = FaceAnalysis(
            name="buffalo_l",
            root=str(root),
            providers=["CPUExecutionProvider"],
        )
        analysis.prepare(ctx_id=-1, det_size=(640, 640))

        expected = {
            "detection",
            "recognition",
            "landmark_2d_106",
            "landmark_3d_68",
            "genderage",
        }
        assert set(analysis.models) == expected
        assert all(
            model.session.prepared_providers == ["CPUExecutionProvider"]
            for model in analysis.models.values()
        )
        print(json.dumps({"heads": sorted(analysis.models), "source": insightface.__file__}))
        """
    )

    dependency_paths = [Path(path) for path in site.getsitepackages()]
    user_site = Path(site.getusersitepackages())
    if user_site.is_dir():
        dependency_paths.append(user_site)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_VENDOR_ROOT), *(str(path) for path in dependency_paths)]
    )
    env["AYASE_TEST_INSIGHTFACE_ROOT"] = str(tmp_path / "standalone insightface root")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result["heads"] == [
        "detection",
        "genderage",
        "landmark_2d_106",
        "landmark_3d_68",
        "recognition",
    ]
    assert Path(result["source"]).resolve() == (_PACKAGE_ROOT / "__init__.py").resolve()

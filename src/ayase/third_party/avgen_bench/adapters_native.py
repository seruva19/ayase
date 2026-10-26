"""Source-backed native adapters for the local AVGen-Bench worker.

This module deliberately imports only the Python standard library at import time.  Each
dimension is executed in its own managed environment, so the upstream evaluators' generic
module names (``dataset``, ``utils``, ``template``) cannot collide with another head.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional


SCHEMA_VERSION = 1
SOURCE_REVISION = "1049eabac472d479fe5feeb1ee202961f8e0982a"
DIMENSIONS = ("Vis", "Aud", "AV", "Lip", "Face", "Lo-Phy")

_PROTOCOL_IDENTITIES = {
    "Vis": "avgen-vis-qalign-video-1fps-joint-v1",
    "Aud": "avgen-aud-audiobox-pq-native-audio-v1",
    "AV": "avgen-av-synchformer-continuous-expectation-v1",
    "Lip": "avgen-lip-syncnet-s3fd-tracks-conf1-v1",
    "Face": "avgen-face-buffalo-count40-stability60-v1",
    "Lo-Phy": "avgen-lophy-videophy2-pc-32frames-parser0to5-v1",
}

_BACKEND_IDENTITIES = {
    "Vis": "QAlignVideoScorer/q-future-one-align",
    "Aud": "AudioboxAesthetics/PQ",
    "AV": "Synchformer/24-01-04T16-39-21",
    "Lip": "SyncNet-v2+S3FD",
    "Face": "InsightFace/buffalo_l",
    "Lo-Phy": "VideoPhy2/videophy_2_auto",
}


class AdapterError(RuntimeError):
    """An invalid request or unusable pinned-source adapter configuration."""


def _import_dependency(name: str) -> ModuleType:
    return importlib.import_module(name)


@contextlib.contextmanager
def _source_context(search_root: Path, cwd: Optional[Path] = None):
    """Temporarily expose one upstream evaluator root and its expected working directory."""

    old_path = list(sys.path)
    old_cwd = Path.cwd()
    sys.path.insert(0, str(search_root))
    if cwd is not None:
        cwd.mkdir(parents=True, exist_ok=True)
        os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path


def _load_source_module(name: str, path: Path, search_root: Path) -> ModuleType:
    """Load an upstream file under a private name without installing AVGen-Bench."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AdapterError(f"cannot load source module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with _source_context(search_root):
        spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _qalign_local_nested_assets(
    model_dir: Path, modeling_module: Optional[ModuleType] = None
):
    """Redirect QAlign's two hardcoded nested HF loads to the verified snapshot.

    The pinned source constructor looks up these names in its own module globals.  Replacing
    those two globals avoids mutating the process-wide Transformers classes and is restored
    immediately after model construction.
    """

    module = modeling_module or importlib.import_module("q_align.model.modeling_mplug_owl2")
    original_tokenizer = module.AutoTokenizer
    original_processor = module.CLIPImageProcessor
    resolved = str(model_dir.resolve())
    provenance: Dict[str, Any] = {
        "strategy": "scoped q_align.model.modeling_mplug_owl2 globals",
        "model_dir": resolved,
        "local_files_only": True,
        "redirect_counts": {"tokenizer": 0, "image_processor": 0},
    }

    class LocalTokenizer:
        @staticmethod
        def from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
            if str(pretrained_model_name_or_path) == "q-future/one-align":
                provenance["redirect_counts"]["tokenizer"] += 1
                kwargs = dict(kwargs)
                kwargs["local_files_only"] = True
                return original_tokenizer.from_pretrained(resolved, *args, **kwargs)
            return original_tokenizer.from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )

    class LocalImageProcessor:
        @staticmethod
        def from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
            if str(pretrained_model_name_or_path) == "q-future/one-align":
                provenance["redirect_counts"]["image_processor"] += 1
                kwargs = dict(kwargs)
                kwargs["local_files_only"] = True
                return original_processor.from_pretrained(resolved, *args, **kwargs)
            return original_processor.from_pretrained(
                pretrained_model_name_or_path, *args, **kwargs
            )

    module.AutoTokenizer = LocalTokenizer
    module.CLIPImageProcessor = LocalImageProcessor
    try:
        yield provenance
    finally:
        module.AutoTokenizer = original_tokenizer
        module.CLIPImageProcessor = original_processor


def _require_path(mapping: Mapping[str, Any], key: str, *, directory: bool = False) -> Path:
    raw = mapping.get(key)
    if not raw:
        raise AdapterError(f"missing required asset: {key}")
    path = Path(str(raw)).expanduser().resolve()
    if directory and not path.is_dir():
        raise AdapterError(f"asset directory not found ({key}): {path}")
    if not directory and not path.is_file():
        raise AdapterError(f"asset file not found ({key}): {path}")
    return path


def _finite(value: Any, label: str = "score") -> float:
    if isinstance(value, bool):
        raise ValueError(f"boolean is not a numeric {label}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {label}: {result!r}")
    return result


def _json_safe(value: Any) -> Any:
    """Convert native evidence to strict JSON values and reject non-finite floats."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    scalar = getattr(value, "item", None)
    if callable(scalar):
        try:
            return _json_safe(scalar())
        except Exception:
            pass
    return str(value)


def _bounded_error(exc: BaseException, limit: int = 1200) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_digest() -> str:
    return _sha256_file(Path(__file__).resolve())


def _record_path(record: Mapping[str, Any]) -> Path:
    raw = record.get("staged_video_path") or record.get("video_path")
    return Path(str(raw)).expanduser().resolve() if raw else Path()


def _base_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "id": str(record.get("id", "")),
        "status": "failed",
        "eligible": True,
        "raw_details": {},
    }


def _source_summary(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    values = [float(row["raw_score"]) for row in rows if row.get("status") == "scored"]
    return {
        "count_scored": len(values),
        "raw_sum": float(sum(values)),
        "raw_mean": (float(sum(values) / len(values)) if values else None),
    }


def _counts(records: Iterable[Mapping[str, Any]]) -> Dict[str, int]:
    rows = list(records)
    out: Dict[str, int] = {"requested": len(rows)}
    out["located"] = sum(row.get("status") != "missing_video" for row in rows)
    out["eligibility_known_true"] = sum(row.get("eligible") is True for row in rows)
    out["eligibility_known_false"] = sum(row.get("eligible") is False for row in rows)
    out["eligibility_unknown"] = sum(row.get("eligible") is None for row in rows)
    out["scored"] = sum(row.get("status") == "scored" for row in rows)
    out["failed"] = sum(row.get("status") == "failed" for row in rows)
    for status in (
        "ineligible",
        "missing_prompt",
        "missing_video",
        "missing_face_label",
        "no_face_detection",
    ):
        out[status] = sum(row.get("status") == status for row in rows)
    out["summary_eligible_numeric_count"] = out["scored"]
    return out


def _prepare_records(request: Mapping[str, Any]) -> tuple[List[Mapping[str, Any]], Dict[str, Dict[str, Any]]]:
    records = request.get("records")
    if not isinstance(records, list):
        raise AdapterError("request.records must be a list")
    runnable: List[Mapping[str, Any]] = []
    results: Dict[str, Dict[str, Any]] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise AdapterError(f"record {index} must be an object")
        record_id = str(record.get("id", ""))
        if not record_id or record_id in results:
            raise AdapterError(f"record {index} has missing or duplicate id")
        result = _base_record(record)
        results[record_id] = result
        path = _record_path(record)
        if not path.is_file():
            result.update(status="missing_video", eligible=True, error=f"video not found: {path}")
        else:
            runnable.append(record)
    return runnable, results


def _score_result(
    record_id: str,
    score: Any,
    details: Mapping[str, Any],
    *,
    evidence_path: Optional[Path] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "id": record_id,
        "status": "scored",
        "raw_score": _finite(score),
        "raw_details": dict(details),
        "eligible": True,
    }
    if evidence_path is not None:
        result["evidence_path"] = str(evidence_path.resolve())
    return result


def _run_vis(
    request: Mapping[str, Any], records: List[Mapping[str, Any]], results: MutableMapping[str, Dict[str, Any]]
) -> Dict[str, Any]:
    source_root = Path(str(request["source_dir"])).resolve() / "eval" / "Q-Align"
    source = _load_source_module(
        "_ayase_avgen_qalign_scorer",
        source_root / "q_align" / "evaluate" / "scorer.py",
        source_root,
    )
    model_dir = _require_path(request["assets"], "onealign_model_dir", directory=True)
    device = str(request.get("config", {}).get("device", "cuda"))
    with _qalign_local_nested_assets(model_dir) as nested_asset_provenance:
        scorer = source.QAlignVideoScorer(pretrained=str(model_dir), device=device)
    expected_redirects = {"tokenizer": 1, "image_processor": 1}
    if nested_asset_provenance["redirect_counts"] != expected_redirects:
        raise AdapterError(
            "QAlign constructor did not perform the expected local nested asset loads: "
            f"{nested_asset_provenance['redirect_counts']!r}"
        )

    for record in records:
        record_id = str(record["id"])
        try:
            frames = source.load_video(str(_record_path(record)))
            if not frames:
                raise ValueError("QAlign load_video returned no 1 fps frames")
            # The list of all sampled frames is one video item in one video-quality call.
            score = scorer([frames]).tolist()[0]
            results[record_id] = _score_result(
                record_id,
                score,
                {
                    "frames_sampled": len(frames),
                    "sampling": "source load_video: int(fps*i), 1fps, floor(duration)",
                    "model_call": "QAlignVideoScorer([all_sampled_frames])",
                    "nested_asset_redirect": nested_asset_provenance,
                },
            )
        except Exception as exc:
            results[record_id].update(error=_bounded_error(exc))
    summary = _source_summary(results.values())
    summary["nested_asset_redirect"] = nested_asset_provenance
    return summary


def _run_aud(
    request: Mapping[str, Any], records: List[Mapping[str, Any]], results: MutableMapping[str, Dict[str, Any]]
) -> Dict[str, Any]:
    source_root = Path(str(request["source_dir"])).resolve() / "eval" / "audiobox-aesthetics"
    source = _load_source_module(
        "_ayase_avgen_audiobox_batch", source_root / "batch_eval.py", source_root / "src"
    )
    checkpoint = _require_path(request["assets"], "audiobox_checkpoint")
    predictor = source.initialize_predictor(ckpt=str(checkpoint))

    metric_values: Dict[str, List[float]] = {key: [] for key in source.METRICS}
    for record in records:
        record_id = str(record["id"])
        try:
            metrics = source.score_one_video_audio(predictor, _record_path(record))
            clean = {key: _finite(metrics[key], key) for key in source.METRICS}
            for key, value in clean.items():
                metric_values[key].append(value)
            results[record_id] = _score_result(
                record_id,
                clean["PQ"],
                {
                    "metrics": clean,
                    "selected_metric": "PQ",
                    "audio_decode": "torchaudio.load(mp4), original sample rate passed to predictor",
                },
            )
        except Exception as exc:
            results[record_id].update(error=_bounded_error(exc))

    summary = _source_summary(results.values())
    summary["mean_metrics"] = {
        key: (source.safe_mean(values) if values else None) for key, values in metric_values.items()
    }
    return summary


def _run_av(
    request: Mapping[str, Any], records: List[Mapping[str, Any]], results: MutableMapping[str, Dict[str, Any]]
) -> Dict[str, Any]:
    source_root = Path(str(request["source_dir"])).resolve() / "eval" / "Syncformer"
    source = _load_source_module(
        "_ayase_avgen_synchformer_batch", source_root / "batch_eval.py", source_root
    )
    checkpoint = _require_path(request["assets"], "synchformer_checkpoint")
    config_path = _require_path(request["assets"], "synchformer_config")
    config = request.get("config", {})
    device_name = str(config.get("device", "cuda:0"))
    ffmpeg_path = config.get("ffmpeg_path")
    if ffmpeg_path:
        ffmpeg = Path(str(ffmpeg_path)).expanduser().resolve()
        if not ffmpeg.is_file():
            raise AdapterError(f"ffmpeg_path not found: {ffmpeg}")
        source.which_ffmpeg = lambda: str(ffmpeg)

    work_root = Path(str(request["work_dir"])).resolve() / "AV"
    torch = source.torch
    with _source_context(source_root, work_root / "setup"):
        cfg = source.patch_config(source.OmegaConf.load(str(config_path)))
        max_off_sec = cfg.data.max_off_sec
        num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
        grid = source.make_class_grid(-max_off_sec, max_off_sec, num_cls)
        device = torch.device(device_name)
        _, model = source.get_model(cfg, device)
        ckpt = torch.load(
            str(checkpoint), map_location=torch.device("cpu"), weights_only=False
        )
        model.load_state_dict(ckpt["model"])
        model.eval()
        transforms_test = source.get_transforms(cfg, ["test"])["test"]

    no_reencode = bool(config.get("no_reencode", False))
    v_start = float(config.get("v_start_i_sec", 0.0))
    continuous_values: List[float] = []
    argmax_values: List[float] = []
    for record in records:
        record_id = str(record["id"])
        try:
            record_work = work_root / hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16]
            with _source_context(source_root, record_work):
                used_path = str(_record_path(record))
                if not no_reencode:
                    used_path = source.maybe_reencode_if_needed(
                        used_path, vfps=25, afps=16000, in_size=256
                    )
                pred_cont, pred_argmax, pred_idx = source.predict_offset_for_video(
                    vid_path=used_path,
                    device=device,
                    cfg=cfg,
                    model=model,
                    grid=grid,
                    transforms_test=transforms_test,
                    v_start_i_sec=v_start,
                )
            cont = _finite(pred_cont, "continuous offset")
            argmax = _finite(pred_argmax, "argmax offset")
            continuous_values.append(abs(cont))
            argmax_values.append(abs(argmax))
            results[record_id] = _score_result(
                record_id,
                abs(cont),
                {
                    "pred_offset_sec_cont": cont,
                    "pred_offset_sec_argmax": argmax,
                    "pred_class_idx": int(pred_idx),
                    "abs_pred_offset_sec_cont": abs(cont),
                    "abs_pred_offset_sec_argmax": abs(argmax),
                    "used_path": used_path,
                    "selected_observation": "abs softmax expectation over source offset grid",
                },
            )
        except Exception as exc:
            results[record_id].update(error=_bounded_error(exc))

    return {
        "count_scored": len(continuous_values),
        "mean_abs_offset_cont_sec": (
            source.safe_mean(continuous_values) if continuous_values else None
        ),
        "mean_abs_offset_argmax_sec": (
            source.safe_mean(argmax_values) if argmax_values else None
        ),
    }


def _replace_exact(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise AdapterError(f"SyncNet overlay patch {label!r} expected once, found {count}")
    return text.replace(old, new)


def _build_syncnet_overlay(source_root: Path, work_dir: Path) -> Path:
    """Copy pinned SyncNet source and apply path/shell-only plumbing repairs."""

    overlay = work_dir / "syncnet_source_overlay"
    marker = overlay / ".ayase-overlay-v1"
    if marker.is_file():
        return overlay
    if overlay.exists():
        shutil.rmtree(overlay)
    shutil.copytree(
        source_root,
        overlay,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "data"),
    )

    inference_path = overlay / "inference.py"
    text = inference_path.read_text(encoding="utf-8")
    text = _replace_exact(
        text,
        "import sys, os, argparse, glob, subprocess, cv2",
        "import sys, os, argparse, glob, subprocess, cv2\n"
        "FFMPEG = os.environ['AYASE_AVGEN_FFMPEG']\n"
        "SYNCNET_CHECKPOINT = os.environ['AYASE_AVGEN_SYNCNET_CHECKPOINT']",
        "environment paths",
    )
    text = _replace_exact(
        text,
        "    subprocess.call(f\"ffmpeg -y -i \\\"{os.path.join(opt.avi_dir,opt.reference,'audio.wav')}\\\" -ss {audiostart:.3f} -to {audioend:.3f} \\\"{audiotmp}\\\"\", shell=True, stdout=None, stderr=subprocess.DEVNULL)\n"
        "    subprocess.call(f\"ffmpeg -y -i \\\"{cropfile}t.avi\\\" -i \\\"{audiotmp}\\\" -c:v copy -c:a copy \\\"{cropfile}.avi\\\"\", shell=True, stdout=None, stderr=subprocess.DEVNULL)",
        "    subprocess.run([FFMPEG, '-y', '-i', os.path.join(opt.avi_dir,opt.reference,'audio.wav'), '-ss', f'{audiostart:.3f}', '-to', f'{audioend:.3f}', audiotmp], check=True, stdout=None, stderr=subprocess.DEVNULL)\n"
        "    subprocess.run([FFMPEG, '-y', '-i', cropfile+'t.avi', '-i', audiotmp, '-c:v', 'copy', '-c:a', 'copy', cropfile+'.avi'], check=True, stdout=None, stderr=subprocess.DEVNULL)",
        "crop ffmpeg",
    )
    text = _replace_exact(
        text,
        "    if not os.path.exists(\"data/syncnet_v2.model\"):\n"
        "        print(\"Error: model file not found: data/syncnet_v2.model\")",
        "    if not os.path.exists(SYNCNET_CHECKPOINT):\n"
        "        print(f\"Error: model file not found: {SYNCNET_CHECKPOINT}\")",
        "checkpoint check",
    )
    text = _replace_exact(
        text,
        "    cmd_base = f\"ffmpeg -y -i \\\"{opt.videofile}\\\" -qscale:v 2\"\n"
        "    subprocess.call(f\"{cmd_base} -async 1 -r 25 \\\"{os.path.join(opt.avi_dir,opt.reference,'video.avi')}\\\"\", shell=True, stderr=subprocess.DEVNULL)\n"
        "    subprocess.call(f\"ffmpeg -y -i \\\"{os.path.join(opt.avi_dir,opt.reference,'video.avi')}\\\" -qscale:v 2 -threads 1 -f image2 \\\"{os.path.join(opt.frames_dir,opt.reference,'%06d.jpg')}\\\"\", shell=True, stderr=subprocess.DEVNULL)\n"
        "    subprocess.call(f\"ffmpeg -y -i \\\"{os.path.join(opt.avi_dir,opt.reference,'video.avi')}\\\" -ac 1 -vn -acodec pcm_s16le -ar 16000 \\\"{os.path.join(opt.avi_dir,opt.reference,'audio.wav')}\\\"\", shell=True, stderr=subprocess.DEVNULL)",
        "    source_video = os.path.join(opt.avi_dir,opt.reference,'video.avi')\n"
        "    subprocess.run([FFMPEG, '-y', '-i', opt.videofile, '-qscale:v', '2', '-async', '1', '-r', '25', source_video], check=True, stderr=subprocess.DEVNULL)\n"
        "    subprocess.run([FFMPEG, '-y', '-i', source_video, '-qscale:v', '2', '-threads', '1', '-f', 'image2', os.path.join(opt.frames_dir,opt.reference,'%06d.jpg')], check=True, stderr=subprocess.DEVNULL)\n"
        "    subprocess.run([FFMPEG, '-y', '-i', source_video, '-ac', '1', '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', os.path.join(opt.avi_dir,opt.reference,'audio.wav')], check=True, stderr=subprocess.DEVNULL)",
        "preprocess ffmpeg",
    )
    text = _replace_exact(
        text,
        '    s.loadParameters("data/syncnet_v2.model")',
        "    s.loadParameters(SYNCNET_CHECKPOINT)",
        "checkpoint load",
    )
    inference_path.write_text(text, encoding="utf-8")

    detector_path = overlay / "detectors" / "s3fd" / "__init__.py"
    text = detector_path.read_text(encoding="utf-8")
    text = _replace_exact(
        text,
        "import os",
        "import os",
        "detector os import",
    ) if "import os" in text else "import os\n" + text
    text = _replace_exact(
        text,
        "PATH_WEIGHT = './detectors/s3fd/weights/sfd_face.pth'",
        "PATH_WEIGHT = os.environ['AYASE_AVGEN_S3FD_CHECKPOINT']",
        "S3FD checkpoint",
    )
    detector_path.write_text(text, encoding="utf-8")

    instance_path = overlay / "SyncNetInstance.py"
    text = instance_path.read_text(encoding="utf-8")
    text = text.replace('["ffmpeg",', '[os.environ["AYASE_AVGEN_FFMPEG"],')
    if text.count("AYASE_AVGEN_FFMPEG") != 2:
        raise AdapterError("SyncNet overlay expected two list-style FFmpeg calls")
    instance_path.write_text(text, encoding="utf-8")
    marker.write_text(SOURCE_REVISION + "\n", encoding="ascii")
    return overlay


def _run_lip(
    request: Mapping[str, Any], records: List[Mapping[str, Any]], results: MutableMapping[str, Dict[str, Any]]
) -> Dict[str, Any]:
    source_root = Path(str(request["source_dir"])).resolve() / "eval" / "syncnet_python"
    work_root = Path(str(request["work_dir"])).resolve() / "Lip"
    overlay = _build_syncnet_overlay(source_root, work_root)
    source = _load_source_module(
        "_ayase_avgen_syncnet_batch", source_root / "batch_eval.py", source_root
    )
    syncnet_checkpoint = _require_path(request["assets"], "syncnet_checkpoint")
    s3fd_checkpoint = _require_path(request["assets"], "s3fd_checkpoint")
    ffmpeg = _require_path(request.get("config", {}), "ffmpeg_path")
    config = request.get("config", {})
    track_logs: Dict[str, Path] = {}

    def owned_run_inference(
        inference_py: Path,
        videofile: Path,
        data_dir: Path,
        batch_size: int,
        vshift: int,
        frame_rate: int,
        facedet_scale: float,
        crop_scale: float,
        min_track: int,
        num_failed_det: int,
        min_face_size: int,
        timeout_sec: int = 3600,
    ) -> tuple[int, str]:
        del inference_py
        command = [
            sys.executable,
            str(overlay / "inference.py"),
            "--videofile",
            str(videofile),
            "--data_dir",
            str(data_dir),
            "--batch_size",
            str(batch_size),
            "--vshift",
            str(vshift),
            "--frame_rate",
            str(frame_rate),
            "--facedet_scale",
            str(facedet_scale),
            "--crop_scale",
            str(crop_scale),
            "--min_track",
            str(min_track),
            "--num_failed_det",
            str(num_failed_det),
            "--min_face_size",
            str(min_face_size),
        ]
        env = dict(os.environ)
        env.update(
            AYASE_AVGEN_FFMPEG=str(ffmpeg),
            AYASE_AVGEN_SYNCNET_CHECKPOINT=str(syncnet_checkpoint),
            AYASE_AVGEN_S3FD_CHECKPOINT=str(s3fd_checkpoint),
        )
        completed = subprocess.run(
            command,
            cwd=str(overlay),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_sec,
            check=False,
        )
        evidence = data_dir / "syncnet_output.log"
        evidence.parent.mkdir(parents=True, exist_ok=True)
        evidence.write_text(completed.stdout, encoding="utf-8")
        track_logs[str(videofile.resolve())] = evidence
        return completed.returncode, completed.stdout

    source.run_inference = owned_run_inference
    passing_offsets: List[float] = []
    n_tracks_total = 0
    n_videos_ran = 0
    n_videos_with_tracks = 0
    n_videos_with_pass_tracks = 0
    failures = 0
    for record in records:
        record_id = str(record["id"])
        path = _record_path(record)
        outcome = source.eval_one_video(
            str(record.get("category", "")),
            path,
            overlay / "inference.py",
            work_root / "runs",
            int(config.get("batch_size", 20)),
            int(config.get("vshift", 15)),
            int(config.get("frame_rate", 25)),
            float(config.get("facedet_scale", 0.25)),
            float(config.get("crop_scale", 0.40)),
            int(config.get("min_track", 30)),
            int(config.get("num_failed_det", 25)),
            int(config.get("min_face_size", 100)),
            1.0,
            int(config.get("timeout_sec", 3600)),
        )
        if outcome.get("ran"):
            n_videos_ran += 1
        status = outcome.get("status")
        if status == "ok":
            total = int(outcome.get("tracks_total", 0))
            passing = [
                {
                    "offset_frames": int(track["offset"]),
                    "confidence": _finite(track["conf"], "track confidence"),
                    "distance": _finite(track["min_dist"], "track distance"),
                }
                for track in outcome.get("pass_tracks", [])
            ]
            n_tracks_total += total
            n_videos_with_tracks += 1
            evidence = track_logs.get(str(path.resolve()))
            if passing:
                n_videos_with_pass_tracks += 1
                offsets = [abs(float(track["offset_frames"])) for track in passing]
                passing_offsets.extend(offsets)
                results[record_id] = _score_result(
                    record_id,
                    sum(offsets) / len(offsets),
                    {
                        "passing_tracks": passing,
                        "tracks_total": total,
                        "confidence_threshold": 1.0,
                        "pooling": "dataset aggregate flattens every passing track",
                    },
                    evidence_path=evidence,
                )
            else:
                results[record_id].update(
                    status="ineligible",
                    eligible=False,
                    raw_details={
                        "passing_tracks": [],
                        "tracks_total": total,
                        "confidence_threshold": 1.0,
                    },
                )
                if evidence is not None:
                    results[record_id]["evidence_path"] = str(evidence.resolve())
        elif status == "no_tracks":
            results[record_id].update(
                status="no_face_detection",
                eligible=None,
                raw_details={"passing_tracks": [], "tracks_total": 0, "confidence_threshold": 1.0},
            )
        else:
            failures += 1
            results[record_id].update(error=str(outcome.get("error") or status or "SyncNet failed"))

    return {
        "conf_th": 1.0,
        "n_videos_total": len(records),
        "n_videos_ran": n_videos_ran,
        "n_videos_with_tracks": n_videos_with_tracks,
        "n_videos_with_pass_tracks": n_videos_with_pass_tracks,
        "n_tracks_total": n_tracks_total,
        "n_tracks_pass_conf": len(passing_offsets),
        "failures": failures,
        "mean_abs_offset_frames": (
            source.safe_mean(passing_offsets) if passing_offsets else None
        ),
    }


def _face_root(buffalo_dir: Path) -> Path:
    if (buffalo_dir / "models" / "buffalo_l").is_dir():
        return buffalo_dir
    if buffalo_dir.name == "buffalo_l" and buffalo_dir.parent.name == "models":
        return buffalo_dir.parent.parent
    raise AdapterError(
        "buffalo_dir must be an InsightFace root containing models/buffalo_l "
        "or the models/buffalo_l directory itself"
    )


def _prepare_face_root(buffalo_dir: Path, work_dir: Path) -> Path:
    """Return an InsightFace root, staging a direct buffalo_l snapshot when necessary."""

    try:
        return _face_root(buffalo_dir)
    except AdapterError:
        files = [path for path in buffalo_dir.rglob("*") if path.is_file()]
        if not files or not any(path.suffix.lower() == ".onnx" for path in files):
            raise
        token = hashlib.sha256(str(buffalo_dir).encode("utf-8")).hexdigest()[:16]
        root = work_dir / "insightface_roots" / token
        destination = root / "models" / "buffalo_l"
        for source_file in files:
            relative = source_file.relative_to(buffalo_dir)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == source_file.stat().st_size:
                continue
            if target.exists():
                raise AdapterError(f"stale buffalo_l staged asset: {target}")
            try:
                os.link(source_file, target)
            except OSError:
                shutil.copy2(source_file, target)
        return root


def _expected_minmax(record: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    expected = record.get("expected_faces")
    if not isinstance(expected, Mapping):
        return None
    primary = expected.get("primary_faces", expected)
    if not isinstance(primary, Mapping) or "max_ids" not in primary:
        return None
    minimum = int(primary.get("min_ids", 0) or 0)
    maximum = int(primary.get("max_ids", minimum) or minimum)
    return minimum, max(minimum, maximum)


def _run_face(
    request: Mapping[str, Any], records: List[Mapping[str, Any]], results: MutableMapping[str, Dict[str, Any]]
) -> Dict[str, Any]:
    source_root = Path(str(request["source_dir"])).resolve() / "eval" / "facial_consistency"
    source = _load_source_module(
        "_ayase_avgen_face_batch", source_root / "batch_eval.py", source_root
    )
    buffalo_dir = _require_path(request["assets"], "buffalo_dir", directory=True)
    config = request.get("config", {})
    insightface = _import_dependency("insightface.app")
    analyzer = source.MultiFaceConsistencyV2.__new__(source.MultiFaceConsistencyV2)
    analyzer.app = insightface.FaceAnalysis(
        name="buffalo_l",
        root=str(
            _prepare_face_root(
                buffalo_dir, Path(str(request["work_dir"])).resolve() / "Face"
            )
        ),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    det_size_raw = config.get("det_size", (640, 640))
    if isinstance(det_size_raw, str):
        det_size = tuple(int(item) for item in det_size_raw.split(","))
    else:
        det_size = tuple(int(item) for item in det_size_raw)
    analyzer.app.prepare(ctx_id=int(config.get("ctx_id", 0)), det_size=det_size)

    cfg = {
        "sample_rate": int(config.get("sample_rate", 1)),
        "min_face_px": int(config.get("min_face_px", 30)),
        "min_det_score": float(config.get("min_det_score", 0.6)),
        "min_rel_area": float(config.get("min_rel_area", 0.0005)),
        "max_gap_frames": int(config.get("max_gap_frames", 3)),
        "min_iou": float(config.get("min_iou", 0.1)),
        "min_cos": float(config.get("min_cos", 0.35)),
        "min_tracklet_len": int(config.get("min_tracklet_len", 3)),
        "dbscan_eps": float(config.get("dbscan_eps", 0.65)),
        "dbscan_min_samples": int(config.get("dbscan_min_samples", 1)),
        "primary_ratio_thresh": float(config.get("primary_ratio_thresh", 0.10)),
        "min_primary_cluster_dets": int(config.get("min_primary_cluster_dets", 30)),
    }

    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    key_to_id: Dict[tuple[str, str], str] = {}
    label_count = sum(_expected_minmax(item) is not None for item in request["records"])
    positive_count = sum(
        expected is not None and expected[1] > 0
        for expected in (_expected_minmax(item) for item in request["records"])
    )
    for record in records:
        record_id = str(record["id"])
        expected = _expected_minmax(record)
        if expected is None:
            results[record_id].update(status="missing_face_label", eligible=None)
            continue
        minimum, maximum = expected
        if maximum == 0:
            results[record_id].update(
                status="ineligible",
                eligible=False,
                raw_details={
                    "expected_primary_faces": {"min_ids": minimum, "max_ids": maximum},
                    "reason": "source skips labels with max_ids=0",
                },
            )
            continue
        category = str(record.get("category", ""))
        safe_content = str(record.get("safe_content", ""))
        if not category or not safe_content:
            results[record_id].update(error="Face requires category and safe_content")
            continue
        key_to_id[(category, safe_content)] = record_id
        grouped.setdefault(category, []).append(
            {
                "content": safe_content,
                "prompt": str(record.get("prompt_text", "")),
                "primary_faces": {"min_ids": minimum, "max_ids": maximum},
            }
        )

    summaries = []
    videos_root = str(Path(str(request["videos_dir"])).resolve())
    for category, label_items in grouped.items():
        per_video, summary = source.evaluate_category(
            category=category,
            label_items=label_items,
            root_videos=videos_root,
            analyzer=analyzer,
            cfg=cfg,
        )
        summaries.append(summary)
        for native in per_video:
            record_id = key_to_id[(category, str(native.get("content", "")))]
            native_status = native.get("status")
            if native_status == "ok":
                results[record_id] = _score_result(
                    record_id,
                    native["final_score"],
                    {
                        "expected_primary_faces": native["expected_primary_faces"],
                        "breakdown": native["breakdown"],
                        "observed": native["observed"],
                        "metrics": native["metrics"],
                        "dbscan": native["dbscan"],
                        "score_formula": "40*count_range + 60*primary_stability",
                    },
                )
            elif native_status == "missing_video":
                results[record_id].update(status="missing_video", eligible=True)
            elif native_status in {"no_faces_detected", "no_tracklets", "no_clusters"}:
                results[record_id].update(
                    status="no_face_detection",
                    eligible=True,
                    raw_details={
                        "native_status": native_status,
                        "expected_primary_faces": native.get("expected_primary_faces"),
                    },
                )
            else:
                results[record_id].update(error=str(native.get("error") or native_status))

    summary = _source_summary(results.values())
    summary.update(
        {
            "count_annotations": label_count,
            "count_label_eligible": positive_count,
            "canonical_expected_annotations": 220,
            "canonical_expected_label_eligible": 103,
            "summaries_by_category": summaries,
        }
    )
    return summary


def _videophy_parse_provenance(output: str) -> Dict[str, Any]:
    normalized = output.lower().strip()
    for token, value in (
        ("zero", 0),
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("0", 0),
        ("1", 1),
        ("2", 2),
        ("3", 3),
        ("4", 4),
        ("5", 5),
    ):
        if token in normalized:
            return {"branch": "substring", "matched": token, "parsed_score": value}
    digits = "".join(character for character in normalized if character.isdigit())
    if digits:
        try:
            value = int(digits)
            if value in {0, 1, 2, 3, 4, 5}:
                return {"branch": "digits", "matched": digits, "parsed_score": value}
        except ValueError:
            pass
    return {"branch": "fallback_zero", "matched": None, "parsed_score": 0}


def _run_lophy(
    request: Mapping[str, Any], records: List[Mapping[str, Any]], results: MutableMapping[str, Dict[str, Any]]
) -> Dict[str, Any]:
    source_root = (
        Path(str(request["source_dir"])).resolve() / "eval" / "videophy" / "VIDEOPHY2"
    )
    source = _load_source_module(
        "_ayase_avgen_videophy2_batch", source_root / "batch_eval.py", source_root
    )
    checkpoint = _require_path(request["assets"], "videophy_model_dir", directory=True)
    lora_raw = request.get("assets", {}).get("videophy_lora_checkpoint")
    lora = str(_require_path(request["assets"], "videophy_lora_checkpoint")) if lora_raw else None
    device = str(request.get("config", {}).get("device", "cuda"))
    model, processor, tokenizer = source.build_model_and_processor(
        checkpoint=str(checkpoint),
        lora_checkpoint=lora,
        device=device,
        dtype=source.torch.bfloat16,
    )
    prompt = source.build_prompt(task="pc")
    values: List[float] = []
    for record in records:
        record_id = str(record["id"])
        try:
            score, raw_output = source.score_video(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                videopath=str(_record_path(record)),
                prompt=prompt,
                num_frames=32,
            )
            # Re-run the actual source parser explicitly; score_video uses the same function internally.
            parsed = int(source.parse_score_from_output(raw_output))
            if int(score) != parsed:
                raise ValueError(f"source score/parser mismatch: {score!r} != {parsed!r}")
            provenance = _videophy_parse_provenance(raw_output)
            if provenance["parsed_score"] != parsed:
                raise ValueError("owned parser provenance does not match source parser")
            clean = _finite(parsed)
            values.append(clean)
            results[record_id] = _score_result(
                record_id,
                clean,
                {
                    "raw_output": raw_output,
                    "parser": provenance,
                    "task": "pc",
                    "num_frames": 32,
                    "prompt": prompt,
                    "generation_kwargs": dict(source.generate_kwargs),
                },
            )
        except Exception as exc:
            results[record_id].update(error=_bounded_error(exc))
    return {
        "count_scored": len(values),
        "mean_score": (source.safe_mean(values) if values else None),
        "task": "pc",
        "num_frames": 32,
        "generation_kwargs": dict(source.generate_kwargs),
    }


_RUNNERS: Dict[
    str,
    Callable[
        [Mapping[str, Any], List[Mapping[str, Any]], MutableMapping[str, Dict[str, Any]]],
        Dict[str, Any],
    ],
] = {
    "Vis": _run_vis,
    "Aud": _run_aud,
    "AV": _run_av,
    "Lip": _run_lip,
    "Face": _run_face,
    "Lo-Phy": _run_lophy,
}


def run_dimension(request: dict) -> dict:
    """Run one AVGen-Bench dimension and return the schema-1 native result object."""

    if not isinstance(request, dict):
        raise AdapterError("request must be a dict")
    if int(request.get("schema", -1)) != SCHEMA_VERSION:
        raise AdapterError(f"unsupported request schema: {request.get('schema')!r}")
    dimension = request.get("dimension")
    if dimension not in _RUNNERS:
        raise AdapterError(f"unsupported native dimension: {dimension!r}")
    source_dir = Path(str(request.get("source_dir", ""))).expanduser().resolve()
    if not Path(str(request.get("source_dir", ""))).expanduser().is_absolute():
        raise AdapterError("request.source_dir must be absolute")
    if not source_dir.is_dir():
        raise AdapterError(f"source_dir not found: {source_dir}")
    requested_revision = request.get("source_revision")
    if requested_revision not in (None, SOURCE_REVISION):
        raise AdapterError(
            f"source revision mismatch: expected {SOURCE_REVISION}, got {requested_revision}"
        )
    for key in ("assets", "config"):
        if not isinstance(request.get(key, {}), Mapping):
            raise AdapterError(f"request.{key} must be an object")

    runnable, by_id = _prepare_records(request)
    if dimension == "Face":
        face_runnable = []
        for record in runnable:
            record_id = str(record["id"])
            expected = _expected_minmax(record)
            if expected is None:
                by_id[record_id].update(status="missing_face_label", eligible=None, error=None)
            elif expected[1] == 0:
                by_id[record_id].update(
                    status="ineligible",
                    eligible=False,
                    raw_details={
                        "expected_primary_faces": {
                            "min_ids": expected[0],
                            "max_ids": expected[1],
                        },
                        "reason": "source skips labels with max_ids=0",
                    },
                )
            else:
                face_runnable.append(record)
        # Face applicability is label-first in the released evaluator, even if media is absent.
        for record in request["records"]:
            record_id = str(record["id"])
            expected = _expected_minmax(record)
            if expected is None:
                by_id[record_id].update(status="missing_face_label", eligible=None, error=None)
            elif expected[1] == 0:
                by_id[record_id].update(
                    status="ineligible",
                    eligible=False,
                    error=None,
                    raw_details={
                        "expected_primary_faces": {
                            "min_ids": expected[0],
                            "max_ids": expected[1],
                        },
                        "reason": "source skips labels with max_ids=0",
                    },
                )
        runnable = face_runnable
    output_dir = Path(str(request.get("output_dir") or request.get("work_dir"))).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_tag = str(request.get("run_tag", "run"))
    safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in run_tag) or "run"
    log_path = output_dir / f"{safe_tag}-{dimension.replace('-', '_')}.log"

    raw_summary: Dict[str, Any] = {}
    with log_path.open("w", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            try:
                if runnable:
                    raw_summary = _RUNNERS[dimension](request, runnable, by_id)
                else:
                    raw_summary = _source_summary(by_id.values())
                    if dimension == "Face":
                        expected = [_expected_minmax(item) for item in request["records"]]
                        raw_summary.update(
                            count_annotations=sum(item is not None for item in expected),
                            count_label_eligible=sum(
                                item is not None and item[1] > 0 for item in expected
                            ),
                            canonical_expected_annotations=220,
                            canonical_expected_label_eligible=103,
                            summaries_by_category=[],
                        )
            except Exception as exc:
                print(traceback.format_exc())
                message = _bounded_error(exc)
                for record in runnable:
                    record_id = str(record["id"])
                    if by_id[record_id].get("status") == "failed":
                        by_id[record_id]["error"] = message

    ordered = [by_id[str(record["id"])] for record in request["records"]]
    result = {
        "schema": SCHEMA_VERSION,
        "dimension": dimension,
        "source_revision": SOURCE_REVISION,
        "protocol_identity": _PROTOCOL_IDENTITIES[dimension],
        "backend_identity": _BACKEND_IDENTITIES[dimension],
        "adapter_digest": _adapter_digest(),
        "records": ordered,
        "raw_summary": raw_summary,
        "counts": _counts(ordered),
        "log_path": str(log_path),
    }
    result["counts"]["located"] = sum(
        _record_path(record).is_file() for record in request["records"]
    )
    if isinstance(request.get("asset_identities"), Mapping):
        result["asset_identities"] = dict(request["asset_identities"])
    return _json_safe(result)


__all__ = ["AdapterError", "DIMENSIONS", "SOURCE_REVISION", "run_dimension"]

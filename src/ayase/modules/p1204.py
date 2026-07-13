"""ITU-T P.1204.3 -- Bitstream-based NR Video Quality for UHD.

ITU-T Rec. P.1204.3 (2020) is a no-reference *bitstream* video-quality model:
it reads per-frame/per-GOP coding statistics parsed from the compressed
H.264/HEVC/VP9 stream (never from decoded pixels) and predicts a Mean Opinion
Score. The reference implementation
(https://github.com/Telecommunication-Telemedia-Assessment/bitstream_mode3_p1204_3)
combines a parametric baseline (per-codec coding + upscaling + framerate
degradations on the ITU R-scale) with a trained random-forest *residual*.

Both parts are wired here from the JSON coefficients mirrored on the Hugging
Face Hub (``p1204/models/p1204_3/``):

* ``config.json`` -- per-device (``pc``/``mobile``) parametric coefficients plus
  the name of the RF regressor file.
* ``mode3_{pc,mobile}_20trees_depth_8_reg.json`` -- a scikit-learn
  ``RandomForestRegressor`` serialized as node arrays (20 trees, depth 8,
  16 input features). Loaded and run with a self-contained pure-NumPy tree
  forward (verified bit-for-bit against scikit-learn's own predict), so it does
  not depend on any particular scikit-learn internal ``Tree`` layout.
* ``mode3_{pc,mobile}_20trees_depth_8_fs.json`` -- a 20-element boolean
  feature-selection mask applied before the regressor.

The RF model is therefore fully loaded and ready. The *bitstream features* it
consumes, however, require the companion patched-FFmpeg parser
``bitstream_mode3_videoparser`` (https://github.com/Telecommunication-Telemedia-
Assessment/bitstream_mode3_videoparser), which is a compiled C++/FFmpeg binary
and is **not** pip-installable. When that parser is not available the module
degrades to an honest ``unavailable`` state (model loaded, but no features can
be extracted) and emits no score -- it never fabricates bitstream features from
pixels under the P.1204.3 name. Point the env var
``AYASE_P1204_VIDEOPARSER`` at the built ``parser.sh`` to enable the real path.

p1204_mos -- 1-5, higher = better

REVIVAL NOTES (provisional -- EXTERNAL): RF model + parametric baseline are already code-complete
(verified bit-for-bit vs sklearn). To revive: build the bitstream feature parser
``bitstream_mode3_videoparser`` (patched ffmpeg; run its build.sh) and set env var
AYASE_P1204_VIDEOPARSER=/path/to/parser.sh (+ ffprobe on PATH). No code change needed.
Source: https://github.com/Telecommunication-Telemedia-Assessment/bitstream_mode3_videoparser
"""

import bz2
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

_HF_REPO = "AkaneTendo25/ayase-models"
_HF_MODEL_DIR = "p1204/models/p1204_3"
_HF_CONFIG = _HF_MODEL_DIR + "/config.json"

# ITU-T R <-> MOS conversion (P.1204.3 modelutils). MOS_MIN/MAX and the
# closed-form mos_from_r match the reference; the r_from_mos lookup table is
# regenerated exactly from mos_from_r (verified identical to the upstream
# hardcoded 389-entry table).
_MOS_MIN = 1.05
_MOS_MAX = 4.9


def _mos_from_r(q: float) -> float:
    if q <= 0:
        return _MOS_MIN
    if q >= 100:
        return _MOS_MAX
    return (
        _MOS_MIN
        + (_MOS_MAX - _MOS_MIN) * q / 100.0
        + q * (q - 60.0) * (100.0 - q) * 0.000007
    )


# R-scale support points used by the reference table: {0} U {3.25, 3.5, ..., 100}
_R_VALUES = np.array([0.0] + list(np.arange(3.25, 100.0001, 0.25)))
_R_KEYS = np.array([_mos_from_r(v) for v in _R_VALUES])


def _r_from_mos(mos: float) -> float:
    mos = min(max(mos, _MOS_MIN), _MOS_MAX)
    return float(np.interp(mos, _R_KEYS, _R_VALUES))


def _map_to_5(x: float) -> float:
    """Linear remap [1, 4.5] -> [1, 5] (values >= 4.5 clamp to 5)."""
    if x >= 4.5:
        return 5.0
    return 1.0 + ((5.0 - 1.0) / (4.5 - 1.0)) * (x - 1.0)


# --------------------------------------------------------------------------- #
# Random-forest ensemble: parse the sklearn-serialized JSON into plain node
# arrays and run a pure-NumPy forward. Verified bit-for-bit against
# sklearn.ensemble.RandomForestRegressor.predict on the same JSON.
# --------------------------------------------------------------------------- #
class _RandomForest:
    """A serialized sklearn RandomForestRegressor + boolean feature mask."""

    def __init__(self, trees: List[Dict[str, np.ndarray]], fs_mask: Optional[np.ndarray]):
        self._trees = trees
        self._fs_mask = fs_mask

    @classmethod
    def from_json(cls, reg_path: str, fs_path: Optional[str]) -> "_RandomForest":
        with open(reg_path) as fp:
            model_dict = json.load(fp)
        trees = [cls._build_tree(est["tree_"]) for est in model_dict["estimators_"]]
        fs_mask = None
        if fs_path and os.path.isfile(fs_path):
            with open(fs_path) as fp:
                fs = json.load(fp)
            fs_mask = np.array(fs, dtype=bool)
        return cls(trees, fs_mask)

    @staticmethod
    def _build_tree(tree_dict: dict) -> Dict[str, np.ndarray]:
        # nodes: [left_child, right_child, feature, threshold, impurity,
        #         n_node_samples, weighted_n_node_samples]
        nodes = tree_dict["nodes"]
        left = np.array([n[0] for n in nodes], dtype=np.int64)
        right = np.array([n[1] for n in nodes], dtype=np.int64)
        feature = np.array([n[2] for n in nodes], dtype=np.int64)
        threshold = np.array([n[3] for n in nodes], dtype=np.float64)
        values = np.array(
            [v[0][0] for v in tree_dict["values"]], dtype=np.float64
        )
        return {
            "left": left,
            "right": right,
            "feature": feature,
            "threshold": threshold,
            "value": values,
        }

    @staticmethod
    def _predict_tree(tree: Dict[str, np.ndarray], x: np.ndarray) -> float:
        left = tree["left"]
        right = tree["right"]
        feature = tree["feature"]
        threshold = tree["threshold"]
        node = 0
        # leaf nodes have left_child == -1 (sklearn TREE_LEAF)
        while left[node] != -1:
            if x[feature[node]] <= threshold[node]:
                node = left[node]
            else:
                node = right[node]
        return float(tree["value"][node])

    def predict_one(self, x: np.ndarray) -> float:
        if self._fs_mask is not None:
            x = x[self._fs_mask]
        preds = [self._predict_tree(t, x) for t in self._trees]
        return float(np.mean(preds))


# QP normalisation divisors per (codec, bit-depth) -- from the reference model.
_CODEC_QP_DIVISOR = {
    "h264": 51,
    "h264_10bit": 63,
    "hevc": 51,
    "hevc_10bit": 63,
    "vp9": 255,
}
_CODECS = ["h264", "h264_10bit", "hevc", "hevc_10bit", "vp9"]

# The 20 RF input columns, sorted exactly as in the reference model.py.
_RF_COLUMNS = sorted(
    set(
        [
            "Bitrate",
            "Resolution",
            "Framerate",
            "mean_Av_QPBB_non_i",
            "predicted_mos_mode3_baseline",
            "quant",
            "1.0_quantil_FrameSize",
            "kurtosis_Av_Motion",
            "iqr_Av_QPBB_non-i",
            "kurtosis_FrameSize_non-i",
            "std_FrameSize_non-i",
            "kurtosis_Av_QPBB_non-i",
            "iqr_min_QP",
            "0.0_quantil_StdDev_MotionX_non-i",
            "std_max_QP_non-i",
        ]
        + _CODECS
    )
)


class P1204Module(PipelineModule):
    name = "p1204"
    provisional = True  # no turnkey real backend in a standard install
    description = "ITU-T P.1204.3 bitstream NR quality (2020)"
    default_config = {
        "device_type": "pc",  # pc | tv | tablet | mobile
        "device_resolution": "3840x2160",  # 3840x2160 | 2560x1440
    }
    metric_groups = {
        "p1204_mos": "codec",
    }

    def __init__(self, config=None):
        super().__init__(config)
        device = str(self.config.get("device_type", "pc")).lower()
        self._device_key = "pc" if device in ("pc", "tv") else "mobile"
        res = str(self.config.get("device_resolution", "3840x2160"))
        try:
            w, h = res.split("x")
            self._display_res = float(w) * float(h)
        except Exception:
            self._display_res = 3840.0 * 2160.0
        self._ml_available = False
        self._backend = "unavailable"
        self._forest: Optional[_RandomForest] = None
        self._params: Optional[dict] = None
        self._parser_script: Optional[str] = None

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        if self.test_mode:
            return

        # (1) Load the trained model from the JSON coefficients. This part is
        #     fully reproducible and leaves the RF ready to predict.
        try:
            self._load_model()
        except Exception as exc:  # missing hub / offline / bad JSON
            self._backend = "unavailable"
            self._ml_available = False
            logger.info(
                "P.1204.3 unavailable: could not load trained model from "
                "'%s' (%s). No score emitted.",
                _HF_REPO,
                exc,
            )
            return

        # (2) The bitstream FEATURES require the compiled videoparser, which is
        #     not pip-installable. Without it we stay honestly unavailable.
        parser = os.environ.get("AYASE_P1204_VIDEOPARSER", "").strip()
        if parser and Path(parser).is_file():
            self._parser_script = parser
            self._backend = "real"
            self._ml_available = True
            return

        self._backend = "unavailable"
        self._ml_available = False
        logger.info(
            "P.1204.3 model loaded but bitstream features unavailable: the "
            "compiled 'bitstream_mode3_videoparser' parser is required "
            "(https://github.com/Telecommunication-Telemedia-Assessment/"
            "bitstream_mode3_videoparser). Set AYASE_P1204_VIDEOPARSER to its "
            "built parser.sh to enable scoring. No score emitted."
        )

    def _load_model(self) -> None:
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=_HF_REPO, filename=_HF_CONFIG)
        with open(config_path) as fp:
            config = json.load(fp)
        device_cfg = config[self._device_key]
        self._params = device_cfg["params"]

        reg_name = device_cfg["rf"]  # e.g. mode3_pc_20trees_depth_8_reg.json
        fs_name = reg_name.replace("_reg.json", "_fs.json")
        reg_path = hf_hub_download(
            repo_id=_HF_REPO, filename=f"{_HF_MODEL_DIR}/{reg_name}"
        )
        try:
            fs_path = hf_hub_download(
                repo_id=_HF_REPO, filename=f"{_HF_MODEL_DIR}/{fs_name}"
            )
        except Exception:
            fs_path = None
        self._forest = _RandomForest.from_json(reg_path, fs_path)

    # ---------------------------------------------------------------- process
    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or self._backend != "real":
            return sample
        if not sample.is_video:
            return sample
        if self._forest is None or self._params is None or self._parser_script is None:
            return sample

        mos = self._compute_mos(sample)
        if mos is not None:
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.p1204_mos = float(np.clip(mos, 1.0, 5.0))
        return sample

    def _compute_mos(self, sample: Sample) -> Optional[float]:
        try:
            probe = self._ffprobe(str(sample.path))
            if probe is None or probe["codec"] not in ("h264", "hevc", "vp9"):
                logger.info(
                    "P.1204.3: unsupported/unknown codec for %s", sample.path
                )
                return None
            frames = self._run_parser(str(sample.path))
            if not frames:
                return None
            feat = self._extract_features(frames, probe)
            if feat is None:
                return None
            return self._predict(feat)
        except Exception as exc:
            logger.info("P.1204.3 scoring failed for %s: %s", sample.path, exc)
            return None

    # ------------------------------------------------------- feature pipeline
    @staticmethod
    def _ffprobe(path: str) -> Optional[dict]:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-show_format",
            path,
        ]
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            return None
        data = json.loads(out.stdout)
        vstream = None
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                vstream = s
                break
        if vstream is None:
            return None
        fmt = data.get("format", {})

        def _fps(val: str) -> float:
            try:
                if "/" in val:
                    n, d = val.split("/")
                    return float(n) / float(d) if float(d) else 60.0
                return float(val)
            except Exception:
                return 60.0

        bitrate = vstream.get("bit_rate") or fmt.get("bit_rate") or 0
        return {
            "codec": vstream.get("codec_name", ""),
            "bitrate": float(bitrate),
            "avg_frame_rate": _fps(vstream.get("avg_frame_rate", "60/1")),
            "width": int(vstream.get("width", 0)),
            "height": int(vstream.get("height", 0)),
            "duration": float(fmt.get("duration", 0.0) or 0.0),
        }

    def _run_parser(self, video_path: str) -> List[dict]:
        """Run the compiled bitstream parser and return its per-frame stats."""
        import tempfile

        tmp = Path(tempfile.gettempdir()) / (
            Path(video_path).stem + "_p1204.json.bz2"
        )
        cmd = [
            "bash",
            self._parser_script,
            video_path,
            "--output",
            str(tmp),
        ]
        ret = subprocess.run(cmd, capture_output=True, text=True)
        if ret.returncode != 0 or not tmp.is_file():
            logger.info(
                "P.1204.3 bitstream parser failed (%s): %s",
                self._parser_script,
                ret.stderr[:200],
            )
            return []
        opener = bz2.open if str(tmp).endswith(".bz2") else open
        with opener(tmp, "rt") as fp:
            frames = json.load(fp)
        return frames

    def _extract_features(self, frames: List[dict], probe: dict) -> Optional[dict]:
        # bit depth from the parsed sequence header
        bitdepth = 8
        for fr in frames:
            seq = fr.get("Seq", {})
            if "BitDepth" in seq:
                bitdepth = seq["BitDepth"]
                break

        codec = probe["codec"]
        if codec == "h264" and bitdepth == 10:
            video_codec = "h264_10bit"
        elif codec == "hevc" and bitdepth == 10:
            video_codec = "hevc_10bit"
        else:
            video_codec = codec
        if video_codec not in _CODEC_QP_DIVISOR:
            return None

        fsz = self._stats_per_gop(frames, ["FrameSize"])
        qp = self._stats_per_gop(frames, ["Av_QP", "Av_QPBB", "max_QP", "min_QP"])
        mot = self._stats_per_gop(
            frames,
            [
                "Av_Motion",
                "Av_MotionDif",
                "Av_MotionX",
                "Av_MotionY",
                "StdDev_Motion",
                "StdDev_MotionDif",
                "StdDev_MotionX",
                "StdDev_MotionY",
            ],
        )

        return {
            "video_codec": video_codec,
            "Bitrate": probe["bitrate"] / 1024.0,
            "Resolution": float(probe["width"] * probe["height"]),
            "Framerate": probe["avg_frame_rate"],
            "mean_Av_QPBB_non-i": qp.get("mean_Av_QPBB_non-i", 0.0),
            "iqr_Av_QPBB_non-i": qp.get("iqr_Av_QPBB_non-i", 0.0),
            "kurtosis_Av_QPBB_non-i": qp.get("kurtosis_Av_QPBB_non-i", 0.0),
            "iqr_min_QP": qp.get("iqr_min_QP", 0.0),
            "std_max_QP_non-i": qp.get("std_max_QP_non-i", 0.0),
            "1.0_quantil_FrameSize": fsz.get("1.0_quantil_FrameSize", 0.0),
            "std_FrameSize_non-i": fsz.get("std_FrameSize_non-i", 0.0),
            "kurtosis_FrameSize_non-i": fsz.get("kurtosis_FrameSize_non-i", 0.0),
            "kurtosis_Av_Motion": mot.get("kurtosis_Av_Motion", 0.0),
            "0.0_quantil_StdDev_MotionX_non-i": mot.get(
                "0.0_quantil_StdDev_MotionX_non-i", 0.0
            ),
        }

    @staticmethod
    def _stats_per_gop(frames: List[dict], needed: List[str]) -> dict:
        """Port of the reference ``stats_per_gop``: per-GOP mean/std/skew/
        kurtosis/iqr + 11 quantiles for each column, over all frames and over
        non-I frames, then averaged across GOPs (NaN-skipping)."""
        import scipy.stats as st

        def gops():
            gop: List[dict] = []
            for fr in frames:
                if fr.get("IsIDR", 0) == 1 and gop:
                    yield gop
                    gop = []
                gop.append(fr)
            if gop:
                yield gop

        def col_stats(rows: List[dict], col: str, suffix: str, out: dict) -> None:
            vals = np.array(
                [r[col] for r in rows if col in r and r[col] is not None],
                dtype=float,
            )
            if vals.size == 0:
                keys = ["mean", "median", "std", "skew", "kurtosis", "iqr"]
                for k in keys:
                    out[f"{k}_{col}{suffix}"] = np.nan
                for i in range(11):
                    q = round(0.1 * i, 1)
                    out[f"{q}_quantil_{col}{suffix}"] = np.nan
                return
            out[f"mean_{col}{suffix}"] = float(np.mean(vals))
            out[f"median_{col}{suffix}"] = float(np.median(vals))
            out[f"std_{col}{suffix}"] = (
                float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
            )
            out[f"skew_{col}{suffix}"] = float(st.skew(vals))
            out[f"kurtosis_{col}{suffix}"] = float(st.kurtosis(vals))
            out[f"iqr_{col}{suffix}"] = float(st.iqr(vals))
            for i in range(11):
                q = round(0.1 * i, 1)
                out[f"{q}_quantil_{col}{suffix}"] = float(np.quantile(vals, q))

        gop_results: List[dict] = []
        for gop in gops():
            res: dict = {}
            for col in needed:
                col_stats(gop, col, "", res)
            non_i = [f for f in gop if f.get("FrameType") != 1]
            for col in needed:
                col_stats(non_i, col, "_non-i", res)
            gop_results.append(res)

        if not gop_results:
            return {}
        keys = set().union(*[r.keys() for r in gop_results])
        agg: dict = {}
        for k in keys:
            col = np.array([r.get(k, np.nan) for r in gop_results], dtype=float)
            if np.all(np.isnan(col)):
                agg[k] = 0.0
            else:
                agg[k] = float(np.nanmean(col))
        return agg

    # ------------------------------------------------------------- prediction
    def _predict(self, feat: dict) -> Optional[float]:
        params = self._params
        video_codec = feat["video_codec"]
        divisor = _CODEC_QP_DIVISOR.get(video_codec)
        if divisor is None:
            return None

        quant = feat["mean_Av_QPBB_non-i"] / divisor

        # parametric coding degradation (R-scale) for the active codec
        a = params[f"{video_codec}_a"]
        b = params[f"{video_codec}_b"]
        c = params[f"{video_codec}_c"]
        d = params[f"{video_codec}_d"]
        mos_q = float(np.clip(a + b * np.exp(c * quant + d), 1.0, 5.0))
        cod_deg = float(np.clip(100.0 - _r_from_mos(mos_q), 0.0, 100.0))

        resolution_deg = float(
            np.clip(
                params["x"]
                * np.log(params["y"] * (feat["Resolution"] / self._display_res)),
                0.0,
                100.0,
            )
        )
        framerate_deg = float(
            np.clip(
                params["z"] * np.log(params["k"] * feat["Framerate"] / 60.0),
                0.0,
                100.0,
            )
        )

        pred_r = 100.0 - (cod_deg + resolution_deg + framerate_deg)
        pred = float(np.clip(_mos_from_r(pred_r), 1.0, 5.0))
        baseline = _map_to_5(pred)

        # RF residual on the 20-column feature vector
        row = {
            "Bitrate": feat["Bitrate"],
            "Resolution": feat["Resolution"],
            "Framerate": feat["Framerate"],
            "quant": quant,
            "predicted_mos_mode3_baseline": baseline,
            "mean_Av_QPBB_non_i": feat["mean_Av_QPBB_non-i"],
            "1.0_quantil_FrameSize": feat["1.0_quantil_FrameSize"],
            "kurtosis_Av_Motion": feat["kurtosis_Av_Motion"],
            "iqr_Av_QPBB_non-i": feat["iqr_Av_QPBB_non-i"],
            "kurtosis_FrameSize_non-i": feat["kurtosis_FrameSize_non-i"],
            "std_FrameSize_non-i": feat["std_FrameSize_non-i"],
            "kurtosis_Av_QPBB_non-i": feat["kurtosis_Av_QPBB_non-i"],
            "iqr_min_QP": feat["iqr_min_QP"],
            "0.0_quantil_StdDev_MotionX_non-i": feat[
                "0.0_quantil_StdDev_MotionX_non-i"
            ],
            "std_max_QP_non-i": feat["std_max_QP_non-i"],
        }
        for cod in _CODECS:
            row[cod] = 1.0 if video_codec == cod else 0.0

        x = np.array([row.get(col, 0.0) for col in _RF_COLUMNS], dtype=float)
        x = np.nan_to_num(x, nan=0.0)
        residual = self._forest.predict_one(x)

        predicted = float(np.clip(_map_to_5(pred) + residual, 1.0, 5.0))
        final = 0.5 * baseline + 0.5 * predicted
        return float(np.clip(final, 1.0, 5.0))

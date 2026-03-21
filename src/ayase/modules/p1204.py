"""ITU-T P.1204.3 — Bitstream-based NR Video Quality for UHD.

GitHub: https://github.com/Telecommunication-Telemedia-Assessment/bitstream_mode3_p1204_3
p1204_mos — 1-5, higher = better
"""
import logging, subprocess, json
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)

class P1204Module(PipelineModule):
    name = "p1204"; description = "ITU-T P.1204.3 bitstream NR quality (2020)"; default_config = {}
    def __init__(self, c=None):
        super().__init__(c); self._backend = "heuristic"
    def setup(self):
        try:
            import itu_p1204_3; self._model = itu_p1204_3; self._backend = "native"; return
        except ImportError: pass
        self._backend = "heuristic"
    def process(self, sample):
        if not sample.is_video: return sample
        try:
            if self._backend == "native":
                mos = float(self._model.predict(str(sample.path)))
            else:
                mos = self._heuristic(sample)
            if mos is not None:
                if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
                sample.quality_metrics.p1204_mos = mos
        except Exception as e:
            logger.warning(f"P.1204 failed: {e}")
        return sample

    def _heuristic(self, sample):
        """Heuristic: bitrate + resolution + fps + codec → predicted MOS."""
        vm = sample.video_metadata
        if vm is None: return None
        # Resolution factor
        pixels = max(vm.width * vm.height, 1)
        if pixels >= 3840*2160: res_f = 1.0
        elif pixels >= 1920*1080: res_f = 0.9
        elif pixels >= 1280*720: res_f = 0.75
        elif pixels >= 854*480: res_f = 0.6
        else: res_f = 0.4
        # Bitrate factor
        if vm.bitrate and vm.bitrate > 0:
            bpp = vm.bitrate / (pixels * max(vm.fps, 1))
            br_f = min(bpp / 0.1, 1.0)
        else: br_f = 0.5
        # FPS factor
        fps_f = min(vm.fps / 30.0, 1.0) if vm.fps > 0 else 0.5
        # Codec factor
        codec_map = {"h265": 1.0, "hevc": 1.0, "av1": 1.0, "vp9": 0.9, "h264": 0.85, "avc": 0.85}
        codec_f = codec_map.get((vm.codec or "").lower(), 0.7)
        # Combine → MOS 1-5
        quality = 0.35*res_f + 0.30*br_f + 0.15*fps_f + 0.20*codec_f
        mos = 1.0 + 4.0 * quality
        return float(min(max(mos, 1.0), 5.0))

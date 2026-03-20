"""SQI — Streaming Quality Index (2016). sqi_score — higher = better"""
import logging
from ayase.models import Sample, QualityMetrics
from ayase.pipeline import PipelineModule
logger = logging.getLogger(__name__)
class SQIModule(PipelineModule):
    name = "sqi"; description = "SQI streaming quality index (2016)"; default_config = {}
    def process(self, sample):
        """Heuristic: base_quality * stalling_factor. Without network data, proxy from metadata."""
        if not sample.is_video: return sample
        try:
            vm = sample.video_metadata
            if vm is None: return sample
            # Base quality from resolution + bitrate
            pixels = vm.width * vm.height
            res_q = min(pixels / (1920*1080), 1.0)
            br_q = min((vm.bitrate or 5_000_000) / 10_000_000, 1.0)
            fps_q = min(vm.fps / 30.0, 1.0) if vm.fps > 0 else 0.5
            base = 0.4*res_q + 0.4*br_q + 0.2*fps_q
            # No stalling data available from file, assume no stalls
            stalling_factor = 1.0
            score = base * stalling_factor
            if sample.quality_metrics is None: sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.sqi_score = float(min(max(score, 0), 1))
        except Exception as e: logger.warning(f"SQI failed: {e}")
        return sample

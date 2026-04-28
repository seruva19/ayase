"""Video/camera motion analysis using the Kandinsky VideoMAE-V2 motion predictor.

Predicts Kandinsky camera, object, and dynamics scores. Higher values indicate
more motion in the corresponding channel. Flags static videos with low dynamics.
"""

import logging
from ayase.models import QualityMetrics, Sample, ValidationIssue, ValidationSeverity
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)

class KandinskyMotionModule(PipelineModule):
    name = "kandinsky_motion"
    description = "Video/Camera Motion Analysis using Kandinsky Video Tools (VideoMAE-V2)"
    default_config = {"models_dir": "models"}
    models = [
        {
            "id": "ai-forever/kandinsky-video-motion-predictor",
            "type": "huggingface",
            "task": "VideoMAE-V2 camera/object/dynamics motion predictor",
            "notes": "Loaded through bundled Kandinsky third-party wrapper",
        },
    ]
    metric_info = {
        "kandinsky_camera_motion_score": "Camera motion prediction score (higher=more camera motion)",
        "kandinsky_object_motion_score": "Object motion prediction score (higher=more object motion)",
        "kandinsky_dynamics_score": "Overall dynamics prediction score (higher=more dynamic)",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._device = "cpu"
        self._ml_available = False

    def setup(self) -> None:
        try:
            import torch

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Setting up Kandinsky Motion Predictor on {self._device}...")
            
            # Import vendored module
            try:
                from ayase.third_party.kandinsky.video_motion_predictor.model import VideoMotionPredictor
            except ImportError as e:
                logger.warning(f"Failed to import Kandinsky Video Tools: {e}")
                return

            models_dir = self.config.get("models_dir", "models")

            # Load model from HuggingFace
            # We assume the repo is 'ai-forever/kandinsky-video-motion-predictor' and weights are in 'models/video_motion_predictor'
            try:
                self._model = VideoMotionPredictor.from_pretrained(
                    "ai-forever/kandinsky-video-motion-predictor", 
                    subfolder="models/video_motion_predictor",
                    cache_dir=models_dir
                ).to(self._device).eval()
            except Exception as e_sub:
                logger.info(f"Could not load with subfolder, trying root: {e_sub}")
                self._model = VideoMotionPredictor.from_pretrained(
                    "ai-forever/kandinsky-video-motion-predictor",
                    cache_dir=models_dir
                ).to(self._device).eval()

            self._ml_available = True
            logger.info("Kandinsky Motion Predictor loaded successfully.")

        except Exception as e:
            logger.warning(f"Failed to setup Kandinsky Motion Module: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available or not sample.is_video:
            return sample

        try:
            # inference returns a dict: 
            # {
            #   "camera_movement_score": [float], 
            #   "object_movement_score": [float], 
            #   "dynamics_score": [float] 
            # }
            # It expects a list of paths or single path.
            
            results = self._model.inference(str(sample.path))
            
            # Results are lists (batch size 1)
            cam_score = results["camera_movement_score"][0]
            obj_score = results["object_movement_score"][0]
            dyn_score = results["dynamics_score"][0]
            
            # Add to metrics or issues
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            sample.quality_metrics.kandinsky_camera_motion_score = float(cam_score)
            sample.quality_metrics.kandinsky_object_motion_score = float(obj_score)
            sample.quality_metrics.kandinsky_dynamics_score = float(dyn_score)
            
            # Interpret scores (Assuming 0-1 or 0-10 scale?)
            # VideoMAE scores are typically logits or normalized. 
            # The model code had 'targets_norm=[3,3,3]', suggesting outputs might be scaled to 0-3 range?
            # Or maybe just normalized. Let's log them as INFO first.
            
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Kandinsky Motion - Dynamics: {dyn_score:.2f}, Camera: {cam_score:.2f}, Object: {obj_score:.2f}",
                    details={
                        "kandinsky_dynamics": dyn_score,
                        "kandinsky_camera": cam_score,
                        "kandinsky_object": obj_score
                    }
                )
            )
            
            # Heuristics based on scores (TBD)
            if dyn_score < 0.1:
                 sample.validation_issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message="Video appears static (Low Dynamics Score)",
                        details={"dynamics_score": dyn_score}
                    )
                )

        except Exception as e:
            logger.warning(f"Kandinsky Motion inference failed: {e}")

        return sample

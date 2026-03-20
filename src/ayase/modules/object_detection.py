import logging
import cv2
import numpy as np
from typing import Optional

from ayase.models import Sample, ValidationIssue, ValidationSeverity, QualityMetrics
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ObjectDetectionModule(PipelineModule):
    name = "object_detection"
    description = "Detects objects (GRiT / YOLOv8) - Supports Heavy Models"
    default_config = {
        "model_name": "yolov8n.pt",  # Default small model
        "use_yolo_world": False,  # Enable for open-vocabulary detection
        "use_grit": False, # Enable GRiT (Grounding Representation in Transformers)
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.model_name = self.config.get("model_name", "yolov8n.pt")
        self.use_yolo_world = self.config.get("use_yolo_world", False)
        self.use_grit = self.config.get("use_grit", False)

        self._model = None
        self._ml_available = False
        self._mode = "yolo" # yolo or grit

    def on_mount(self) -> None:
        super().on_mount()
        # 1. Try GRiT if requested
        if self.use_grit:
            try:
                import torch
                from grit.predictor import VisualizationDemo

                grit_config = self.config.get("grit_config")
                grit_weights = self.config.get("grit_weights")
                if grit_config and grit_weights:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    logger.info(f"Loading GRiT on {device}...")
                    self._model = VisualizationDemo(
                        grit_config, grit_weights, device=device
                    )
                    self._mode = "grit"
                    self._ml_available = True
                    return
                logger.warning("GRiT requested but grit_config/grit_weights not set; falling back to YOLO.")
            except ImportError:
                logger.warning("GRiT not found. Falling back to YOLO.")
            except Exception as e:
                logger.warning(f"Failed to load GRiT: {e}. Falling back to YOLO.")

        # 2. YOLO Fallback
        try:
            from ultralytics import YOLO
            import os

            models_dir = self.config.get("models_dir", "models")
            os.makedirs(models_dir, exist_ok=True)

            if self.use_yolo_world:
                logger.info("Loading YOLO-World (Open Vocabulary)...")
                if "world" not in self.model_name:
                    self.model_name = "yolov8s-world.pt"
                # Forcing download to models_dir by providing full path
                model_path = os.path.join(models_dir, self.model_name)
                self._model = YOLO(model_path)
            else:
                logger.info(f"Loading YOLO ({self.model_name})...")
                model_path = os.path.join(models_dir, self.model_name)
                self._model = YOLO(model_path)

            self._ml_available = True
            self._mode = "yolo"
        except ImportError:
            logger.warning("ultralytics not installed. Object detection disabled.")
        except Exception as e:
            logger.warning(f"Failed to load YOLO: {e}")

    def process(self, sample: Sample) -> Sample:
        if not self._ml_available:
            return sample

        try:
            from ayase.utils.sampling import FrameSampler
            frames = FrameSampler.sample_frames(sample.path, num_frames=8)
            
            if not frames:
                return sample

            all_detected_classes = set()
            if not hasattr(sample, 'detections') or sample.detections is None:
                sample.detections = []
            # Strategy: Store detections from the *middle* frame for visualization, 
            # but use ALL frames for consistency check.
            
            # Or store all? If 8 frames x 10 objects = 80 items. Reasonable.
            
            for i, image in enumerate(frames):
                if self._mode == "grit":
                    self._process_grit_frame(sample, image, all_detected_classes, frame_idx=i)
                else:
                    self._process_yolo_frame(sample, image, all_detected_classes, frame_idx=i)
            
            # Consistency check using caption (Union of all frames)
            self._check_consistency(sample, all_detected_classes)

            confidences = [d.get("conf") for d in sample.detections if d.get("conf") is not None]
            if sample.quality_metrics is None:
                sample.quality_metrics = QualityMetrics()
            if confidences:
                sample.quality_metrics.detection_score = float(np.mean(confidences)) * 100.0
            else:
                sample.quality_metrics.detection_score = 0.0
            sample.quality_metrics.count_score = min(len(sample.detections) / 10.0, 1.0) * 100.0

            if sample.detections:
                label_scores = {}
                for d in sample.detections:
                    label = d.get("label")
                    conf = d.get("conf", 0.0)
                    if label is None:
                        continue
                    label_scores[label] = label_scores.get(label, 0.0) + float(conf)
                total = sum(label_scores.values())
                if total > 0:
                    probs = np.array([v / total for v in label_scores.values()], dtype=np.float64)
                    entropy = -np.sum(probs * np.log(probs + 1e-9))
                    sample.quality_metrics.detection_diversity = float(np.exp(entropy))
                else:
                    sample.quality_metrics.detection_diversity = 0.0
            else:
                sample.quality_metrics.detection_diversity = 0.0

        except Exception as e:
            logger.warning(f"Object detection failed: {e}")

        return sample

    def _process_yolo_frame(self, sample: Sample, image: np.ndarray, all_detected_classes: set, frame_idx: int):
        results = self._model(image, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = self._model.names[cls_id]
                all_detected_classes.add(class_name)

                # Store in sample (limit total count?)
                x, y, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x
                h = y2 - y
                conf = float(box.conf[0])

                sample.detections.append(
                    {
                        "label": class_name,
                        "box": [x, y, w, h],
                        "conf": conf,
                        "frame_idx": frame_idx # Track which frame
                    }
                )

    def _check_consistency(self, sample: Sample, all_detected_classes: set):
        """Check consistency between detected objects and caption if available."""
        if not sample.caption or not all_detected_classes:
            return

        caption_lower = sample.caption.text.lower()
        detected_in_caption = sum(
            1 for cls in all_detected_classes if cls.lower() in caption_lower
        )
        ratio = detected_in_caption / len(all_detected_classes) if all_detected_classes else 0.0

        if ratio < 0.1 and len(all_detected_classes) > 2:
            sample.validation_issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.INFO,
                    message=f"Low caption-detection overlap ({detected_in_caption}/{len(all_detected_classes)} objects mentioned)",
                    details={
                        "detected_classes": sorted(all_detected_classes),
                        "overlap_ratio": ratio,
                    },
                    recommendation="Caption may not describe the visual content accurately.",
                )
            )

    def _process_grit_frame(self, sample: Sample, image: np.ndarray, all_detected_classes: set, frame_idx: int):
        if not hasattr(self._model, "run_on_image"):
            logger.warning("GRiT model does not expose run_on_image; skipping GRiT inference.")
            return

        try:
            predictions, _ = self._model.run_on_image(image)
            instances = predictions.get("instances")
            if instances is None or len(instances) == 0:
                return

            if hasattr(instances, "pred_classes") and hasattr(instances, "scores") and hasattr(instances, "pred_boxes"):
                classes = instances.pred_classes.tolist()
                scores = instances.scores.tolist()
                boxes = instances.pred_boxes.tensor.tolist()
                for cls_id, conf, box in zip(classes, scores, boxes):
                    class_name = str(cls_id)
                    all_detected_classes.add(class_name)
                    x, y, x2, y2 = box
                    sample.detections.append(
                        {
                            "label": class_name,
                            "box": [x, y, x2 - x, y2 - y],
                            "conf": float(conf),
                            "frame_idx": frame_idx,
                        }
                    )
        except Exception as e:
            logger.warning(f"GRiT inference failed: {e}")

"""Active-speaker separation: is exactly ONE mouth in sync with the audio.

For a multi-person talking clip, ``lip_sync`` reports one LSE pair for the whole
frame, picking whichever face it finds. That cannot distinguish a clip where the
right person speaks from one where every face mouths along -- a common and very
visible failure of audio-driven avatar models.

Whether the CORRECT person speaks cannot be checked without an annotation of who
is supposed to speak when. This module answers the question that is checkable
from the clip alone: how far the best-synced face is ahead of the runner-up.
A margin near zero means the model animated every mouth, or none.

Each face track is cropped into its own clip -- the crop follows the face frame by
frame, because one fixed box around a moving person leaves the face too small for
SyncNet to find -- and carries the ORIGINAL audio, so faces are compared under an
identical soundtrack. A face for which no talking mouth is detected scores zero
rather than being dropped: that is the answer, not a measurement failure, and the
count of such faces is reported alongside.
"""

import logging
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np

from ayase.models import QualityMetrics, Sample
from ayase.pipeline import PipelineModule

logger = logging.getLogger(__name__)


class ActiveSpeakerModule(PipelineModule):
    name = "active_speaker"
    description = "Lip-sync separation between faces: is exactly one mouth in sync"
    default_config = {
        "model_name": "buffalo_l",
        "models_dir": "models",
        "stride": 2,
        "max_faces": 3,
        "crop_size": 256,
        "crop_pad": 0.8,
        "fps": 25,
    }
    metric_groups = {
        "active_speaker_margin": "audio",
        "active_speaker_best_lse_c": "audio",
        "active_speaker_silent_faces": "audio",
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.stride = int(self.config.get("stride", 2))
        self.max_faces = int(self.config.get("max_faces", 3))
        self.crop_size = int(self.config.get("crop_size", 256))
        self.crop_pad = float(self.config.get("crop_pad", 0.8))
        self.fps = int(self.config.get("fps", 25))
        self._app = None
        self._lip = None

    def setup(self) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            logger.warning("active_speaker: insightface not installed; metric disabled")
            return
        try:
            app = FaceAnalysis(
                name=self.config.get("model_name", "buffalo_l"),
                root=self.config.get("models_dir", "models"),
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            self._app = app
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("active_speaker: face backend failed to load (%s)", exc)
            return

        try:
            from ayase.modules.lip_sync import LipSyncModule

            lip = LipSyncModule({"models_dir": self.config.get("models_dir", "models"),
                                 "device": self.config.get("device", "auto")})
            lip.setup()
            self._lip = lip
        except Exception as exc:  # pragma: no cover - backend specific
            logger.warning("active_speaker: lip-sync backend unavailable (%s)", exc)

    def process(self, sample: Sample) -> Sample:
        if self._app is None or self._lip is None or not sample.is_video:
            return sample
        try:
            result = self._score(str(sample.path))
        except Exception as exc:  # pragma: no cover - depends on decoder/backend
            logger.warning("active_speaker failed for %s: %s", sample.path, exc)
            return sample
        if result is None:
            return sample

        if sample.quality_metrics is None:
            sample.quality_metrics = QualityMetrics()
        sample.quality_metrics.active_speaker_best_lse_c = result["best"]
        sample.quality_metrics.active_speaker_margin = result["margin"]
        sample.quality_metrics.active_speaker_silent_faces = result["silent"]
        return sample

    def _score(self, video: str) -> Optional[Dict[str, float]]:
        import cv2

        from ayase.faces import face_tracks
        from ayase.models import Sample as InnerSample

        def frames():
            capture = cv2.VideoCapture(video)
            if not capture.isOpened():
                return
            index = 0
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    if index % self.stride == 0:
                        yield index, np.ascontiguousarray(frame)
                    index += 1
            finally:
                capture.release()

        tracks = face_tracks(self._app, frames())
        if len(tracks) < 2:
            return None
        tracks = sorted(tracks, key=lambda t: -len(t["frames"]))[:self.max_faces]

        scores: List[float] = []
        silent = 0
        workdir = tempfile.mkdtemp(prefix="ayase_asd_")
        try:
            for position, track in enumerate(tracks):
                path = os.path.join(workdir, "face%d.mp4" % position)
                if not self._crop_track(video, track, path):
                    continue
                inner = InnerSample(path=path, is_video=True)
                self._lip.process(inner)
                metrics = getattr(inner, "quality_metrics", None)
                value = getattr(metrics, "lse_c", None) if metrics else None
                if value is None:
                    silent += 1
                    scores.append(0.0)
                else:
                    scores.append(float(value))
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        if len(scores) < 2:
            return None
        scores.sort(reverse=True)
        return {
            "best": round(scores[0], 4),
            "margin": round(scores[0] - scores[1], 4),
            "silent": float(silent),
        }

    def _crop_track(self, video: str, track: Dict[str, Any], out_path: str) -> bool:
        """Write a face-following crop of ``track`` with the original audio."""
        import subprocess

        import cv2

        indices = np.asarray(track["frames"], dtype=float)
        boxes = np.asarray(track["boxes"], dtype=float)
        if len(indices) < 2:
            return False

        capture = cv2.VideoCapture(video)
        if not capture.isOpened():
            return False
        source_fps = capture.get(cv2.CAP_PROP_FPS) or float(self.fps)
        silent_path = out_path + ".silent.mp4"
        writer = cv2.VideoWriter(
            silent_path, cv2.VideoWriter_fourcc(*"mp4v"), source_fps,
            (self.crop_size, self.crop_size),
        )
        if not writer.isOpened():
            capture.release()
            return False

        written = 0
        index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                box = [float(np.interp(index, indices, boxes[:, k])) for k in range(4)]
                width, height = box[2] - box[0], box[3] - box[1]
                if width > 1 and height > 1:
                    centre_x, centre_y = box[0] + width / 2.0, box[1] + height / 2.0
                    side = max(width, height) * (1.0 + self.crop_pad)
                    frame_h, frame_w = frame.shape[:2]
                    x1 = int(max(0, min(frame_w - 2, centre_x - side / 2)))
                    y1 = int(max(0, min(frame_h - 2, centre_y - side / 2)))
                    x2 = int(max(x1 + 2, min(frame_w, centre_x + side / 2)))
                    y2 = int(max(y1 + 2, min(frame_h, centre_y + side / 2)))
                    crop = frame[y1:y2, x1:x2]
                    if crop.size:
                        writer.write(cv2.resize(crop, (self.crop_size, self.crop_size)))
                        written += 1
                index += 1
        finally:
            writer.release()
            capture.release()

        if written < 10:
            self._unlink(silent_path)
            return False

        command = [
            "ffmpeg", "-y", "-v", "error", "-i", silent_path, "-i", video,
            "-map", "0:v:0", "-map", "1:a:0", "-r", str(self.fps),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-c:a", "aac", "-ar", "16000", "-ac", "1", "-shortest", out_path,
        ]
        try:
            done = subprocess.run(command, capture_output=True, text=True)
        except FileNotFoundError:
            logger.warning("active_speaker: ffmpeg not found; metric disabled")
            self._unlink(silent_path)
            return False
        self._unlink(silent_path)
        return done.returncode == 0 and os.path.exists(out_path)

    @staticmethod
    def _unlink(path: str) -> None:
        try:
            os.unlink(path)
        except OSError:
            pass

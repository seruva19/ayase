"""Face tracks and crops (port of ``run_pipeline.py`` from joonson/syncnet_python, MIT).

Steps and numeric parameters match the original: re-encode to 25 fps, S3FD on every
frame, scene split, IoU tracking, smoothed 224x224 crop.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Sequence, Tuple

import cv2
import numpy as np
from scenedetect import SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scipy import signal
from scipy.interpolate import interp1d

from .ffmpeg import run_ffmpeg
from .scoring import FACE_SIZE, read_frame

logger = logging.getLogger(__name__)

FACE_DETECTION_CONFIDENCE = 0.9
TRACK_IOU_THRESHOLD = 0.5
CROP_SMOOTHING_KERNEL = 13
CROP_PADDING_VALUE = 110
CROP_FOURCC = "XVID"
# ContentDetector threshold for PySceneDetect 0.6.x compatibility.
SCENE_CONTENT_THRESHOLD = 27.0

FaceDetection = Dict[str, Any]
FaceTrack = Dict[str, np.ndarray]


class FaceDetector(Protocol):
    def detect_faces(
        self, image: np.ndarray, conf_th: float = 0.8, scales: Tuple[float, ...] = (1.0,)
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class FaceTrackParams:
    """Tracking and crop parameters; defaults are those of ``run_pipeline.py``."""

    facedet_scale: float = 0.25
    crop_scale: float = 0.40
    min_track: int = 100
    frame_rate: int = 25
    num_failed_det: int = 25
    min_face_size: int = 100


def bb_intersection_over_union(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter_area / float(box_a_area + box_b_area - inter_area)


def track_shot(scenefaces: List[List[FaceDetection]], params: FaceTrackParams) -> List[FaceTrack]:
    """Link per-frame detections of one scene into tracks.

    Detections are linked in iteration order. Removing joined detections from a
    frame's list during iteration determines which detections join a track.
    ``scenefaces`` is consumed (joined detections are removed).
    """
    tracks: List[FaceTrack] = []

    while True:
        track: List[FaceDetection] = []
        for framefaces in scenefaces:
            for face in framefaces:
                if not track:
                    track.append(face)
                    framefaces.remove(face)
                elif face["frame"] - track[-1]["frame"] <= params.num_failed_det:
                    iou = bb_intersection_over_union(face["bbox"], track[-1]["bbox"])
                    if iou > TRACK_IOU_THRESHOLD:
                        track.append(face)
                        framefaces.remove(face)
                        continue
                else:
                    break

        if not track:
            break
        if len(track) > params.min_track:
            framenum = np.array([f["frame"] for f in track])
            bboxes = np.array([np.array(f["bbox"]) for f in track])

            frame_i = np.arange(framenum[0], framenum[-1] + 1)

            bboxes_i = np.stack(
                [interp1d(framenum, bboxes[:, ij])(frame_i) for ij in range(0, 4)], axis=1
            )

            mean_w = np.mean(bboxes_i[:, 2] - bboxes_i[:, 0])
            mean_h = np.mean(bboxes_i[:, 3] - bboxes_i[:, 1])
            if max(mean_w, mean_h) > params.min_face_size:
                tracks.append({"frame": frame_i, "bbox": bboxes_i})

    return tracks


def crop_video(
    track: FaceTrack,
    frame_files: List[Path],
    audio_path: Path,
    *,
    crop_base: Path,
    tmp_audio_path: Path,
    params: FaceTrackParams,
) -> Path:
    """Write the track as a 224x224 video with the matching audio slice; return its path."""
    silent_crop = crop_base.with_name(crop_base.name + "t.avi")
    final_crop = crop_base.with_name(crop_base.name + ".avi")

    fourcc = cv2.VideoWriter.fourcc(*CROP_FOURCC)
    v_out = cv2.VideoWriter(str(silent_crop), fourcc, params.frame_rate, (FACE_SIZE, FACE_SIZE))

    sizes = []
    centers_x = []
    centers_y = []
    for det in track["bbox"]:
        sizes.append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        centers_y.append((det[1] + det[3]) / 2)
        centers_x.append((det[0] + det[2]) / 2)

    smooth_s = signal.medfilt(sizes, kernel_size=CROP_SMOOTHING_KERNEL)
    smooth_x = signal.medfilt(centers_x, kernel_size=CROP_SMOOTHING_KERNEL)
    smooth_y = signal.medfilt(centers_y, kernel_size=CROP_SMOOTHING_KERNEL)

    cs = params.crop_scale
    for fidx, frame_idx in enumerate(track["frame"]):
        bs = smooth_s[fidx]
        bsi = int(bs * (1 + 2 * cs))

        image = read_frame(frame_files[frame_idx])
        padded = np.pad(
            image,
            ((bsi, bsi), (bsi, bsi), (0, 0)),
            "constant",
            constant_values=(CROP_PADDING_VALUE, CROP_PADDING_VALUE),
        )
        my = smooth_y[fidx] + bsi
        mx = smooth_x[fidx] + bsi

        face = padded[
            int(my - bs) : int(my + bs * (1 + 2 * cs)),
            int(mx - bs * (1 + cs)) : int(mx + bs * (1 + cs)),
        ]
        v_out.write(cv2.resize(face, (FACE_SIZE, FACE_SIZE)))

    v_out.release()

    audiostart = track["frame"][0] / params.frame_rate
    audioend = (track["frame"][-1] + 1) / params.frame_rate

    run_ffmpeg(
        [
            "-y",
            "-i",
            str(audio_path),
            "-ss",
            f"{audiostart:.3f}",
            "-to",
            f"{audioend:.3f}",
            str(tmp_audio_path),
        ]
    )
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(silent_crop),
            "-i",
            str(tmp_audio_path),
            "-c:v",
            "copy",
            "-c:a",
            "copy",
            str(final_crop),
        ]
    )
    silent_crop.unlink()

    return final_crop


def detect_faces_in_frames(
    detector: FaceDetector, frame_files: List[Path], facedet_scale: float
) -> List[List[FaceDetection]]:
    """Run the detector on every frame; one list of ``frame / bbox / conf`` dicts per frame."""
    dets: List[List[FaceDetection]] = []
    for fidx, fname in enumerate(frame_files):
        image = read_frame(fname)
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = detector.detect_faces(
            image_np, conf_th=FACE_DETECTION_CONFIDENCE, scales=(facedet_scale,)
        )
        dets.append(
            [{"frame": fidx, "bbox": (bbox[:-1]).tolist(), "conf": bbox[-1]} for bbox in bboxes]
        )
    return dets


def detect_scenes(video_path: Path) -> List[Tuple[int, int]]:
    """Scenes as ``(first frame, frame after the last)``.

    Without a cut the single scene spans the frames the decoder read.
    This count can be smaller than the number of frames ffmpeg extracted.
    """
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=SCENE_CONTENT_THRESHOLD))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        return [(0, video.frame_number)]
    return [(start.get_frames(), end.get_frames()) for start, end in scene_list]


def extract_face_tracks(
    video_path: Path, work_dir: Path, detector: FaceDetector, params: FaceTrackParams
) -> List[Path]:
    """Find face tracks in ``video_path`` and write each as a crop video; return crop paths."""
    frames_dir = work_dir / "pyframes"
    crop_dir = work_dir / "pycrop"
    frames_dir.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    converted_video = work_dir / "video.avi"
    audio_path = work_dir / "audio.wav"

    run_ffmpeg(
        [
            "-y",
            "-i",
            str(video_path),
            "-qscale:v",
            "2",
            "-async",
            "1",
            "-r",
            str(params.frame_rate),
            str(converted_video),
        ]
    )
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(converted_video),
            "-qscale:v",
            "2",
            "-threads",
            "1",
            "-f",
            "image2",
            str(frames_dir / "%06d.jpg"),
        ]
    )
    run_ffmpeg(
        [
            "-y",
            "-i",
            str(converted_video),
            "-ac",
            "1",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            str(audio_path),
        ]
    )

    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        raise RuntimeError(f"ffmpeg extracted no frames from {video_path}")

    faces = detect_faces_in_frames(detector, frame_files, params.facedet_scale)
    scenes = detect_scenes(converted_video)
    logger.debug("%s: %d frames, %d scenes", video_path.name, len(frame_files), len(scenes))

    alltracks: List[FaceTrack] = []
    for start, end in scenes:
        if end - start >= params.min_track:
            alltracks.extend(track_shot(faces[start:end], params))

    return [
        crop_video(
            track,
            frame_files,
            audio_path,
            crop_base=crop_dir / f"{idx:05d}",
            tmp_audio_path=work_dir / "track_audio.wav",
            params=params,
        )
        for idx, track in enumerate(alltracks)
    ]

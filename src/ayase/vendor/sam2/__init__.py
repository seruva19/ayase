# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Self-contained SAM 2.1 (Hiera-Large) backend for video mask propagation.

This vendored package provides the mask-propagation backend used by VMBench's
Temporal Coherence Score. It exposes a small, stateful public API built around
two paths:

* an image path (:meth:`Sam2Wrapper.image_mask`) that turns an ``xyxy`` box into
  a boolean mask on a single frame, used to seed masks from detector boxes;
* a video path (:meth:`Sam2Wrapper.init_state` / :meth:`Sam2Wrapper.add_new_mask`
  / :meth:`Sam2Wrapper.propagate`) that seeds masks on chosen frames and
  propagates them across an in-memory clip.

The model is built without Hydra/OmegaConf (the architecture is reproduced
explicitly in :mod:`ayase.vendor.sam2.build_sam`) and runs entirely on the pure
PyTorch path, so no compiled CUDA extension or config framework is required. The
Hiera-Large checkpoint is fetched from the Hugging Face Hub at load time.

Public API:
    load_sam2(models_dir, device) -> Sam2Wrapper
    Sam2Wrapper.image_mask(frame_rgb, box_xyxy) -> np.ndarray            # HxW bool
    Sam2Wrapper.init_state(frames_rgb_list) -> state
    Sam2Wrapper.add_new_mask(state, frame_idx, obj_id, mask_bool) -> None
    Sam2Wrapper.propagate(state, max_frame_num_to_track=None) -> iterator
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Tuple, Union

import numpy as np
import torch

__all__ = ["load_sam2", "Sam2Wrapper"]

# Hugging Face location of the SAM 2.1 Hiera-Large checkpoint used by VMBench.
_HF_REPO_ID = "GD-ML/VMBench"
_HF_CHECKPOINT_FILENAME = "sam2.1_hiera_large.pt"


def load_sam2(models_dir: str = "models", device: str = "cuda") -> "Sam2Wrapper":
    """Build the SAM 2.1 Hiera-L predictors and return a stateful wrapper.

    The checkpoint is downloaded from the Hugging Face Hub (repo
    ``GD-ML/VMBench``) and cached in ``models_dir`` on first use. A single model
    instance backs both the image and video paths.

    Args:
        models_dir: directory used as the Hugging Face download cache. If falsy,
            the default Hugging Face cache is used.
        device: torch device string (e.g. ``"cuda"`` or ``"cpu"``).

    Returns:
        A :class:`Sam2Wrapper` with the model loaded once.
    """
    from huggingface_hub import hf_hub_download

    from ayase.vendor.sam2.build_sam import build_sam2_video_predictor
    from ayase.vendor.sam2.sam2_image_predictor import SAM2ImagePredictor

    cache_dir = models_dir if models_dir else None
    checkpoint_path = hf_hub_download(
        repo_id=_HF_REPO_ID,
        filename=_HF_CHECKPOINT_FILENAME,
        cache_dir=cache_dir,
    )

    video_predictor, load_result = build_sam2_video_predictor(
        ckpt_path=checkpoint_path, device=device, mode="eval"
    )
    # The image predictor reuses the same underlying SAM2 model (the video
    # predictor is a SAM2Base subclass), so no second copy is loaded.
    image_predictor = SAM2ImagePredictor(video_predictor)

    return Sam2Wrapper(
        video_predictor=video_predictor,
        image_predictor=image_predictor,
        device=device,
        load_result=load_result,
    )


class Sam2Wrapper:
    """Stateful SAM 2.1 backend (model loaded once, reused for both paths)."""

    def __init__(self, video_predictor, image_predictor, device="cuda", load_result=None):
        self._video = video_predictor
        self._image = image_predictor
        self.device = device
        # (missing_keys, unexpected_keys) from the checkpoint load, for reporting.
        self.load_result = load_result

    # ------------------------------------------------------------------ image
    @torch.no_grad()
    def image_mask(
        self,
        frame_rgb: np.ndarray,
        box_xyxy: Union[np.ndarray, List[float], torch.Tensor],
    ) -> np.ndarray:
        """Return a single-frame boolean mask from an ``xyxy`` box prompt.

        Args:
            frame_rgb: one frame as an HxWx3 ``uint8`` RGB numpy array.
            box_xyxy: a length-4 box in pixel ``xyxy`` coordinates.

        Returns:
            An HxW boolean numpy array (the highest-confidence single mask).
        """
        h, w = frame_rgb.shape[:2]
        if box_xyxy is None:
            return np.zeros((h, w), dtype=bool)

        box = np.asarray(
            box_xyxy.detach().cpu().numpy()
            if isinstance(box_xyxy, torch.Tensor)
            else box_xyxy,
            dtype=np.float32,
        ).reshape(-1)
        if box.size != 4:
            raise ValueError(f"box_xyxy must have 4 elements, got {box.size}")

        self._image.set_image(frame_rgb)
        masks, _scores, _low_res = self._image.predict(
            box=box[None, :],
            multimask_output=False,
        )
        # masks: [1, H, W] (float 0/1) -> boolean HxW.
        return masks[0].astype(bool)

    # ------------------------------------------------------------------ video
    def init_state(self, frames_rgb_list: Union[List[np.ndarray], np.ndarray]):
        """Initialize a video inference state from in-memory frames.

        Args:
            frames_rgb_list: a list of HxWx3 ``uint8`` RGB numpy arrays, or a
                single ``[N, H, W, 3]`` ``uint8`` RGB array. No JPEG folder is
                needed; frames are normalized/resized exactly as the reference
                JPEG loader does.

        Returns:
            An opaque inference-state object to pass to :meth:`add_new_mask` and
            :meth:`propagate`.
        """
        return self._video.init_state(video_path=frames_rgb_list)

    def reset_state(self, state) -> None:
        """Clear all seeded masks/objects from an inference state so it can be
        re-seeded (used for segment-wise re-detection across keyframes)."""
        self._video.reset_state(state)

    def add_new_mask(self, state, frame_idx: int, obj_id: int, mask_bool: np.ndarray) -> None:
        """Seed a boolean mask for one object on one frame.

        Args:
            state: the inference state from :meth:`init_state`.
            frame_idx: index of the frame to seed on.
            obj_id: client-side object id (any hashable int).
            mask_bool: an HxW boolean (or 0/1) mask at the original frame
                resolution.
        """
        self._video.add_new_mask(
            inference_state=state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask_bool,
        )

    def propagate(
        self,
        state,
        max_frame_num_to_track: int = None,
        start_frame_idx: int = None,
        reverse: bool = False,
    ) -> Iterator[Tuple[int, Dict[int, np.ndarray]]]:
        """Propagate seeded masks across the clip.

        Yields per frame a ``(frame_idx, {obj_id: bool_mask})`` tuple, where each
        ``bool_mask`` is an HxW boolean numpy array at the original video
        resolution.

        Args:
            state: the inference state from :meth:`init_state`.
            max_frame_num_to_track: optional cap on how many frames to track
                forward from the start frame (``None`` tracks the whole clip).
            start_frame_idx: optional first frame to track from (defaults to the
                earliest seeded frame).
            reverse: track backward in time if ``True``.
        """
        for frame_idx, obj_ids, video_res_masks in self._video.propagate_in_video(
            inference_state=state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        ):
            # video_res_masks: [num_obj, 1, H, W] logits -> per-object bool mask.
            bool_masks = (video_res_masks[:, 0] > 0.0).cpu().numpy()
            per_obj = {
                obj_id: bool_masks[i].astype(bool)
                for i, obj_id in enumerate(obj_ids)
            }
            yield frame_idx, per_obj

"""VMBench Temporal Coherence (TCS) scoring helpers — vendored verbatim.

Source: github.com/AMAP-ML/VMBench, ``bench_utils/tcs_utils.py`` (Apache-2.0). TCS
penalises objects that implausibly vanish or emerge mid-clip. Subject masks are
propagated across frames and tracked; an object that disappears (or appears) is
counted as an *error* only when it is NOT explained by a benign cause — leaving
the frame edge, shrinking below a size floor, or a tracker detection error. The
score is the fraction of objects free of vanish/emerge errors:

    vanish_score = (objects_count - disappear_objects_count) / objects_count
    emerge_score = (objects_count - appear_objects_count) / objects_count
    temporal_coherence_score = (vanish_score + emerge_score) / 2

Only the pure-torch/NumPy classification of vanish/emerge events is kept here; the
grounding, SAM2 mask propagation and point tracking live in the ayase module. The
functions below are copied verbatim (thresholds and control flow unchanged).
"""

import numpy as np
import torch


def get_disappear_objects(tracking_result):
    result = []
    first_appearances = {}

    for i, current_dict in enumerate(tracking_result):
        for key, value in current_dict.items():
            if key not in first_appearances:
                first_appearances[key] = {
                    'frame': i,
                    'mask': np.array(value['mask']) if 'mask' in value else None
                }

    for i in range(len(tracking_result) - 1):
        dict1 = tracking_result[i]
        dict2 = tracking_result[i + 1]

        disappeared_keys = set(dict1.keys()) - set(dict2.keys())

        for key in disappeared_keys:
            disappeared_object_info = {
                'object_id': key,
                'mask': first_appearances[key]['mask'],
                'first_appearance': first_appearances[key]['frame'],
                'last_frame': i
            }

            result.append(disappeared_object_info)

    return result


def is_edge_vanish(pred_tracks, pred_visibility, start, width=720, height=480, visibility_ratio=0.8, point_ratio=0.5):
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    indices = indices[indices > start]
    selected_frames = pred_tracks[0, indices]

    left_mask = selected_frames[:, :, 0] < 0
    right_mask = selected_frames[:, :, 0] > width
    top_mask = selected_frames[:, :, 1] < 0
    bottom_mask = selected_frames[:, :, 1] > height
    out_of_screen_mask = left_mask | right_mask | top_mask | bottom_mask
    out_of_screen_ratio = out_of_screen_mask.float().mean(dim=1)
    valid_frames_mask = out_of_screen_ratio >= point_ratio
    vanish_indices = indices[valid_frames_mask]

    if len(vanish_indices) > 0:
        edge_vanish = True
    else:
        edge_vanish = False

    return edge_vanish


def is_small_vanish(pred_tracks, pred_visibility, start, width=720, height=480,
                    visibility_ratio=-1, point_ratio=0.8, size_threshold=0.07):
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    indices = indices[indices > start]
    pred_tracks = pred_tracks[0, indices]
    small_object_frames = []

    for i, frame in enumerate(pred_tracks):
        left_mask = frame[:, 0] > 0
        right_mask = frame[:, 0] < width
        top_mask = frame[:, 1] > 0
        bottom_mask = frame[:, 1] < height
        in_screen_mask = left_mask & right_mask & top_mask & bottom_mask

        valid_points = frame[in_screen_mask]
        if in_screen_mask.float().mean(dim=0) >= point_ratio and valid_points.shape[0] > 1:
            q_low = torch.quantile(valid_points, 0.1, dim=0)
            q_high = torch.quantile(valid_points, 0.9, dim=0)
            object_width = (q_high[0] - q_low[0]) / width
            object_height = (q_high[1] - q_low[1]) / height
            object_size = max(object_width, object_height)

            if object_size < size_threshold:
                small_object_frames.append(i)

    if len(small_object_frames) > 0:
        small_vanish = True
    else:
        small_vanish = False

    return small_vanish


def is_vanish_detect_error(pred_tracks, pred_visibility, start, visibility_ratio=1.0):
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    indices = indices[indices > start]

    if len(indices) == 0:
        detect_error = True
    else:
        detect_error = False

    return detect_error


def get_appear_objects(dict_list):
    result = []
    first_appearances = {}

    for i, current_dict in enumerate(dict_list):
        for key, value in current_dict.items():
            if key not in first_appearances:
                first_appearances[key] = {
                    'frame': i,
                    'mask': np.array(value['mask']) if 'mask' in value else None
                }

    for i in range(1, len(dict_list)):
        dict1 = dict_list[i - 1]
        dict2 = dict_list[i]

        appeared_keys = set(dict2.keys()) - set(dict1.keys())

        for key in appeared_keys:
            appeared_object_info = {
                'object_id': key,
                'mask': first_appearances[key]['mask'],
                'first_appearance': first_appearances[key]['frame'],
            }

            result.append(appeared_object_info)

    return result


def is_edge_emerge(pred_tracks, pred_visibility, start, width=720, height=480, visibility_ratio=0.85, point_ratio=0.5):
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    indices = indices[indices < start]
    selected_frames = pred_tracks[0, indices]

    left_mask = selected_frames[:, :, 0] < 0
    right_mask = selected_frames[:, :, 0] > width
    top_mask = selected_frames[:, :, 1] < 0
    bottom_mask = selected_frames[:, :, 1] > height
    out_of_screen_mask = left_mask | right_mask | top_mask | bottom_mask
    out_of_screen_ratio = out_of_screen_mask.float().mean(dim=1)
    valid_frames_mask = out_of_screen_ratio >= point_ratio
    emerge_indices = indices[valid_frames_mask]

    if len(emerge_indices) > 0:
        edge_emerge = True
    else:
        edge_emerge = False

    return edge_emerge


def is_small_emerge(pred_tracks, pred_visibility, start, width=720, height=480,
                    visibility_ratio=-1, point_ratio=0.8, size_threshold=0.03):
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    indices = indices[indices < start]
    pred_tracks = pred_tracks[0, indices]
    small_object_frames = []

    for i, frame in enumerate(pred_tracks):
        left_mask = frame[:, 0] > 0
        right_mask = frame[:, 0] < width
        top_mask = frame[:, 1] > 0
        bottom_mask = frame[:, 1] < height
        in_screen_mask = left_mask & right_mask & top_mask & bottom_mask

        valid_points = frame[in_screen_mask]
        if in_screen_mask.float().mean(dim=0) >= point_ratio and valid_points.shape[0] > 1:
            q_low = torch.quantile(valid_points, 0.1, dim=0)
            q_high = torch.quantile(valid_points, 0.9, dim=0)
            object_width = (q_high[0] - q_low[0]) / width
            object_height = (q_high[1] - q_low[1]) / height
            object_size = max(object_width, object_height)

            if object_size < size_threshold:
                small_object_frames.append(i)

    if len(small_object_frames) > 0:
        small_emerge = True
    else:
        small_emerge = False

    return small_emerge


def is_emerge_detect_error(pred_tracks, pred_visibility, start, visibility_ratio=0.8):
    false_ratio = (~pred_visibility).float().mean(dim=2).squeeze(0)
    indices = torch.where(false_ratio >= visibility_ratio)[0]
    indices = indices[indices < start]

    if len(indices) == 0:
        detect_error = True
    else:
        detect_error = False

    return detect_error

"""VMBench Perceptible Amplitude (PAS) scoring — vendored verbatim.

Source: github.com/AMAP-ML/VMBench, ``perceptible_amplitude_score.py`` (Apache-2.0).
PAS measures how much the *subject* moves relative to camera/background motion:
grid points are tracked over the clip, their per-point displacements are summed
over time and normalised by the frame diagonal, and the subject's mean amplitude
has the background (camera) amplitude subtracted off. Only the pure-torch motion
aggregation and the subject/background combination are kept here; the grounding,
masking and point-tracking backends live in the ayase module.

    motion_degree = mean_over_points( sum_over_time( ||p_t - p_{t-1}|| / diagonal ) )
    PAS = subject_motion_degree - background_motion_degree   (when subject moves more)
"""


def calculate_motion_degree(keypoints, video_width, video_height):
    """Mean per-point trajectory length, normalised by the frame diagonal.

    ``keypoints`` is a ``[1, T, N, 2]`` tensor of tracked point positions (pixel
    coordinates) — batch 1, T frames, N points, xy. Returns a ``[1]`` tensor with
    the clip's mean normalised motion amplitude.
    """
    import torch

    diagonal = torch.sqrt(
        torch.tensor(video_width ** 2 + video_height ** 2, dtype=torch.float32)
    )
    distances = torch.norm(keypoints[:, 1:] - keypoints[:, :-1], dim=3)  # [1, T-1, N]
    normalized_distances = distances / diagonal
    total_normalized_distances = torch.sum(normalized_distances, dim=1)  # [1, N]
    motion_amplitudes = torch.mean(total_normalized_distances, dim=1)    # [1]
    return motion_amplitudes


def combine_subject_background(subject_motion_degree, background_motion_degree,
                              subject_detected):
    """VMBench's final PAS combination.

    With a detected subject, the camera/background amplitude is subtracted from
    the subject amplitude when the subject moves more; the result is the PAS. If
    the subject amplitude is unavailable (NaN) or no subject was detected, PAS
    falls back to the background amplitude.
    """
    import math

    bg = float(background_motion_degree)
    if not subject_detected:
        return bg
    subj = float(subject_motion_degree)
    if math.isnan(subj):
        return bg
    if subj > bg:
        subj = subj - bg
    return subj

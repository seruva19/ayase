"""Velocity/acceleration histogram features from the official FVMD runtime.

Modified for Ayase from upstream ``fvmd/extract_motion_features.py``: CLI,
filesystem, JSON, and Loguru paths were removed; the dense 1-D histogram
implementation is preserved as a library API.
"""

import numpy as np


def cut_subcube(vectors: np.ndarray, cell_size: int = 5, cube_frames: int = 4):
    """Split a ``(B, S, H, W, 2)`` vector field into local volumes."""
    batch, frames, height, width, _ = vectors.shape
    cells_h = height // cell_size
    cells_w = width // cell_size
    cells_t = frames // cube_frames
    vectors = vectors[
        :, : cells_t * cube_frames, : cells_h * cell_size, : cells_w * cell_size, :
    ]
    vectors = vectors.reshape(
        batch, cells_t, cube_frames, cells_h, cell_size, cells_w, cell_size, 2
    )
    vectors = vectors.transpose(0, 1, 3, 5, 2, 4, 6, 7)
    vectors = vectors.reshape(-1, cube_frames, cell_size, cell_size, 2)
    return vectors, cells_t, cells_h, cells_w


def count_subcube_hist(
    vector_cell: np.ndarray, angle_bins: int = 8, magnitude_bins: int = 256
) -> np.ndarray:
    """Compute one dense 1-D magnitude-weighted orientation histogram."""
    histogram = np.zeros(angle_bins)
    frames, height, width, _ = vector_cell.shape
    angles = np.arctan2(vector_cell[:, :, :, 0], vector_cell[:, :, :, 1])
    angle_indices = (angles + np.pi) // (2 * np.pi / angle_bins)
    angle_indices = np.clip(angle_indices, 0, angle_bins - 1)

    magnitudes = np.linalg.norm(vector_cell, axis=3)
    magnitudes = np.clip(magnitudes, 0, magnitude_bins - 1)
    magnitudes = np.log2(magnitudes + 1)
    magnitudes = np.clip(magnitudes, 0, int(np.log2(magnitude_bins)))
    magnitudes = np.ceil(magnitudes) / np.log2(magnitude_bins)

    for frame in range(frames):
        for row in range(height):
            for col in range(width):
                histogram[int(angle_indices[frame, row, col])] += magnitudes[frame, row, col]
    return histogram


def calc_hist(
    vectors: np.ndarray, cell_size: int = 5, angle_bins: int = 8, cube_frames: int = 4
) -> np.ndarray:
    """Build official FVMD dense 1-D histograms from ``(B, S, N, 2)`` fields."""
    batch, frames, point_count, dims = vectors.shape
    if dims != 2:
        raise ValueError("vectors must have shape (B, S, N, 2)")
    side = int(np.sqrt(point_count).round())
    if side * side != point_count:
        raise ValueError("point_count must be a perfect square")
    vectors = vectors.reshape(batch, frames, side, side, 2)
    vectors, cells_t, cells_h, cells_w = cut_subcube(vectors, cell_size, cube_frames)
    if vectors.shape[0] == 0:
        raise ValueError("field is too small for the requested histogram volume")
    histograms = np.stack(
        [count_subcube_hist(cell, angle_bins=angle_bins) for cell in vectors], axis=0
    )
    return histograms.reshape(batch, cells_t, cells_h, cells_w, angle_bins)


def combine_motion_histograms(velocity: np.ndarray, acceleration: np.ndarray) -> np.ndarray:
    """Return the paper-default flattened velocity+acceleration feature."""
    velocity_hist = calc_hist(velocity)
    acceleration_hist = calc_hist(acceleration)
    batch = velocity_hist.shape[0]
    return np.concatenate(
        [velocity_hist.reshape(batch, -1), acceleration_hist.reshape(batch, -1)], axis=1
    )

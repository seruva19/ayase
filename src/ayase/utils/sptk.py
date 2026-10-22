"""Small NumPy ports of SPTK transforms used by Ayase.

The frequency-transform recurrence is derived from SPTK 3's ``freqt``.
Copyright (c) 2015 Ryuichi Yamamoto; used under the MIT License.  The original
implementation is distributed with pysptk: https://github.com/r9y9/pysptk
"""

from __future__ import annotations

import numpy as np


def freqt(cepstrum: np.ndarray, order: int, alpha: float) -> np.ndarray:
    """Apply SPTK's all-pass frequency transformation along the last axis."""

    values = np.asarray(cepstrum, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("cepstrum must have a non-empty final axis")
    if order < 0:
        raise ValueError("order must be non-negative")

    flat = values.reshape((-1, values.shape[-1]))
    output = np.zeros((flat.shape[0], order + 1), dtype=np.float64)
    beta = 1.0 - alpha * alpha
    # Vectorize independent frames, while retaining SPTK's sequential order
    # recurrence within each coefficient update.
    for coefficient in flat.T[::-1]:
        previous = output.copy()
        output[:, 0] = coefficient + alpha * previous[:, 0]
        if order >= 1:
            output[:, 1] = beta * previous[:, 0] + alpha * previous[:, 1]
        for index in range(2, order + 1):
            output[:, index] = previous[:, index - 1] + alpha * (
                previous[:, index] - output[:, index - 1]
            )
    return output.reshape(values.shape[:-1] + (order + 1,))


def sp2mc(power_spectrum: np.ndarray, order: int, alpha: float) -> np.ndarray:
    """Convert positive power spectra to mel-cepstra like ``pysptk.sp2mc``."""

    spectrum = np.asarray(power_spectrum)
    if spectrum.ndim < 1 or spectrum.shape[-1] < 2:
        raise ValueError("power spectrum must have at least two frequency bins")
    if not np.issubdtype(spectrum.dtype, np.floating):
        spectrum = spectrum.astype(np.float64)
    if not np.isfinite(spectrum).all() or np.any(spectrum <= 0.0):
        raise ValueError("power spectrum values must be finite and positive")

    cepstrum = np.fft.irfft(np.log(spectrum), axis=-1)
    cepstrum[..., 0] /= 2.0
    result = freqt(cepstrum, order, alpha)
    # pysptk restores the input floating dtype after its float64 C calculation.
    return result.astype(spectrum.dtype, copy=False)


__all__ = ["freqt", "sp2mc"]

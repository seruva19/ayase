"""Conformance tests for the internal SPTK transforms."""

import numpy as np


def test_freqt_matches_sptk3_golden_vector():
    from ayase.utils.sptk import freqt

    source = np.array([0.5, -0.25, 0.125, 0.75, -0.5], dtype=np.float64)
    expected = np.array(
        [
            0.45705751999999999,
            0.085427086400000005,
            0.33095443568000005,
            -0.34335063035519997,
            0.021094635796671923,
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(freqt(source, 4, 0.42), expected, rtol=0.0, atol=1e-12)


def test_sp2mc_preserves_batch_shape_and_float_dtype():
    from ayase.utils.sptk import sp2mc

    spectrum = np.stack([np.linspace(0.25, 2.0, 513), np.linspace(2.0, 0.25, 513)]).astype(
        np.float32
    )
    result = sp2mc(spectrum, 23, 0.42)
    assert result.shape == (2, 24)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_freqt_batch_matches_independent_sptk_recurrence():
    from ayase.utils.sptk import freqt

    rng = np.random.default_rng(741)
    source = rng.normal(size=(7, 33))

    def reference(row, order, alpha):
        output = np.zeros(order + 1)
        beta = 1.0 - alpha * alpha
        for coefficient in row[::-1]:
            previous = output.copy()
            output[0] = coefficient + alpha * previous[0]
            output[1] = beta * previous[0] + alpha * previous[1]
            for index in range(2, order + 1):
                output[index] = previous[index - 1] + alpha * (previous[index] - output[index - 1])
        return output

    expected = np.stack([reference(row, 12, 0.42) for row in source])
    np.testing.assert_array_equal(freqt(source, 12, 0.42), expected)


def test_sp2mc_rejects_nonpositive_spectrum():
    import pytest
    from ayase.utils.sptk import sp2mc

    with pytest.raises(ValueError, match="positive"):
        sp2mc(np.array([1.0, 0.0, 2.0]), 2, 0.42)

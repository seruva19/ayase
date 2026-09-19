"""Fréchet distance helpers used by the official FVMD runtime.

Adapted from the Apache-2.0 FVMD ``frechet_distance.py``, itself adapted from
pytorch-fid/TTUR. Modified for Ayase to remove CLI, Loguru, tqdm, and stdout
side effects while retaining the released numerical calculation.

Copyright 2018 Institute of Bioinformatics, JKU Linz
"""

import numpy as np
from scipy import linalg


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-5):
    """Compute squared Fréchet distance between two multivariate Gaussians."""
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    if mu1.shape != mu2.shape:
        raise ValueError("mean vectors have different lengths")
    if sigma1.shape != sigma2.shape:
        raise ValueError("covariances have different dimensions")

    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    covmean, _ = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError(f"imaginary component {np.max(np.abs(covmean.imag))}")
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def calculate_activation_statistics(vectors):
    """Estimate empirical mean and covariance from batched feature vectors."""
    batch = vectors.shape[0]
    activations = vectors.reshape(batch, -1)
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def calculate_fd_given_vectors(features_1, features_2):
    """Compute Fréchet distance between two batches of feature vectors."""
    mean_1, covariance_1 = calculate_activation_statistics(features_1)
    mean_2, covariance_2 = calculate_activation_statistics(features_2)
    return calculate_frechet_distance(mean_1, covariance_1, mean_2, covariance_2)

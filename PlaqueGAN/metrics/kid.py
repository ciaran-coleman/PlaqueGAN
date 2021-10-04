# https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py

import numpy as np
import torch
from sklearn.metrics.pairwise import polynomial_kernel
from scipy import linalg

from tqdm import tqdm

def _sqn(arr):
    flat = np.ravel(arr)
    return flat.dot(flat)

def mmd2(K_XX, K_XY, K_YY, unit_diagonal=False,
                       mmd_est='unbiased'):

    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if unit_diagonal:
        diag_X = diag_Y = 1
        sum_diag_X = sum_diag_Y = m
        sum_diag2_X = sum_diag2_Y = m
    else:
        diag_X = np.diagonal(K_XX)
        diag_Y = np.diagonal(K_YY)

        sum_diag_X = diag_X.sum()
        sum_diag_Y = diag_Y.sum()

        sum_diag2_X = _sqn(diag_X)
        sum_diag2_Y = _sqn(diag_Y)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)
    K_XY_sums_1 = K_XY.sum(axis=1)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    if mmd_est == 'biased':
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
                + (Kt_YY_sum + sum_diag_Y) / (m * m)
                - 2 * K_XY_sum / (m * m))
    else:
        assert mmd_est in {'unbiased', 'u-statistic'}
        mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m-1))
        if mmd_est == 'unbiased':
            mmd2 -= 2 * K_XY_sum / (m * m)
        else:
            mmd2 -= 2 * (K_XY_sum - np.trace(K_XY)) / (m * (m-1))

    return mmd2

def compute_kid(real_features, fake_features, n_subsets=10, max_subset_size=1000):
    m = min(min(real_features.shape[0], fake_features.shape[0]), max_subset_size)
    mmds = np.zeros(n_subsets)

    choice = np.random.choice

    # with tqdm(range(n_subsets), desc='MMD', file=output) as bar:
    #     for i in bar:
    #         real = real_features[choice(len(real_features), m, replace=False)]
    #         fake = fake_features[choice(len(fake_features), m, replace=False)]
    #         o = polynomial_mmd(real, fake)
    #         mmds[i] = o

    for i in range(n_subsets):
        real = real_features[choice(len(real_features), m, replace=False)]
        fake = fake_features[choice(len(fake_features), m, replace=False)]
        o = polynomial_mmd(real, fake)
        mmds[i] = o
    return {'mean':np.mean(mmds),
            'std': np.std(mmds)}

def polynomial_mmd(real_features, fake_features, degree=3, gamma=None, coef0=1):
    X = fake_features
    Y = real_features

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return mmd2(K_XX, K_XY, K_YY)

# def compute_kid(real_features, fake_features):

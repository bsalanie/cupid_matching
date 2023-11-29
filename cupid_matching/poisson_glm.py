"""Estimates the semilinear Choo and Siow homoskedastic (2006) model
using Poisson GLM.
"""

from math import sqrt

import numpy as np
import scipy.linalg as spla
from sklearn import linear_model

from cupid_matching.matching_utils import Matching, VarianceMatching, variance_muhat
from cupid_matching.poisson_glm_utils import PoissonGLMResults, prepare_data
from cupid_matching.utils import make_XY_K_mat


def _stderrs_u(
    varcov_gamma: np.ndarray,
    n_norm: np.ndarray,
    var_muhat_norm: VarianceMatching,
    A_inv_Z: np.ndarray,
    X: int,
    Y: int,
):
    XY = X * Y
    u_std = np.zeros(X)
    var_allmus_norm = var_muhat_norm.var_allmus
    var_munm_norm = var_muhat_norm.var_munm
    var_n_norm = var_munm_norm[XY : (XY + X), XY : (XY + X)]

    ix = XY
    for x in range(X):
        n_norm_x = n_norm[x]
        A_inv_Zx = A_inv_Z[x, :]
        var_log_nx = var_n_norm[x, x] / n_norm_x / n_norm_x
        slice_x = slice(x * Y, (x + 1) * Y)
        covar_nx = var_allmus_norm[:, ix] + np.sum(var_allmus_norm[:, slice_x], 1)
        cov_a_lognx = (A_inv_Zx @ covar_nx) / n_norm_x
        ux_var = varcov_gamma[x, x] + var_log_nx + 2.0 * cov_a_lognx
        u_std[x] = sqrt(ux_var)
        ix += 1
    return u_std


def _stderrs_v(
    varcov_gamma: np.ndarray,
    m_norm: np.ndarray,
    var_muhat_norm: VarianceMatching,
    A_inv_Z: np.ndarray,
    X: int,
    Y: int,
):
    XY = X * Y
    v_std = np.zeros(Y)
    var_allmus_norm = var_muhat_norm.var_allmus
    var_munm_norm = var_muhat_norm.var_munm
    var_m_norm = var_munm_norm[(XY + X) :, (XY + X) :]

    iy, jy = X, XY + X
    for y in range(Y):
        m_norm_y = m_norm[y]
        A_inv_Zy = A_inv_Z[iy, :]
        var_log_my = var_m_norm[y, y] / m_norm_y / m_norm_y
        slice_y = slice(y, XY, Y)
        covar_b_my = var_allmus_norm[:, jy] + np.sum(var_allmus_norm[:, slice_y], 1)
        cov_b_logmy = (A_inv_Zy @ covar_b_my) / m_norm_y
        vy_var = varcov_gamma[iy, iy] + var_log_my + 2.0 * cov_b_logmy
        v_std[y] = sqrt(vy_var)
        iy += 1
        jy += 1
    return v_std


def _stderrs_u_no_singles(
    varcov_gamma: np.ndarray,
    n_norm: np.ndarray,
    var_muhat_norm: VarianceMatching,
    A_inv_Z: np.ndarray,
    X: int,
    Y: int,
):
    XY = X * Y
    var_muxy_norm = var_muhat_norm.var_allmus[:XY, :XY]
    var_munm_norm = var_muhat_norm.var_munm
    var_n_norm = var_munm_norm[XY : (XY + X), XY : (XY + X)]
    n_norm1 = n_norm[0]
    var_log_n1 = var_n_norm[0, 0] / n_norm1 / n_norm1

    u_std = np.zeros(X)
    u_std[0] = 0.0
    slice_1 = slice(0, Y)
    for x in range(1, X):
        A_inv_Zx = A_inv_Z[x, :]
        n_norm_x = n_norm[x]
        var_log_nx = var_n_norm[x, x] / n_norm_x / n_norm_x
        slice_x = slice(x * Y, (x + 1) * Y)
        covar_nx = np.sum(var_muxy_norm[:, slice_x], 1)
        covar_n1 = np.sum(var_muxy_norm[:, slice_1], 1)
        covar_nx_n1 = var_n_norm[x, 0]
        cov_a_lognx = (A_inv_Zx @ covar_nx) / n_norm_x
        cov_a_logn1 = (A_inv_Zx @ covar_n1) / n_norm_x
        cov_lognx_logn1 = covar_nx_n1 / n_norm_x / n_norm1
        ux_var = (
            varcov_gamma[x, x]
            + var_log_nx
            + var_log_n1
            + 2.0 * cov_a_lognx
            - 2.0 * cov_a_logn1
            - 2.0 * cov_lognx_logn1
        )
        u_std[x - 1] = sqrt(ux_var)
    return u_std


def _stderrs_v_no_singles(
    varcov_gamma: np.ndarray,
    m_norm: np.ndarray,
    n_norm: np.ndarray,
    var_muhat_norm: VarianceMatching,
    A_inv_Z: np.ndarray,
    X: int,
    Y: int,
):
    XY = X * Y
    var_muxy_norm = var_muhat_norm.var_allmus[:XY, :XY]
    var_munm_norm = var_muhat_norm.var_munm
    var_n_norm = var_munm_norm[XY : (XY + X), XY : (XY + X)]
    var_m_norm = var_munm_norm[(XY + X) :, (XY + X) :]
    n_norm1 = n_norm[0]
    var_log_n1 = var_n_norm[0, 0] / n_norm1 / n_norm1

    v_std = np.zeros(Y)
    iy, jy = X, XY + X
    slice_1 = slice(0, Y)
    for y in range(Y):
        A_inv_Zy = A_inv_Z[iy, :]
        m_norm_y = m_norm[y]
        var_log_my = var_m_norm[y, y] / m_norm_y / m_norm_y
        slice_y = slice(y, XY, Y)
        covar_my = np.sum(var_muxy_norm[:, slice_y], 1)
        cov_b_logmy = (A_inv_Zy @ covar_my) / m_norm_y
        covar_n1 = np.sum(var_muxy_norm[:, slice_1], 1)
        cov_b_logn1 = (A_inv_Zy @ covar_n1) / n_norm1
        covar_my_n1 = var_munm_norm[jy, XY]
        cov_logmy_logn1 = covar_my_n1 / m_norm_y / n_norm1
        vy_var = (
            varcov_gamma[iy, iy]
            + var_log_my
            + var_log_n1
            + 2.0 * cov_b_logmy
            + 2.0 * cov_b_logn1
            + 2.0 * cov_logmy_logn1
        )
        v_std[y] = sqrt(vy_var)

    return v_std


def choo_siow_poisson_glm(
    muhat: Matching,
    phi_bases: np.ndarray,
    no_singles: bool = False,
    tol: float | None = 1e-12,
    max_iter: int | None = 10000,
    verbose: int | None = 1,
) -> PoissonGLMResults:
    """Estimates the semilinear Choo and Siow homoskedastic (2006) model
        using Poisson GLM.

    Args:
        muhat: the observed Matching
        phi_bases: an (X, Y, K) array of bases
        no_singles: if True, we do not observe the singles
        tol: tolerance level for `linear_model.PoissonRegressor.fit`
        max_iter: maximum number of iterations
            for `linear_model.PoissonRegressor.fit`
        verbose: defines how much output we want (0 = least)

    Returns:
        a `PoissonGLMResults` instance

    Example:
        ```py
        n_households = 1e6
        X, Y, K = 4, 3, 6
        # we setup a quadratic set of basis functions
        phi_bases = np.zeros((X, Y, K))
        phi_bases[:, :, 0] = 1
        for x in range(X):
            phi_bases[x, :, 1] = x
            phi_bases[x, :, 3] = x * x
            for y in range(Y):
                phi_bases[x, y, 4] = x * y
        for y in range(Y):
            phi_bases[:, y, 2] = y
            phi_bases[:, y, 5] = y * y

        lambda_true = np.random.randn(K)
        phi_bases = np.random.randn(X, Y, K)
        Phi = phi_bases @ lambda_true

        # we simulate a Choo and Siow sample from a population
        #  with equal numbers of men and women of each type
        n = np.ones(X)
        m = np.ones(Y)
        choo_siow_instance = ChooSiowPrimitives(Phi, n, m)
        mus_sim = choo_siow_instance.simulate(n_households)
        muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()

        results = choo_siow_poisson_glm(mus_sim, phi_bases)

        # compare true and estimated parameters
        results.print_results(
            lambda_true,
            u_true=-np.log(mux0_sim / n_sim),
            v_true=-np.log(mu0y_sim / m_sim)
        )
        ```

    """
    X, Y, K = phi_bases.shape
    XY = X * Y

    # the vector of weights for the Poisson regression
    w = (
        2 * np.ones(XY)
        if no_singles
        else np.concatenate((2 * np.ones(XY), np.ones(X + Y)))
    )
    # reshape the bases
    phi_mat = make_XY_K_mat(phi_bases)

    id_X = np.eye(X)
    id_Y = np.eye(Y)
    ones_X = np.ones((X, 1))
    ones_Y = np.ones((Y, 1))
    if no_singles:
        Z_unweighted = np.hstack(
            [-np.kron(id_X, ones_Y), -np.kron(ones_X, id_Y), phi_mat]
        )
        # we need to normalize u_1 = 0, so we delete the first column
        Z_unweighted = Z_unweighted[:, 1:]
    else:
        zeros_XK = np.zeros((X, K))
        zeros_YK = np.zeros((Y, K))
        zeros_XY = np.zeros((X, Y))
        zeros_YX = np.zeros((Y, X))
        Z_unweighted = np.vstack(
            [
                np.hstack([-np.kron(id_X, ones_Y), -np.kron(ones_X, id_Y), phi_mat]),
                np.hstack([-id_X, zeros_XY, zeros_XK]),
                np.hstack([zeros_YX, -id_Y, zeros_YK]),
            ]
        )
    Z = Z_unweighted / w.reshape((-1, 1))

    var_muhat = variance_muhat(muhat)
    (
        muhat_norm,
        var_muhat_norm,
        n_households,
        n_individuals,
    ) = prepare_data(muhat, var_muhat, no_singles=no_singles)

    clf = linear_model.PoissonRegressor(
        fit_intercept=False,
        tol=tol,
        verbose=verbose,
        alpha=0,
        max_iter=max_iter,
    )
    if no_singles:
        muxyhat_norm = muhat_norm[:XY]
        clf.fit(Z, muxyhat_norm, sample_weight=w)
    else:
        clf.fit(Z, muhat_norm, sample_weight=w)
    gamma_est = clf.coef_

    # we compute_ the variance-covariance of the estimator
    var_allmus_norm = var_muhat_norm.var_allmus
    var_norm = var_allmus_norm[:XY, :XY] if no_singles else var_allmus_norm
    nr, nc = Z.shape
    exp_Zg = np.exp(Z @ gamma_est).reshape(nr)
    A_hat = np.zeros((nc, nc))
    B_hat = np.zeros((nc, nc))
    for i in range(nr):
        Zi = Z[i, :]
        wi = w[i]
        A_hat += wi * exp_Zg[i] * np.outer(Zi, Zi)
        for j in range(nr):
            Zj = Z[j, :]
            B_hat += wi * w[j] * var_norm[i, j] * np.outer(Zi, Zj)

    A_inv = spla.inv(A_hat)
    varcov_gamma = A_inv @ B_hat @ A_inv
    stderrs_gamma = np.sqrt(np.diag(varcov_gamma))

    beta_est = gamma_est[-K:]
    varcov_beta = varcov_gamma[-K:, -K:]
    beta_std = stderrs_gamma[-K:]
    Phi_est = phi_bases @ beta_est

    # we correct for the effect of the normalization
    _, _, _, n, m = muhat.unpack()
    n_norm = n / n_individuals
    m_norm = m / n_individuals
    if no_singles:
        u_est = gamma_est[: (X - 1)]
        v_est = gamma_est[(X - 1) : -K]
        # normalize u_1 = 0
        n_0 = n_norm[0]
        u_est = np.concatenate((np.zeros(1), u_est + np.log(n_norm[1:] / n_0)))
        v_est += np.log(m_norm * n_0)
    else:
        u_est = gamma_est[:X] + np.log(n_norm)
        v_est = gamma_est[X:-K] + np.log(m_norm)

    # since u and v are translated from gamma we need to adjust the estimated stderrs
    A_inv_Z = A_inv @ Z_unweighted.T
    if no_singles:
        u_std = _stderrs_u_no_singles(
            varcov_gamma, n_norm, var_muhat_norm, A_inv_Z, X, Y
        )
        v_std = _stderrs_v_no_singles(
            varcov_gamma, m_norm, n_norm, var_muhat_norm, A_inv_Z, X, Y
        )
    else:
        u_std = _stderrs_u(varcov_gamma, n_norm, var_muhat_norm, A_inv_Z, X, Y)
        v_std = _stderrs_v(varcov_gamma, m_norm, var_muhat_norm, A_inv_Z, X, Y)

    results = PoissonGLMResults(
        X=X,
        Y=Y,
        K=K,
        number_households=n_households,
        number_individuals=n_individuals,
        estimated_gamma=gamma_est,
        estimated_Phi=Phi_est,
        estimated_beta=beta_est,
        estimated_u=u_est,
        estimated_v=v_est,
        varcov_gamma=varcov_gamma,
        varcov_beta=varcov_beta,
        stderrs_gamma=stderrs_gamma,
        stderrs_beta=beta_std,
        stderrs_u=u_std,
        stderrs_v=v_std,
    )

    return results

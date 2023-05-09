"""The components of the derivative of the entropy
for the Choo and Siow fully heteroskedastic model.

We normalize the standard error for X=1 at 1, and we estimate the standard errors
for all other types on the X side and for all types on the Y side.
"""
from typing import cast

import numpy as np
from bs_python_utils.bsnputils import ThreeArrays, TwoArrays

from cupid_matching.entropy import EntropyFunctions, EntropyHessians
from cupid_matching.matching_utils import Matching


def e0_choo_siow_heteroskedastic(muhat: Matching) -> np.ndarray:
    """Returns the values of the parameter-independent part $e_0$
    for the Choo and Siow heteroskedastic model;
    we normalized $\\sigma_1=1$

    Args:
        muhat: a Matching

    Returns:
        the (X,Y) matrix of the parameter-independent part
        of the first derivative of the entropy
    """
    muxy, mux0, *_ = muhat.unpack()
    mu1y = muxy[0, :]
    mu10 = mux0[0]
    e0_vals = np.zeros_like(muxy)
    e0_vals[0, :] = -np.log(mu1y / mu10)
    return cast(np.ndarray, e0_vals)


def e0_derivative_mu_heteroskedastic(
    muhat: Matching,
) -> ThreeArrays:
    """Returns the derivatives of the parameter-independent part $e_0$
    wrt $\\mu$ for the Choo and Siow heteroskedastic model;
    we normalized $\\sigma_1=1$

    Args:
        muhat: a Matching

    Returns:
        the parameter-independent part of the hessian of the entropy
        wrt $(\\mu, \\mu)$.
    """
    muxy, mux0, *_ = muhat.unpack()
    X, Y = muxy.shape
    mu1y = muxy[0, :]
    mu10 = mux0[0]
    hess_x = np.zeros((X, Y, Y))
    hess_y = np.zeros((X, Y, X))
    hess_xy = np.zeros((X, Y))
    der_log1y = 1.0 / mu1y
    der_log10 = 1.0 / mu10
    for y in range(Y):
        hess_x[0, y, :] = -der_log10
        hess_xy[0, y] = -der_log1y[y] - der_log10
    return hess_x, hess_y, hess_xy


def e0_derivative_r_heteroskedastic(
    muhat: Matching,
) -> TwoArrays:
    """Returns the derivatives of the parameter-independent part $e_0$
    wrt $r$ for the Choo and Siow heteroskedastic model;
    we normalized $\\sigma_1=1$

    Args:
        muhat: a Matching

    Returns:
        the parameter-independent part of the hessian of the entropy
        wrt $(\\mu,r)$.
    """
    muxy, mux0, *_ = muhat.unpack()
    X, Y = muxy.shape
    mu10 = mux0[0]
    hess_n = np.zeros((X, Y))
    hess_m = np.zeros((X, Y))
    der_log10 = 1.0 / mu10
    for y in range(Y):
        hess_n[0, y] = der_log10
    return hess_n, hess_m


e0_derivative_choo_siow_heteroskedastic = (
    e0_derivative_mu_heteroskedastic,
    e0_derivative_r_heteroskedastic,
)


def e_choo_siow_heteroskedastic(muhat: Matching) -> np.ndarray:
    """Returns the values of the parameter-dependent part  $e$
    for the Choo and Siow heteroskedastic model;
    we normalized $\\sigma_1=1$

    Args:
        muhat: a Matching

    Returns:
        the (X,Y,X+Y-1) parameter-dependent part of the hessian of the entropy.
    """
    muxy, mux0, mu0y, *_ = muhat.unpack()
    X, Y = muxy.shape
    n_alpha = X + Y - 1

    e_vals = np.zeros((X, Y, n_alpha))
    i = 0
    for x in range(1, X):
        e_vals[x, :, i] = -np.log(muxy[x, :] / mux0[x])
        i += 1
    for y in range(Y):
        e_vals[:, y, i] = -np.log(muxy[:, y] / mu0y[y])
        i += 1

    return e_vals


def e_derivative_mu_heteroskedastic(
    muhat: Matching,
) -> ThreeArrays:
    """Returns the derivatives of the parameter-dependent part $e$
    wrt $\\mu$ for the Choo and Siow heteroskedastic model;
    we normalized $\\sigma_1=1$

    Args:
        muhat: a Matching

    Returns:
        the parameter-dependent part of the hessian of the entropy
        wrt $(\\mu,\\mu)$.
    """
    muxy, mux0, mu0y, *_ = muhat.unpack()
    X, Y = muxy.shape
    n_alpha = X + Y - 1
    hess_x = np.zeros((X, Y, Y, n_alpha))
    hess_y = np.zeros((X, Y, X, n_alpha))
    hess_xy = np.zeros((X, Y, n_alpha))

    der_logxy = 1.0 / muxy
    der_logx0 = 1.0 / mux0
    der_log0y = 1.0 / mu0y
    i = 0
    for x in range(1, X):
        # derivatives wrt sigma_x
        dlogx0 = der_logx0[x]
        dlogxy = der_logxy[x, :]
        for y in range(Y):
            hess_x[x, y, :, i] = -dlogx0
            hess_xy[x, y, i] = -dlogxy[y] - dlogx0
        i += 1
    for y in range(Y):
        # derivatives wrt tau_y
        dlog0y = der_log0y[y]
        dlogxy = der_logxy[:, y]
        for x in range(X):
            hess_y[x, y, :, i] = -dlog0y
            hess_xy[x, y, i] = -dlogxy[x] - dlog0y
        i += 1

    return hess_x, hess_y, hess_xy


def e_derivative_r_heteroskedastic(
    muhat: Matching,
) -> TwoArrays:
    """Returns the derivatives of the parameter-dependent part $e$
    wrt $r$ for the Choo and Siow heteroskedastic model;
    we normalized $\\sigma_1=1$

    Args:
        muhat: a Matching

    Returns:
        the parameter-dependent part of the hessian of the entropy
        wrt $r$.
    """
    muxy, mux0, mu0y, *_ = muhat.unpack()
    X, Y = muxy.shape
    n_alpha = X + Y - 1
    hess_n = np.zeros((X, Y, n_alpha))
    hess_m = np.zeros((X, Y, n_alpha))

    der_logx0 = 1.0 / mux0
    der_log0y = 1.0 / mu0y
    i = 0
    for x in range(1, X):
        # derivatives wrt sigma_x
        dlogx0 = der_logx0[x]
        for y in range(Y):
            hess_n[x, y, i] = dlogx0
        i += 1
    for y in range(Y):
        # derivatives wrt tau_y
        dlog0y = der_log0y[y]
        for x in range(X):
            hess_m[x, y, i] = dlog0y
        i += 1

    return hess_n, hess_m


e_derivative_choo_siow_heteroskedastic = (
    e_derivative_mu_heteroskedastic,
    e_derivative_r_heteroskedastic,
)

entropy_choo_siow_heteroskedastic = EntropyFunctions(
    e0_fun=e0_choo_siow_heteroskedastic,
    parameter_dependent=True,
    e_fun=e_choo_siow_heteroskedastic,
    hessian="provided",
    e0_derivative=cast(EntropyHessians, e0_derivative_choo_siow_heteroskedastic),
    e_derivative=cast(EntropyHessians, e_derivative_choo_siow_heteroskedastic),
    description="Choo and Siow heteroskedastic with analytic Hessian",
)

entropy_choo_siow_heteroskedastic_numeric = EntropyFunctions(
    e0_fun=e0_choo_siow_heteroskedastic,
    parameter_dependent=True,
    e_fun=e_choo_siow_heteroskedastic,
    description="Choo and Siow heteroskedastic with numerical Hessian",
)

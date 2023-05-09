"""The components of the derivative of the entropy
for the Choo and Siow gender-heteroskedastic model.

We normalize the standard error for the X side at 1,
and we estimate the standard error on the Y side.
"""
from typing import cast

import numpy as np
from bs_python_utils.bsnputils import ThreeArrays, TwoArrays

from cupid_matching.entropy import EntropyFunctions, EntropyHessians
from cupid_matching.matching_utils import Matching


def e0_choo_siow_gender_heteroskedastic(muhat: Matching) -> np.ndarray:
    """Returns the values of the parameter-independent part $e_0$
    for the Choo and Siow gender-heteroskedastic model; we normalized $\\sigma=1$.

    Args:
        muhat: a Matching

    Returns:
        the (X,Y) matrix of the parameter-independent part
        of the first derivative of the entropy.
    """
    muxy, mux0, *_ = muhat.unpack()
    e0_vals = -np.log(muxy / mux0.reshape((-1, 1)))
    return cast(np.ndarray, e0_vals)


def e0_derivative_mu_gender_heteroskedastic(
    muhat: Matching,
) -> ThreeArrays:
    """Returns the derivatives of the parameter-independent part $e_0$ in $\\mu$.
    for the Choo and Siow gender-heteroskedastic model; we normalized $\\sigma=1$.

    Args:
        muhat: a Matching

    Returns:
        the  parameter-independent part of the hessian of the entropy
        wrt $(\\mu,\\mu)$.
    """
    muxy, mux0, *_ = muhat.unpack()
    X, Y = muxy.shape
    hess_x = np.zeros((X, Y, Y))
    hess_y = np.zeros((X, Y, X))
    hess_xy = np.zeros((X, Y))
    der_logxy = 1.0 / muxy
    der_logx0 = 1.0 / mux0
    for x in range(X):
        dlogx0 = der_logx0[x]
        for y in range(Y):
            hess_x[x, y, :] = -dlogx0
            hess_xy[x, y] = -der_logxy[x, y] - dlogx0
    return hess_x, hess_y, hess_xy


def e0_derivative_r_gender_heteroskedastic(
    muhat: Matching,
) -> TwoArrays:
    """Returns the derivatives of the parameter-independent part $e_0$ wrt $r$
    for the Choo and Siow gender-heteroskedastic model; we normalized $\\sigma=1$.

    Args:
        muhat: a Matching

    Returns:
        the parameter-independent part of the hessian of the entropy
        wrt $(\\mu,r)$.
    """
    muxy, mux0, *_ = muhat.unpack()
    X, Y = muxy.shape
    hess_n = np.zeros((X, Y))
    hess_m = np.zeros((X, Y))
    der_logx0 = 1.0 / mux0
    for x in range(X):
        dlogx0 = der_logx0[x]
        for y in range(Y):
            hess_n[x, y] = dlogx0
    return hess_n, hess_m


e0_derivative_choo_siow_gender_heteroskedastic = (
    e0_derivative_mu_gender_heteroskedastic,
    e0_derivative_r_gender_heteroskedastic,
)


def e_choo_siow_gender_heteroskedastic(muhat: Matching) -> np.ndarray:
    """Returns the values of the parameter-dependent part  $e$
    for the Choo and Siow gender-heteroskedastic model; we normalized $\\sigma=1$.

    Args:
        muhat: a Matching

    Returns:
        the (X,Y,1) array of the parameter-dependent part
        of the first derivative of the entropy.
    """
    muxy, _, mu0y, *_ = muhat.unpack()
    X, Y = muxy.shape
    n_alpha = 1

    e_vals = np.zeros((X, Y, n_alpha))
    e_vals[:, :, 0] = -np.log(muxy / mu0y)
    return e_vals


def e_derivative_mu_gender_heteroskedastic(
    muhat: Matching,
) -> ThreeArrays:
    """Returns the derivatives of the parameter-dependent part $e$
     wrt $\\mu$ for the Choo and Siow gender-heteroskedastic model;
     we normalized $\\sigma_1=1$.

    Args:
        muhat: a Matching

    Returns:
        the parameter-dependent part of the hessian of the entropy
        wrt $(\\mu,\\mu)$.
    """
    muxy, _, mu0y, *_ = muhat.unpack()
    X, Y = muxy.shape

    n_alpha = 1
    hess_x = np.zeros((X, Y, Y, n_alpha))
    hess_y = np.zeros((X, Y, X, n_alpha))
    hess_xy = np.zeros((X, Y, n_alpha))
    der_logxy = 1.0 / muxy
    der_log0y = 1.0 / mu0y
    for x in range(X):
        for y in range(Y):
            dlog0y = der_log0y[y]
            hess_y[x, y, :, 0] = -dlog0y
            hess_xy[x, y, 0] = -der_logxy[x, y] - dlog0y
    return hess_x, hess_y, hess_xy


def e_derivative_r_gender_heteroskedastic(
    muhat: Matching,
) -> TwoArrays:
    """Returns the derivatives of the parameter-dependent part $e$
     wrt $r$ for the Choo and Siow gender-heteroskedastic model;
     we normalized $\\sigma_1=1$.

    Args:
        muhat: a Matching

    Returns:
        the parameter-dependent part of the hessian of the entropy
        wrt $(\\mu,r)$
    """
    muxy, _, mu0y, *_ = muhat.unpack()
    X, Y = muxy.shape

    n_alpha = 1
    hess_n = np.zeros((X, Y, n_alpha))
    hess_m = np.zeros((X, Y, n_alpha))
    der_log0y = 1.0 / mu0y
    for x in range(X):
        for y in range(Y):
            dlog0y = der_log0y[y]
            hess_m[x, y, 0] = dlog0y
    return hess_n, hess_m


e_derivative_choo_siow_gender_heteroskedastic = (
    e_derivative_mu_gender_heteroskedastic,
    e_derivative_r_gender_heteroskedastic,
)

entropy_choo_siow_gender_heteroskedastic = EntropyFunctions(
    e0_fun=e0_choo_siow_gender_heteroskedastic,
    parameter_dependent=True,
    e_fun=e_choo_siow_gender_heteroskedastic,
    hessian="provided",
    e0_derivative=cast(EntropyHessians, e0_derivative_choo_siow_gender_heteroskedastic),
    e_derivative=cast(EntropyHessians, e_derivative_choo_siow_gender_heteroskedastic),
    description="Choo and Siow gender-heteroskedastic with analytic Hessian",
)

entropy_choo_siow_gender_heteroskedastic_numeric = EntropyFunctions(
    e0_fun=e0_choo_siow_gender_heteroskedastic,
    parameter_dependent=True,
    e_fun=e_choo_siow_gender_heteroskedastic,
    description="Choo and Siow gender-heteroskedastic with numerical Hessian",
)

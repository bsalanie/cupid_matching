"""The components of the derivative of the entropy
for the Choo and Siow homoskedastic model.
"""
from typing import Any, Literal, Optional, Union, cast, overload

import numpy as np

from cupid_matching.entropy import EntropyFunctions, EntropyHessians
from cupid_matching.matching_utils import Matching
from cupid_matching.utils import FourArrays, ThreeArrays, TwoArrays, bs_error_abort, \
    npmaxabs, nppow


@overload
def _entropy_choo_siow(muhat: Matching, deriv: Literal[0]) -> np.ndarray:
    ...


@overload
def _entropy_choo_siow(muhat: Matching, deriv: Literal[1]) -> TwoArrays:
    ...


@overload
def _entropy_choo_siow(muhat: Matching, deriv: Literal[2]) -> FourArrays:
    ...


def _entropy_choo_siow(
        muhat: Matching, deriv: Optional[int] = 0
) -> Union[None, np.ndarray, TwoArrays, FourArrays]:
    """Returns the values of $\\mathcal{E}$
    and the first (if `deriv` is 1 or 2) and second (if `deriv` is 2) derivatives
    for the Choo and Siow model

    Args:
        muhat: a Matching
        deriv: if equal 1, we compute the first derivatives too;
               if equals 2, also the hessian

    Returns:
        the value of the generalized entropy
        if deriv = 1 or 2, the (X,Y) matrix of the first derivative of the entropy
        if deriv = 2, the (X,Y,X,Y) array of the second derivative
            wrt $(\\mu,\\mu)$
          and the (X,Y,X+Y) second derivatives
            wrt $(\\mu,(n,m))$
    """
    muxy, mux0, mu0y, n, m = muhat.unpack()

    logxy = np.log(muxy)
    logx0 = np.log(mux0)
    log0y = np.log(mu0y)

    val_entropy = (
            -2.0 * np.sum(muxy * logxy)
            - np.sum(mux0 * logx0)
            - np.sum(mu0y * log0y)
            + np.sum(n * np.log(n))
            + np.sum(m * np.log(m))
    )

    if deriv == 0:
        return val_entropy
    if deriv in [1, 2]:
        der_xy = -2.0 * logxy + log0y
        der_xy += logx0.reshape((-1, 1))
        if deriv == 1:
            return val_entropy, der_xy
        else:  # we compute the Hessians
            X, Y = muxy.shape
            derlogxy = 1.0 / muxy
            derlogx0 = 1.0 / mux0
            derlog0y = 1.0 / mu0y
            der2_xyzt = np.zeros((X, Y, X, Y))
            der2_xyr = np.zeros((X, Y, X + Y))
            for x in range(X):
                dlogx0 = derlogx0[x]
                for y in range(Y):
                    d2xy = np.zeros((X, Y))
                    d2xy[x, :] = -dlogx0
                    d2xy[:, y] -= derlog0y[y]
                    d2xy[x, y] -= 2.0 * derlogxy[x, y]
                    der2_xyzt[x, y, :, :] = d2xy
                    der2_xyr[x, y, x] = derlogx0[x]
                    der2_xyr[x, y, X + y] = derlog0y[y]
            return val_entropy, der_xy, der2_xyzt, der2_xyr
    else:
        bs_error_abort("deriv should be 0, 1, or 2")
        return None


def e0_fun_choo_siow(muhat: Matching) -> np.ndarray:
    """Returns the values of $e_0$ for the Choo and Siow model.

    Args:
        muhat: a Matching

    Returns:
        the (X,Y) matrix of the first derivative of the entropy
    """
    entropy_res = _entropy_choo_siow(muhat, deriv=1)
    return entropy_res[1]


def e0_fun_choo_siow_corrected(
        muhat: Matching
) -> np.ndarray:
    """Returns the values of $e_0$ for the Choo and Siow model,
    using the finite-sample correction.

    Args:
        muhat: a Matching

    Returns:
        the (X,Y) matrix of the first derivative of the entropy
    """
    entropy_res = _entropy_choo_siow(muhat, deriv=1)
    e0_val = entropy_res[1]
    muxy, mux0, mu0y, *_ = muhat.unpack()
    n_households = np.sum(muxy) + np.sum(mux0) + np.sum(mu0y)
    e0_val_corrected = e0_val - (1.0 - muxy / n_households) / muxy
    e0_val_corrected += ((1.0 - mux0 / n_households) / (2.0 * mux0)).reshape((-1, 1))
    e0_val_corrected += ((1.0 - mu0y / n_households) / (2.0 * mu0y))
    # print(f"{npmaxabs(e0_val_corrected/e0_val - 1.0)=}")
    return e0_val_corrected


def hessian_mumu_choo_siow(muhat: Matching) -> ThreeArrays:
    """Returns the derivatives of $e_0$ in $\\mu$
    for the Choo and Siow model.

    Args:
        muhat: a Matching

    Returns:
        the three components of the hessian wrt $(\\mu,\\mu)$ of the entropy
    """
    entropy_res = _entropy_choo_siow(muhat, deriv=2)
    hessmumu = entropy_res[2]
    muxy, *_ = muhat.unpack()
    X, Y = muxy.shape
    hess_x = np.zeros((X, Y, Y))
    hess_y = np.zeros((X, Y, X))
    hess_xy = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            d2xy = hessmumu[x, y, :, :]
            hess_x[x, y, :] = d2xy[x, :]
            hess_y[x, y, :] = d2xy[:, y]
            hess_xy[x, y] = d2xy[x, y]
    return hess_x, hess_y, hess_xy


def hessian_mumu_choo_siow_corrected(
        muhat: Matching
) -> ThreeArrays:
    """Returns the derivatives of $e_0$ in $\\mu$
    for the Choo and Siow model, with the small sample correction

    Args:
        muhat: a Matching

    Returns:
        the three components of the hessian wrt $(\\mu,\\mu)$ of the entropy
    """
    hess_x, hess_y, hess_xy = hessian_mumu_choo_siow(muhat)
    muxy, mux0, mu0y, *_ = muhat.unpack()
    n_households = np.sum(muxy) + np.sum(mux0) + np.sum(mu0y)
    X, Y = hess_xy.shape
    corr_x0 = 1.0 / (2.0*mux0*mux0)
    corr_0y = 1.0 / (2.0*mu0y*mu0y)
    corr_xy = -1.0 / (muxy*muxy)
    for x in range(X):
        for y in range(Y):
            hess_x[x, y, y] += corr_x0[x]
            hess_y[x, y, x] += corr_0y[y]
            hess_xy[x, y] += corr_xy[x, y]

    return hess_x, hess_y, hess_xy


def hessian_mur_choo_siow(muhat: Matching) -> TwoArrays:
    """Returns the derivatives of $e_0$ in $r$
    for the Choo and Siow model.

    Args:
        muhat: a Matching

    Returns:
        the two components of the hessian wrt $(\\mu,r)$ of the entropy
    """
    entropy_res = _entropy_choo_siow(muhat, deriv=2)
    hessmur = entropy_res[3]
    muxy, *_ = muhat.unpack()
    X, Y = muxy.shape
    hess_nx = np.zeros((X, Y))
    hess_my = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            d2r = hessmur[x, y, :]
            hess_nx[x, y] = d2r[x]
            hess_my[x, y] = d2r[X + y]
    return hess_nx, hess_my


def hessian_mur_choo_siow_corrected(
        muhat: Matching
) -> TwoArrays:
    """Returns the derivatives of $e_0$ in $r$
    for the Choo and Siow model, with the small sample correction

    Args:
        muhat: a Matching

    Returns:
        the two components of the hessian wrt $(\\mu,r)$ of the entropy
    """
    hess_nx, hess_my = hessian_mur_choo_siow(muhat)
    _, mux0, mu0y, *_ = muhat.unpack()
    corr_nx = 1.0 / (2.0*mux0*mux0)
    corr_my = 1.0 / (2.0*mu0y*mu0y)
    X, Y = mux0.size, mu0y.size
    for x in range(X):
        hess_my[x, :] += corr_my
    for y in range(Y):
        hess_nx[:, y] += corr_nx
    return hess_nx, hess_my


e0_derivative_choo_siow = (hessian_mumu_choo_siow, hessian_mur_choo_siow)
e0_derivative_choo_siow_corrected = (
    hessian_mumu_choo_siow_corrected,
    hessian_mur_choo_siow_corrected,
)

entropy_choo_siow = EntropyFunctions(
    e0_fun=e0_fun_choo_siow,
    hessian="provided",
    e0_derivative=cast(EntropyHessians, e0_derivative_choo_siow),
    description="Choo and Siow homoskedastic with analytic Hessian",
)

entropy_choo_siow_corrected = EntropyFunctions(
    e0_fun=e0_fun_choo_siow_corrected,
    hessian="provided",
    e0_derivative=cast(EntropyHessians, e0_derivative_choo_siow_corrected),
    description="Choo and Siow homoskedastic with analytic Hessian and finite-sample correction",
)

entropy_choo_siow_numeric = EntropyFunctions(
    e0_fun=e0_fun_choo_siow,
    description="Choo and Siow homoskedastic with numerical Hessian",
)

entropy_choo_siow_corrected_numeric = EntropyFunctions(
    e0_fun=e0_fun_choo_siow_corrected,
    description="Choo and Siow homoskedastic with numerical Hessian and finite-sample correction",
)

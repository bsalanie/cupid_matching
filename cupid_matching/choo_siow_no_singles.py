"""The components of the derivative of the entropy for the Choo and Siow homoskedastic model w/o singles.
"""

from typing import cast

import numpy as np
from bs_python_utils.bsnputils import ThreeArrays, TwoArrays, bs_error_abort

from cupid_matching.entropy import (
    EntropyFunctions,
    EntropyHessians,
    check_additional_parameters,
)
from cupid_matching.matching_utils import Matching


def _entropy_choo_siow_no_singles(
    muhat: Matching, deriv: int | None = 0
) -> (
    None
    | float
    | tuple[float, np.ndarray]
    | tuple[float, np.ndarray, np.ndarray, np.ndarray]
):
    """Returns the values of $\\mathcal{E}$ and the first (if `deriv` is 1 or 2) and second (if `deriv` is 2) derivatives
        for the Choo and Siow model w/o singles

    Args:
        muhat: a `Matching`
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
    muxy, *_, n, m = muhat.unpack()

    logxy = np.log(muxy)
    logn = np.log(n)
    logm = np.log(m)

    val_entropy = -2.0 * np.sum(muxy * logxy) + np.sum(n * logn) + np.sum(m * logm)

    if deriv == 0:
        return cast(float, val_entropy)
    if deriv in [1, 2]:
        der_xy = -2.0 * (logxy + 1.0)
        if deriv == 1:
            return val_entropy, der_xy
        else:  # we compute_ the Hessians
            X, Y = muxy.shape
            der2_xyzt = np.zeros((X, Y, X, Y))
            der2_xyr = np.zeros((X, Y, X + Y))
            for x in range(X):
                for y in range(Y):
                    der2_xyzt[x, y, x, y] = -2.0 / muxy[x, y]
            return val_entropy, der_xy, der2_xyzt, der2_xyr
    else:
        bs_error_abort("deriv should be 0, 1, or 2")
        return None


def _der_entropy_choo_siow_no_singles_corrected(
    muhat: Matching, hessian: bool | None = False
) -> np.ndarray | ThreeArrays:
    """Returns the corrected first derivative of $\\mathcal{E}$ and the corrected second derivative (if `hessian` is True)
        for the Choo and Siow model w/o singles

    Args:
        muhat: a `Matching`
        hessian: if `True`, also compute_ the hessian

    Returns:
        the (X,Y) matrix of the first derivative of the entropy
        if hessian is True, the (X,Y,X,Y) array of the second derivative wrt $(\\mu,\\mu)$
          and the (X,Y,X+Y) second derivatives wrt $(\\mu,(n,m))$
    """
    muxy, *_ = muhat.unpack()
    n_households = np.sum(muxy)

    muxy_corr = muxy + (1.0 - muxy / n_households) / 2.0
    logxy = np.log(muxy_corr)

    der_xy = -2.0 * (logxy + 1.0)
    if not hessian:
        return der_xy
    else:  # we compute_ the Hessians
        X, Y = muxy.shape
        f_corr = 1.0 - 1.0 / n_households / 2.0
        derlogxy = f_corr / muxy_corr
        der2_xyzt = np.zeros((X, Y, X, Y))
        der2_xyr = np.zeros((X, Y, X + Y))
        for x in range(X):
            for y in range(Y):
                der2_xyzt[x, y, x, y] = -2.0 * derlogxy[x, y]
        return der_xy, der2_xyzt, der2_xyr


def e0_fun_choo_siow_no_singles(
    muhat: Matching,
    additional_parameters: list | None = None,
) -> np.ndarray:
    """Returns the values of $e_0$ for the Choo and Siow model w/o singles.

    Args:
        muhat: a `Matching`

    Returns:
        the `(X,Y)` matrix of the first derivative of the entropy
    """
    check_additional_parameters(0, additional_parameters)
    _, der_entropy = cast(
        tuple[float, np.ndarray], _entropy_choo_siow_no_singles(muhat, deriv=1)
    )
    return der_entropy


def e0_fun_choo_siow_no_singles_corrected(
    muhat: Matching,
    additional_parameters: list | None = None,
) -> np.ndarray:
    """Returns the values of $e_0$ for the Choo and Siow model,
        using the finite-sample correction $\\log(p+(1-p)/(2N))$

    Args:
        muhat: a `Matching`

    Returns:
        the (X,Y) matrix of the first derivative of the entropy
    """
    check_additional_parameters(0, additional_parameters)
    e0_val_corrected = _der_entropy_choo_siow_no_singles_corrected(muhat, hessian=False)
    return e0_val_corrected


def hessian_mumu_choo_siow_no_singles(
    muhat: Matching,
    additional_parameters: list | None = None,
) -> ThreeArrays:
    """Returns the hessian of $e_0$ wrt $(\\mu,\\mu)$ for the Choo and Siow model w/o singles.

    Args:
        muhat: a `Matching`

    Returns:
        the three components of the hessian  of the entropy wrt $(\\mu,\\mu)$
    """
    check_additional_parameters(0, additional_parameters)
    _, _, hessmumu, _ = cast(
        tuple[float, np.ndarray, np.ndarray, np.ndarray],
        _entropy_choo_siow_no_singles(muhat, deriv=2),
    )
    muxy, *_ = muhat.unpack()
    X, Y = muxy.shape
    hess_x = np.zeros((X, Y, Y))
    hess_y = np.zeros((X, Y, X))
    hess_xy = np.zeros((X, Y))
    for x in range(X):
        d2x = hessmumu[x, :, x, :]
        for y in range(Y):
            hess_xy[x, y] = d2x[y, y]
    return hess_x, hess_y, hess_xy


def hessian_mumu_choo_siow_no_singles_corrected(
    muhat: Matching,
    additional_parameters: list | None = None,
) -> ThreeArrays:
    """Returns the derivatives of the hessian of $e_0$ wrt $(\\mu,\\mu)$
        for the Choo and Siow model w/o singles, with the small sample correction

    Args:
        muhat: a `Matching`

    Returns:
        the three components of the hessian of the entropy wrt $(\\mu,\\mu)$
    """
    check_additional_parameters(0, additional_parameters)
    _, hessmumu, _ = cast(
        tuple[np.ndarray, np.ndarray, np.ndarray],
        _der_entropy_choo_siow_no_singles_corrected(muhat, hessian=True),
    )
    X, Y = muhat.muxy.shape
    hess_x = np.zeros((X, Y, Y))
    hess_y = np.zeros((X, Y, X))
    hess_xy = np.zeros((X, Y))
    for x in range(X):
        d2x = hessmumu[x, :, x, :]
        for y in range(Y):
            hess_xy[x, y] = d2x[y, y]
    return hess_x, hess_y, hess_xy


def hessian_mur_choo_siow_no_singles(
    muhat: Matching,
    additional_parameters: list | None = None,
) -> TwoArrays:
    """Returns the hessian of $e_0$ wrt $(\\mu,r)$ for the Choo and Siow model w/o singles.

    Args:
        muhat: a Matching

    Returns:
        the two components of the hessian of the entropy wrt $(\\mu,r)$
    """
    check_additional_parameters(0, additional_parameters)
    X, Y = muhat.muxy.shape
    hess_nx = np.zeros((X, Y))
    hess_my = np.zeros((X, Y))
    return hess_nx, hess_my


def hessian_mur_choo_siow_no_singles_corrected(
    muhat: Matching,
    additional_parameters: list | None = None,
) -> TwoArrays:
    """Returns the hessian of $e_0$ wrt $(\\mu,r)$ for the Choo and Siow model w/o singles, with the small sample correction

    Args:
        muhat: a `Matching`

    Returns:
        the two components of the hessian  of the entropy wrt $(\\mu,r)$
    """
    check_additional_parameters(0, additional_parameters)
    X, Y = muhat.muxy.shape
    hess_nx = np.zeros((X, Y))
    hess_my = np.zeros((X, Y))
    return hess_nx, hess_my


e0_derivative_choo_siow_no_singles = (
    hessian_mumu_choo_siow_no_singles,
    hessian_mur_choo_siow_no_singles,
)
e0_derivative_choo_siow_no_singles_corrected = (
    hessian_mumu_choo_siow_no_singles_corrected,
    hessian_mur_choo_siow_no_singles_corrected,
)

entropy_choo_siow_no_singles = EntropyFunctions(
    e0_fun=e0_fun_choo_siow_no_singles,
    hessian="provided",
    e0_derivative=cast(EntropyHessians, e0_derivative_choo_siow_no_singles),
    description="Choo and Siow homoskedastic  w/o singles with analytic Hessian",
)

entropy_choo_siow_no_singles_corrected = EntropyFunctions(
    e0_fun=e0_fun_choo_siow_no_singles_corrected,
    hessian="provided",
    e0_derivative=cast(EntropyHessians, e0_derivative_choo_siow_no_singles_corrected),
    description=(
        "Choo and Siow homoskedastic w/o singles with analytic Hessian and"
        " finite-sample correction"
    ),
)

entropy_choo_siow_no_singles_numeric = EntropyFunctions(
    e0_fun=e0_fun_choo_siow_no_singles,
    description="Choo and Siow homoskedastic w/o singles with numerical Hessian",
)

entropy_choo_siow_no_singles_corrected_numeric = EntropyFunctions(
    e0_fun=e0_fun_choo_siow_no_singles_corrected,
    description=(
        "Choo and Siow homoskedastic w/o singles with numerical Hessian and"
        " finite-sample correction"
    ),
)

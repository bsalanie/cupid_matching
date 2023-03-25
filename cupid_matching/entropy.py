"""Entropies and their derivatives. """
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Literal, Optional, cast

import numpy as np

from .matching_utils import Matching, MatchingFunction, MatchingFunctionParam
from .utils import _EPS, _TWO_EPS, ThreeArrays, TwoArrays, bs_error_abort

EntropyHessianMuMu = Callable[[Matching], ThreeArrays]
"""The type of a function that takes in a Matching
   and returns the three components of the hessian of the entropy
   wrt $(\\mu,\\mu)$.
"""


EntropyHessianMuMuParam = Callable[[Matching, list[Any]], ThreeArrays]
"""The type of a function that takes in a Matching
   and a list of additional parameters
   and returns the three components of the hessian of the entropy
   wrt $(\\mu,\\mu)$.
"""

EntropyHessianMuR = Callable[[Matching], TwoArrays]
"""The type of a function that takes in a Matching
   and returns the two components of the hessian of the entropy
   wrt $(\\mu,n)$ and $(\\mu, m))$.
"""

EntropyHessianMuRParam = Callable[[Matching, list[Any]], TwoArrays]
"""The type of a function that takes in a Matching
   and a list of additional parameters
   and returns the two components of the hessian of the entropy
   wrt $(\\mu,n)$ and $(\\mu, m))$.
"""

EntropyHessianComponents = tuple[ThreeArrays, TwoArrays]
""" combines the tuples of the values 
    of the components of the hessians."""

EntropyHessians = tuple[EntropyHessianMuMu, EntropyHessianMuR]
""" combines the hessian functions """

EntropyHessiansParam = tuple[EntropyHessianMuMuParam, EntropyHessianMuRParam]
""" combines the hessian functions 
    when additional parameters are used"""


@dataclass
class EntropyFunctions:
    """Defines the entropy used, via the derivative $e_0 + e \\cdot \\alpha$

    Attributes:
        e0_fun: required
        parameter_dependent:  if `True`, the entropy depends on parameters.
            Defaults to `False`
        e_fun: only in entropies that depend on parameters.
            Defaults to `None`
        hessian: defaults to `"numeric"`
            * if `"provided"`, we provide the hessian of the entropy.
            * if `"numerical"`, it is computed by central differences.
        e0_derivative: the derivative of `e0_fun`, if available.
            Defaults to `None`
        e_derivative: the derivative of `e_fun`, if available.
            Defaults to `None`
        additional_parameters: additional parameters
            that define the distribution of errors.
            Defaults to `None`
        description: some text describing the model.
            Defaults to `None`

    Examples:
        See `entropy_choo_siow` in `choo_siow.py`
    """

    e0_fun: MatchingFunction | MatchingFunctionParam
    e0_derivative: Optional[EntropyHessians | EntropyHessiansParam] = None
    additional_parameters: Optional[list] = None
    description: Optional[str] = None
    e_fun: Optional[MatchingFunction | MatchingFunctionParam] = None
    e_derivative: Optional[EntropyHessians | EntropyHessiansParam] = None
    hessian: Optional[str] = "numerical"
    parameter_dependent: bool = False

    def __post_init__(self):
        if not self.parameter_dependent:
            if self.hessian == "provided" and self.e0_derivative is None:
                bs_error_abort(
                    "You claim to provide the hessian "
                    + "but you did not provide the e0_derivative."
                )
        else:
            if self.e_fun is None:
                bs_error_abort(
                    "Your entropy is parameter dependent "
                    + " but you did not provide the e_fun."
                )
            if self.hessian == "provided" and self.e_derivative is None:
                bs_error_abort(
                    "Your entropy is parameter dependent, "
                    + "you claim to provide the hessian,\n"
                    + " but I do not see the e_derivative."
                )


def entropy_gradient(
    entropy: EntropyFunctions,
    muhat: Matching,
    alpha: Optional[np.ndarray] = None,
    additional_parameters: Optional[list] = None,
) -> np.ndarray:
    """Computes the derivative of the entropy wrt $\\mu$
     at $(\\mu, n, m, \\alpha, p)$

    Args:
        entropy: the `EntropyFunctions` object
        muhat: a Matching
        alpha: a vector of parameters of the derivative of the entropy, if any
        additional_parameters: a list of additional parameters `p`, if any

    Returns:
        the derivative of the entropy wrt $\\mu$
        at $(\\mu, n, m, \\alpha, p)$.
    """
    e0_fun = entropy.e0_fun
    if additional_parameters is not None:
        e0_fun = cast(MatchingFunctionParam, e0_fun)
        e0_vals = e0_fun(muhat, additional_parameters)
    else:
        e0_fun = cast(MatchingFunction, e0_fun)
        e0_vals = e0_fun(muhat)
    parameter_dependent = entropy.parameter_dependent
    if parameter_dependent:
        if alpha is None:
            bs_error_abort("alpha should be specified for this model")
        e_fun = entropy.e_fun
        if e_fun is None:
            bs_error_abort("we should have an e_fun in this model")
        else:
            if additional_parameters is not None:
                e_fun = cast(MatchingFunctionParam, e_fun)
                e_vals = e_fun(muhat, additional_parameters)
            else:
                e_fun = cast(MatchingFunction, e_fun)
                e_vals = e_fun(muhat)
        return e0_vals + e_vals @ alpha
    else:
        return e0_vals


def _numeric_component(
    muhat: Matching,
    x: int,
    y: int,
    t: int,
    entropy_deriv: MatchingFunction,
    direction: Literal["x", "y", "d", "n", "m"],
) -> float:
    """
    Takes the numerical hessian in one direction

    Args:
        muhat: the Matching
        x, y: the element we are working on
        t: the element of the derivative, if any
        entropy_deriv: the first derivative of the entropy
        direction:
            'y': wrt to mu[x, t]
            'x: wrt to mu[t, y]
            'd': wrt to mu[x, y]
            'n': wrt n[x]
            'm': wrt m[y]

    """
    muxy, _, _, n, m = muhat.unpack()

    if direction == "y":
        muxy[x, t] += _EPS
        mus = Matching(muxy, n, m)
        der_entropy_plus = entropy_deriv(mus)
        muxy[x, t] -= _TWO_EPS
        mus = Matching(muxy, n, m)
    elif direction == "x":
        muxy[t, y] += _EPS
        mus = Matching(muxy, n, m)
        der_entropy_plus = entropy_deriv(mus)
        muxy[t, y] -= _TWO_EPS
    elif direction == "d":
        muxy[x, y] += _EPS
        mus = Matching(muxy, n, m)
        der_entropy_plus = entropy_deriv(mus)
        muxy[x, y] -= _TWO_EPS
    elif direction == "n":
        n[x] += _EPS
        mus = Matching(muxy, n, m)
        der_entropy_plus = entropy_deriv(mus)
        n[x] -= _TWO_EPS
    elif direction == "m":
        m[y] += _EPS
        mus = Matching(muxy, n, m)
        der_entropy_plus = entropy_deriv(mus)
        m[y] -= _TWO_EPS
    else:
        bs_error_abort("Wrong direction parameter.")
    der_entropy_minus = entropy_deriv(mus)
    deriv_value = (der_entropy_plus[x, y] - der_entropy_minus[x, y]) / _TWO_EPS
    return deriv_value


def _numeric_hessian(
    entropy: EntropyFunctions,
    muhat: Matching,
    alpha: Optional[np.ndarray] = None,
    additional_parameters: Optional[list] = None,
) -> EntropyHessianComponents:
    """Evaluates numerically the components of the hessians of the entropy
    wrt $(\\mu,\\mu)$ and $(\\mu,(n,m))$

    Args:
        entropy: the `EntropyFunctions` object
        muhat: a Matching
        alpha: a vector of parameters of the derivative of the entropy, if any
        additional_parameters: a list of additional parameters, if any

    Returns:
        the hessians of the entropy wrt $(\\mu,\\mu)$ and $(\\mu,(n,m))$.
    """
    parameter_dependent = entropy.parameter_dependent
    # we create a derivative of entropy that is only a function
    #  of the Matching
    if not parameter_dependent:
        if additional_parameters is not None:
            entropy_deriv = partial(
                entropy_gradient,
                entropy,
                additional_parameters=additional_parameters,
            )
        else:
            entropy_deriv = partial(entropy_gradient, entropy)
    else:
        if additional_parameters is not None:
            entropy_deriv = partial(
                entropy_gradient,
                entropy,
                alpha=alpha,
                additional_parameters=additional_parameters,
            )
        else:
            entropy_deriv = partial(entropy_gradient, entropy, alpha=alpha)

    muxyhat, _, _, n, m = muhat.unpack()
    X, Y = muxyhat.shape

    # make sure everything is floating point
    muxyhatf = muxyhat.copy().astype(float)
    nf = n.copy().astype(float)
    mf = m.copy().astype(float)
    muhatf = Matching(muxyhatf, nf, mf)

    # start with the hessian wrt (mu, mu)
    hessian_x = np.zeros((X, Y, Y))
    hessian_y = np.zeros((X, Y, X))
    hessian_xy = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            for t in range(Y):
                hessian_x[x, y, t] = _numeric_component(
                    muhatf, x, y, t, entropy_deriv, direction="y"
                )
            for z in range(X):
                hessian_y[x, y, z] = _numeric_component(
                    muhatf, x, y, z, entropy_deriv, direction="x"
                )
            hessian_xy[x, y] = _numeric_component(
                muhatf, x, y, 0, entropy_deriv, direction="d"
            )
    components_mumu = (hessian_x, hessian_y, hessian_xy)

    # now the hessian wrt (mu, r)
    hessian_n = np.zeros((X, Y))
    hessian_m = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            hessian_n[x, y] = _numeric_component(
                muhatf, x, y, 0, entropy_deriv, direction="n"
            )
            hessian_m[x, y] = _numeric_component(
                muhatf, x, y, 0, entropy_deriv, direction="m"
            )

    components_mur = (hessian_n, hessian_m)

    return components_mumu, components_mur


def _fill_hessianMuMu_from_components(
    hessian_components: ThreeArrays,
) -> np.ndarray:
    """Fills the hessian of the entropy wrt $(\\mu,\\mu)$

    Args:
        hessian_components: the three components of the hessian

    Returns:
        the (XY,XY) matrix of the hessian
    """
    hess_x, hess_y, hess_xy = hessian_components
    X, Y = hess_xy.shape
    XY = X * Y
    hessian = np.zeros((XY, XY))

    i = 0
    ix = 0
    for x in range(X):
        for y in range(Y):
            hessian[i, ix : (ix + Y)] = hess_x[x, y, :]
            slice_y = slice(y, XY, Y)
            hessian[i, slice_y] = hess_y[x, y, :]
            hessian[i, i] = hess_xy[x, y]
            i += 1
        ix += Y

    return hessian


def _fill_hessianMuR_from_components(
    hessian_components: TwoArrays,
) -> np.ndarray:
    """Fills the hessian of the entropy wrt $(\\mu,(n,m))$

    Args:
        hessian_components: the two components of the hessian

    Returns:
        the (XY,X+Y) matrix of the hessian
    """
    hess_nx, hess_my = hessian_components
    X, Y = hess_nx.shape[:2]
    XY = X * Y
    hessian = np.zeros((XY, X + Y))

    i = 0
    for x in range(X):
        iy = X
        for y in range(Y):
            hessian[i, x] = hess_nx[x, y]
            hessian[i, iy] = hess_my[x, y]
            i += 1
            iy += 1

    return hessian

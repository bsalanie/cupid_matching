"""Entropies and their derivatives. """

from dataclasses import dataclass
from functools import partial
from typing import Literal, Protocol, cast

import numpy as np
from bs_python_utils.bsnputils import ThreeArrays, TwoArrays
from bs_python_utils.bsutils import bs_error_abort

from cupid_matching.matching_utils import (
    Matching,
    MatchingFunction,
    get_evals,
)
from cupid_matching.utils import _EPS, _TWO_EPS


def check_additional_parameters(
    number_required: int, additional_parameters: list | None
) -> None:
    """checks that the correct number of additional parameters is passed in

    Args:
        number_required: number we want
        additional_parameters: the list of additional parameters passed in, if any
    """
    if number_required == 0:
        if additional_parameters is not None:
            bs_error_abort("no additional parameters should be passed.")
    else:
        if additional_parameters is None:
            bs_error_abort("additional parameters should be passed.")
        additional_parameters = cast(list, additional_parameters)
        if len(additional_parameters) != number_required:
            bs_error_abort(
                f"additional parameters should be a list of {number_required} elements."
            )


class EntropyHessianMuMu(Protocol):
    def __call__(self, mus: Matching, additional_parameters=..., /) -> ThreeArrays:
        ...

    """The type of a function that takes in a `Matching` and possibly a list of additional parameters
    and returns the three components of the hessian of the entropy wrt $(\\mu,\\mu)$.
    """


class EntropyHessianMuR(Protocol):
    def __call__(self, mus: Matching, additional_parameters=..., /) -> TwoArrays:
        ...

    """The type of a function that takes in a `Matching` and possibly a list of additional parameters
    and returns the two components of the hessian of the entropy
    wrt $(\\mu,n)$ and $(\\mu, m))$.
    """


EntropyHessianComponents = tuple[ThreeArrays, TwoArrays]
""" combines the tuples of the values of the components of the hessians."""

EntropyHessians = tuple[EntropyHessianMuMu, EntropyHessianMuR]
""" combines the hessian functions. """


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
            * if `"numerical"`, it is compute_d by central differences.
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

    e0_fun: MatchingFunction  # | MatchingFunctionParam
    e0_derivative: EntropyHessians | None = None
    additional_parameters: list | None = None
    description: str | None = None
    # e_fun: MatchingFunction | MatchingFunctionParam | None = None
    e_fun: MatchingFunction | None = None
    e_derivative: EntropyHessians | None = None
    hessian: str | None = "numerical"
    parameter_dependent: bool = False

    def __post_init__(self):
        if (
            (not self.parameter_dependent)
            and self.hessian == "provided"
            and self.e0_derivative is None
        ):
            bs_error_abort(
                "You claim to provide the hessian "
                + "but you did not provide the e0_derivative."
            )
        if self.parameter_dependent:
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
    alpha: np.ndarray | None = None,
    additional_parameters: list | None = None,
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
    e0_vals = get_evals(entropy.e0_fun, muhat, additional_parameters)
    parameter_dependent = entropy.parameter_dependent
    if parameter_dependent:
        if alpha is None:
            bs_error_abort("alpha should be specified for this model")
        elif entropy.e_fun is None:
            bs_error_abort("we should have an e_fun in this model")
        else:
            e_vals = get_evals(entropy.e_fun, muhat, additional_parameters)
        return cast(np.ndarray, e0_vals + e_vals @ alpha)
    else:
        return cast(np.ndarray, e0_vals)


def _numeric_component(
    muhat: Matching,
    x: int,
    y: int,
    t: int,
    entropy_deriv: MatchingFunction,
    additional_parameters: list | None,
    direction: Literal["x", "y", "d", "n", "m"],
) -> float:
    """
    Takes the numerical hessian in one direction

    Args:
        muhat: the observed `Matching`
        x, y: the element we are working on
        t: the element of the derivative, if any
        entropy_deriv: the first derivative of the entropy
        additional_parameters: additional parameters of the distribution of errors, if any
        direction:
            'y': wrt to mu[x, t]
            'x: wrt to mu[t, y]
            'd': wrt to mu[x, y]
            'n': wrt n[x]
            'm': wrt m[y]

    """
    muxy, _, _, n, m = muhat.unpack()
    muxy1, n1, m1 = muxy.copy(), n.copy(), m.copy()

    if direction == "y":
        muxy1[x, t] += _EPS
        mus1 = Matching(muxy1, n, m)
        der_entropy_plus = get_evals(entropy_deriv, mus1, additional_parameters)
        muxy1[x, t] -= _TWO_EPS
    elif direction == "x":
        muxy1[t, y] += _EPS
        mus1 = Matching(muxy1, n, m)
        der_entropy_plus = get_evals(entropy_deriv, mus1, additional_parameters)
        muxy1[t, y] -= _TWO_EPS
    elif direction == "d":
        muxy1[x, y] += _EPS
        mus1 = Matching(muxy1, n, m)
        der_entropy_plus = get_evals(entropy_deriv, mus1, additional_parameters)
        muxy1[x, y] -= _TWO_EPS
    elif direction == "n":
        n1[x] += _EPS
        mus1 = Matching(muxy, n1, m)
        der_entropy_plus = get_evals(entropy_deriv, mus1, additional_parameters)
        n1[x] -= _TWO_EPS
    elif direction == "m":
        m1[y] += _EPS
        mus1 = Matching(muxy, n, m1)
        der_entropy_plus = get_evals(entropy_deriv, mus1, additional_parameters)
        m1[y] -= _TWO_EPS
    else:
        bs_error_abort("Wrong direction parameter.")

    mus1 = Matching(muxy1, n1, m1)
    der_entropy_minus = get_evals(entropy_deriv, mus1, additional_parameters)
    deriv_value = (der_entropy_plus[x, y] - der_entropy_minus[x, y]) / _TWO_EPS
    return cast(float, deriv_value)


def numeric_hessian(
    entropy: EntropyFunctions,
    muhat: Matching,
    alpha: np.ndarray | None = None,
    additional_parameters: list | None = None,
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
    # we create a derivative of entropy that is only a function of the Matching and the additional parameters
    if not parameter_dependent:
        entropy_deriv = partial(
            entropy_gradient,
            entropy,
        )
    else:
        entropy_deriv = partial(
            entropy_gradient,
            entropy,
            alpha=alpha,
        )
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
                    muhatf,
                    x,
                    y,
                    t,
                    entropy_deriv,
                    additional_parameters,
                    direction="y",
                )
            for z in range(X):
                hessian_y[x, y, z] = _numeric_component(
                    muhatf, x, y, z, entropy_deriv, additional_parameters, direction="x"
                )
            hessian_xy[x, y] = _numeric_component(
                muhatf, x, y, 0, entropy_deriv, additional_parameters, direction="d"
            )
    components_mumu = (hessian_x, hessian_y, hessian_xy)

    # now the hessian wrt (mu, r)
    hessian_n = np.zeros((X, Y))
    hessian_m = np.zeros((X, Y))
    for x in range(X):
        for y in range(Y):
            hessian_n[x, y] = _numeric_component(
                muhatf, x, y, 0, entropy_deriv, additional_parameters, direction="n"
            )
            hessian_m[x, y] = _numeric_component(
                muhatf, x, y, 0, entropy_deriv, additional_parameters, direction="m"
            )

    components_mur = (hessian_n, hessian_m)

    return components_mumu, components_mur


def fill_hessianMuMu_from_components(
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


def fill_hessianMuR_from_components(
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

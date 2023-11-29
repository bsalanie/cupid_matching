""" matching-related utilities """

from dataclasses import dataclass, field
from typing import Any, Final, Protocol, cast

import numpy as np
from bs_python_utils.bsnputils import TwoArrays, check_matrix, check_vector, npmaxabs
from bs_python_utils.bsutils import bs_error_abort

SINGLES_TOL: Final = 1e-3


def get_singles(muxy: np.ndarray, n: np.ndarray, m: np.ndarray) -> TwoArrays:
    """Computes the numbers of singles from the matches and the margins."""
    mux0 = n - np.sum(muxy, 1)
    mu0y = m - np.sum(muxy, 0)
    return mux0, mu0y


def compute_margins(muxy: np.ndarray, mux0: np.ndarray, mu0y: np.ndarray) -> TwoArrays:
    """Computes the margins from the matches and the singles."""
    n = np.sum(muxy, 1) + mux0
    m = np.sum(muxy, 0) + mu0y
    return n, m


def _check_no_singles(mu0: np.ndarray, n_households: float, str_gender: str) -> None:
    maxabs = npmaxabs(mu0)
    if maxabs > SINGLES_TOL * n_households:
        bs_error_abort(
            f"In a model w/o singles, we should not have any single {str_gender}."
        )


@dataclass
class Matching:
    """stores the numbers of couples and singles of every type;

    `muxy` is an (X,Y)-matrix
    `n` is an X-vector
    `m` is an Y-vector

    `no_singles`: if `True`, this is a model w/o singles

    `mux0` and `mu0y` are generated as the corresponding numbers of singles
    as well as the total number of households `n_households`
    and the total number of individuals `n_individuals`
    """

    muxy: np.ndarray
    n: np.ndarray
    m: np.ndarray
    no_singles: bool = False

    mux0: np.ndarray = field(init=False)
    mu0y: np.ndarray = field(init=False)
    n_households: float = field(init=False)
    n_individuals: float = field(init=False)

    def __str__(self):
        X, Y = self.muxy.shape
        n_couples = np.sum(self.muxy)
        n_men, n_women = np.sum(self.n), np.sum(self.m)
        repr_str = f"This is a matching with {n_men} men and {n_women} women.\n"
        repr_str += f"   with {n_couples} couples,\n"
        if self.no_singles:
            repr_str += "     and no singles.\n"
        repr_str += f"\n We have {X} types of men and {Y} of women."
        return repr_str

    def __post_init__(self):
        X, Y = check_matrix(self.muxy)
        Xn = check_vector(self.n)
        Ym = check_vector(self.m)
        if Xn != X:
            bs_error_abort(f"muxy is a ({X}, {Y}) matrix but n has {Xn} elements.")
        if Ym != Y:
            bs_error_abort(f"muxy is a ({X}, {Y}) matrix but m has {Ym} elements.")
        self.mux0, self.mu0y = get_singles(self.muxy, self.n, self.m)
        self.n_households = np.sum(self.muxy) + np.sum(self.mux0) + np.sum(self.mu0y)
        if self.no_singles:
            _check_no_singles(self.mux0, self.n_households, "men")
            _check_no_singles(self.mu0y, self.n_households, "women")
        self.n_individuals = (
            2.0 * np.sum(self.muxy) + np.sum(self.mux0) + np.sum(self.mu0y)
        )

    def unpack(self):
        muxy, mux0, mu0y = self.muxy, self.mux0, self.mu0y
        min_xy, min_x0, min_0y = np.min(muxy), np.min(mux0), np.min(mu0y)
        if min_xy < 0.0:
            bs_error_abort(f"The smallest muxy is {min_xy}")
        if not self.no_singles:
            if min_x0 < 0.0:
                bs_error_abort(f"The smallest mux0 is {min_x0}")
            if min_0y < 0.0:
                bs_error_abort(f"The smallest mux0 is {min_0y}")
        return muxy, mux0, mu0y, self.n, self.m


class MatchingFunction(Protocol):
    def __call__(
        self, mus: Matching, additional_parameters: list | None = ..., /
    ) -> np.ndarray:
        ...


def get_evals(
    fun: MatchingFunction, mus: Matching, additional_parameters: list | None = None
) -> np.ndarray:
    """evaluates fun(mus, additional_parameters)"""
    vals = fun(mus, additional_parameters)
    return cast(np.ndarray, vals)


def get_margins(mus: Matching) -> TwoArrays:
    """computes the numbers of each type from the matching patterns"""
    _, _, _, n, m = mus.unpack()
    return n, m


def simulate_sample_from_mus(
    mus: Matching, n_households: int, no_singles: bool = False, seed: int | None = None
) -> Matching:
    """Draw a sample of `n_households` from the matching patterns in `mus`

    Args:
        mus: the matching patterns
        n_households: the number of households requested
        no_singles: if `True`, this is a model w/o singles
        seed: an integer seed for the random number generator

    Returns:
        the sample matching patterns
    """
    rng = np.random.default_rng(seed)
    muxy, mux0, mu0y, _, _ = mus.unpack()
    X, Y = muxy.shape
    # stack all probabilities
    XY = X * Y
    # make sure we have no zeros
    _MU_EPS = min(1, int(1e-3 * n_households))
    if no_singles:
        pvec = muxy.reshape(XY)
        pvec /= np.sum(pvec)
        matches = rng.multinomial(n_households, pvec)
        muxy_sim = matches.reshape((X, Y))
        mux0_sim = np.full(X, _MU_EPS)
        mu0y_sim = np.full(Y, _MU_EPS)
    else:
        num_choices = XY + X + Y
        pvec = np.zeros(num_choices)
        pvec[:XY] = muxy.reshape(XY)
        pvec[XY : (XY + X)] = mux0
        pvec[(XY + X) :] = mu0y
        pvec /= np.sum(pvec)
        matches = rng.multinomial(n_households, pvec)
        muxy_sim = matches[:XY].reshape((X, Y))
        mux0_sim = matches[XY : (XY + X)]
        mu0y_sim = matches[(XY + X) :]
        muxy_sim += _MU_EPS
        mux0_sim += _MU_EPS
        mu0y_sim += _MU_EPS
    n_sim, m_sim = compute_margins(muxy_sim, mux0_sim, mu0y_sim)
    mus_sim = Matching(muxy=muxy_sim, n=n_sim, m=m_sim, no_singles=no_singles)
    return mus_sim


@dataclass
class VarianceMatching:
    """initialized with the six matrix components of the variance of a `Matching;  computes five more components.

    `var_xyzt` is the (XY, XY) var-cov matrix of `muxy`
    `var_xyz0` is the (XY, X) covariance matrix of `muxy` and `mux0`
    `var_xy0t` is the (XY, Y) covariance matrix of `muxy` and `mu0y`
    `var_x0z0` is the (X, X) var-cov matrix of `mux0`
    `var_x00t` is the (X, Y) covariance matrix of `mux0` and `mu0y`
    `var_0y0t` is the (Y, Y) var-cov matrix of `mu0y`

    `var_xyn` is the (XY, X) covariance matrix of `muxy` and `nx`
    `var_xym` is the (XY, Y) covariance matrix of `muxy` and `my`
    `var_nn` is the (X, X) var-cov matrix of `nx`
    `var_nm` is the (X, Y) covariance matrix of `nx` and `my`
    `var_mm` is the (Y, Y) var-cov matrix of `my`

    `var_allmus` is the (XY+X+Y, XY+X+Y) var-cov matrix of (muxy, mux0, mu0y)
    `var_munm` is the (XY+X+Y, XY+X+Y) var-cov matrix of (muxy, n, m)

    `no_singles`: if `True`, this is a model w/o singles;
        then `var_allmus` and `var_munm` are `(XY, XY)` and `(XY, XY+X+Y)`matrices
    """

    var_xyzt: np.ndarray
    var_xyz0: np.ndarray
    var_xy0t: np.ndarray
    var_x0z0: np.ndarray
    var_x00t: np.ndarray
    var_0y0t: np.ndarray

    no_singles: bool = False

    var_xyn: np.ndarray = field(init=False)
    var_xym: np.ndarray = field(init=False)
    var_nn: np.ndarray = field(init=False)
    var_nm: np.ndarray = field(init=False)
    var_mm: np.ndarray = field(init=False)

    var_allmus: np.ndarray = field(init=False)
    var_munm: np.ndarray = field(init=False)

    def __str__(self):
        n_men = self.var_xyz0.shape[1]
        n_women = self.var_xy0t.shape[1]
        repr_str = f"This is a VarianceMatching with {n_men}  men, {n_women} women.\n"
        if self.no_singles:
            repr_str += "    we have  no singles.\n\n"
        return repr_str

    def __post_init__(self):
        v_xyzt = self.var_xyzt
        XY, XY2 = check_matrix(v_xyzt)
        if XY2 != XY:
            bs_error_abort(f"var_xyzt should be a square matrix, not ({XY}, {XY2})")
        v_xyz0 = self.var_xyz0
        XY3, X = check_matrix(v_xyz0)
        if XY3 != XY:
            bs_error_abort(f"var_xyz0 should have {XY} rows, not {XY3})")
        v_xy0t = self.var_xy0t
        XY4, Y = check_matrix(v_xy0t)
        if XY4 != XY:
            bs_error_abort(f"var_xy0t should have {XY} rows, not {XY4})")
        if X * Y != XY:
            bs_error_abort(
                f"var_xyzt has {XY} rows, but varxyz0 has {X} columns and varxy0t"
                f" has {Y}"
            )
        v_x0z0 = self.var_x0z0
        X2, X3 = check_matrix(v_x0z0)
        if X2 != X:
            bs_error_abort(f"var_x0z0 has {X2} rows, it should have {X}")
        if X3 != X:
            bs_error_abort(f"var_x0z0 has {X3} columns, it should have {X}")
        v_x00t = self.var_x00t
        X4, Y2 = check_matrix(v_x00t)
        if X4 != X:
            bs_error_abort(f"var_x00t has {X4} rows, it should have {X}")
        if Y2 != Y:
            bs_error_abort(f"var_x00t has {Y2} columns, it should have {Y}")
        v_0y0t = self.var_0y0t
        Y3, Y4 = check_matrix(v_0y0t)
        if Y3 != Y:
            bs_error_abort(f"var_x00t has {Y3} rows, it should have {Y}")
        if Y4 != Y:
            bs_error_abort(f"var_x00t has {Y4} columns, it should have {Y}")

        # now we compute the additional components
        if self.no_singles:
            v_xyn = np.zeros_like(v_xyz0)
            iz = 0
            for z in range(X):
                v_xyn[:, z] += np.sum(v_xyzt[:, iz : (iz + Y)], 1)
                iz += Y
            self.var_xyn = v_xyn
            v_xym = np.zeros_like(v_xy0t)
            for t in range(Y):
                slice_t = slice(t, XY, Y)
                v_xym[:, t] += np.sum(v_xyzt[:, slice_t], 1)
            self.var_xym = v_xym
            v_nn = np.zeros_like(v_x0z0)
            ix = 0
            for x in range(X):
                v_nn[x, :] += np.sum(v_xyn[ix : (ix + Y), :], 0)
                ix += Y
            self.var_nn = v_nn
            v_nm = np.zeros_like(v_x00t)
            v_mm = np.zeros_like(v_0y0t)
            for y in range(Y):
                slice_y = slice(y, XY, Y)
                v_nm[:, y] += np.sum(v_xyn[slice_y, :], 0)
                v_mm[y, :] += np.sum(v_xym[slice_y, :], 0)
            self.var_nm = v_nm
            self.var_mm = v_mm
        else:
            v_xyn = v_xyz0.copy()
            sumt_covx0_zt = np.zeros((X, X))
            iz = 0
            for z in range(X):
                v_xyn[:, z] += np.sum(v_xyzt[:, iz : (iz + Y)], 1)
                sumt_covx0_zt[:, z] = np.sum(v_xyz0[iz : (iz + Y), :], 0)
                iz += Y
            self.var_xyn = v_xyn
            v_xym = v_xy0t.copy()
            v_0ym = v_0y0t.copy()
            for t in range(Y):
                slice_t = slice(t, XY, Y)
                v_xym[:, t] += np.sum(v_xyzt[:, slice_t], 1)
                v_0ym[:, t] += np.sum(v_xy0t[slice_t, :], 0)
            self.var_xym = v_xym
            v_x0n = v_x0z0 + sumt_covx0_zt
            v_nn = v_x0n
            v_0yn = v_x00t.copy().T
            ix = 0
            for x in range(X):
                v_nn[x, :] += np.sum(v_xyn[ix : (ix + Y), :], 0)
                v_0yn[:, x] += np.sum(v_xy0t[ix : (ix + Y), :], 0)
                ix += Y
            self.var_nn = v_nn
            v_nm = v_0yn.T
            v_mm = v_0ym
            for y in range(Y):
                slice_y = slice(y, XY, Y)
                v_nm[:, y] += np.sum(v_xyn[slice_y, :], 0)
                v_mm[y, :] += np.sum(v_xym[slice_y, :], 0)
            self.var_nm = v_nm
            self.var_mm = v_mm

        self.var_allmus = self.make_var_allmus()
        self.var_munm = self.make_var_munm()

    def unpack(self):
        """return a tuple of all members of this `VarianceMatching`"""
        return (
            self.var_xyzt,
            self.var_xyz0,
            self.var_xy0t,
            self.var_x0z0,
            self.var_x00t,
            self.var_0y0t,
            self.var_xyn,
            self.var_xym,
            self.var_nn,
            self.var_nm,
            self.var_mm,
        )

    def make_var_allmus(self: Any) -> np.ndarray:
        """create the variance-covariance of `(muxy, mux0, mu0y)`

        Args:
            self:  the `VarianceMatching` object

        Returns:
            an `(XY+X+Y, XY+X+Y)` symmetric positive matrix if there are singles; otherwise `(XY, XY)`
        """
        v_xyzt, v_xyz0, v_xy0t, v_x0z0, v_x00t, v_0y0t, *_ = self.unpack()
        X, Y = v_x0z0.shape[0], v_0y0t.shape[0]
        XY = X * Y

        if not self.no_singles:
            sz = XY + X + Y
            v_allmus = np.zeros((sz, sz))
            v_allmus[:XY, :XY] = v_xyzt
            v_allmus[:XY, XY : (XY + X)] = v_xyz0
            v_allmus[XY : (XY + X), :XY] = v_xyz0.T
            v_allmus[:XY, (XY + X) :] = v_xy0t
            v_allmus[(XY + X) :, :XY] = v_xy0t.T
            v_allmus[XY : (XY + X), XY : (XY + X)] = v_x0z0
            v_allmus[XY : (XY + X), (XY + X) :] = v_x00t
            v_allmus[(XY + X) :, XY : (XY + X)] = v_x00t.T
            v_allmus[(XY + X) :, (XY + X) :] = v_0y0t
        else:
            v_allmus = v_xyzt

        return v_allmus

    def make_var_munm(self: Any) -> np.ndarray:
        """create the variance-covariance of `(muxy, n, m)`

        Args:
            self:  this `VarianceMatching` object

        Returns:
            an `(XY+X+Y, XY+X+Y)` symmetric positive matrix.
        """
        v_xyzt, *_, v_xyn, v_xym, v_nn, v_nm, v_mm = self.unpack()
        X, Y = v_nn.shape[0], v_mm.shape[0]
        XY = X * Y
        sz = XY + X + Y
        v_munm = np.zeros((sz, sz))
        v_munm[:XY, :XY] = v_xyzt
        v_munm[:XY, XY : (XY + X)] = v_xyn
        v_munm[XY : (XY + X), :XY] = v_xyn.T
        v_munm[:XY, (XY + X) :] = v_xym
        v_munm[(XY + X) :, :XY] = v_xym.T
        v_munm[XY : (XY + X), XY : (XY + X)] = v_nn
        v_munm[XY : (XY + X), (XY + X) :] = v_nm
        v_munm[(XY + X) :, XY : (XY + X)] = v_nm.T
        v_munm[(XY + X) :, (XY + X) :] = v_mm

        return v_munm


def var_divide(varmus: VarianceMatching, d: float) -> VarianceMatching:
    """divide all members by the same number"""
    vardiv = VarianceMatching(
        varmus.var_xyzt / d,
        varmus.var_xyz0 / d,
        varmus.var_xy0t / d,
        varmus.var_x0z0 / d,
        varmus.var_x00t / d,
        varmus.var_0y0t / d,
    )
    return vardiv


def variance_muhat(muhat: Matching) -> VarianceMatching:
    """
    Computes the unweighted variance-covariance matrix of the observed matching patterns

    Args:
        muhat: a `Matching` object

    Returns:
        the corresponding `VarianceMatching` object
    """
    muxy, mux0, mu0y, *_ = muhat.unpack()
    X, Y = muxy.shape
    XY = X * Y

    # normalize all proportions
    n_households = muhat.n_households
    muxy_norm = (muxy / n_households).ravel()
    mux0_norm = mux0 / n_households
    mu0y_norm = mu0y / n_households

    # we construct the variance of (muxy, mux0, mu0y)
    # variance of muxy
    v_xyzt = np.diag(muxy_norm) - np.outer(muxy_norm, muxy_norm)
    if muhat.no_singles:
        v_xyz0 = np.zeros((XY, X))
        v_xy0t = np.zeros((XY, Y))
        v_x0z0 = np.zeros((X, X))
        v_x00t = np.zeros((X, Y))
        v_0y0t = np.zeros((Y, Y))
    else:
        # covariance of muxy and mux0
        v_xyz0 = -np.outer(muxy_norm, mux0_norm)
        # covariance of muxy and mu0y
        v_xy0t = -np.outer(muxy_norm, mu0y_norm)
        # variance of mux0
        v_x0z0 = np.diag(mux0_norm) - np.outer(mux0_norm, mux0_norm)
        # covariance of mux0 and mu0y
        v_x00t = -np.outer(mux0_norm, mu0y_norm)
        # variance of mu0y
        v_0y0t = np.diag(mu0y_norm) - np.outer(mu0y_norm, mu0y_norm)

    v_xyzt *= n_households
    v_xyz0 *= n_households
    v_xy0t *= n_households
    v_x0z0 *= n_households
    v_x00t *= n_households
    v_0y0t *= n_households

    varmus = VarianceMatching(
        var_xyzt=v_xyzt,
        var_xyz0=v_xyz0,
        var_xy0t=v_xy0t,
        var_x0z0=v_x0z0,
        var_x00t=v_x00t,
        var_0y0t=v_0y0t,
    )
    return varmus

"""Utility programs used in `min_distance.py`.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import scipy.linalg as spla
from bs_python_utils.bsnputils import ThreeArrays, TwoArrays, npmaxabs
from bs_python_utils.bsutils import bs_error_abort, print_stars

from cupid_matching.entropy import (
    fill_hessianMuMu_from_components,
    fill_hessianMuR_from_components,
)
from cupid_matching.matching_utils import Matching, variance_muhat


def check_args_mde(muhat: Matching, phi_bases: np.ndarray) -> tuple[int, int, int]:
    """check that the arguments to the MDE are consistent"""
    muxyhat, *_ = muhat.unpack()
    X, Y = muxyhat.shape
    ndims_phi = phi_bases.ndim
    if ndims_phi != 3:
        bs_error_abort(f"phi_bases should have 3 dimensions, not {ndims_phi}")
    Xp, Yp, K = phi_bases.shape
    if Xp != X or Yp != Y:
        bs_error_abort(
            f"phi_bases should have shape ({X}, {Y}, {K}) not ({Xp}, {Yp}, {K})"
        )
    return X, Y, K


def get_initial_weighting_matrix(
    parameterized_entropy: bool, initial_weighting_matrix: np.ndarray | None, XY: int
) -> np.ndarray | None:
    """returns the initial weighting matrix for the MDE when the entropy is parameterized

    Args:
        parameterized_entropy: if `True`, the entropy has unknown parameters
        initial_weighting_matrix: the initial weighting matrix, if provided
        XY: = X*Y

    Returns:
        the initial_weighting_matrix, or None if the entropy is not parameterized.
    """
    if parameterized_entropy:
        if initial_weighting_matrix is None:
            print_stars(
                "Using the identity matrix as weighting matrix in the first step."
            )
            S_mat = np.eye(XY)
        else:
            S_mat = initial_weighting_matrix
        return S_mat
    else:
        return None


def make_D2_matrix(X: int, Y: int) -> tuple[np.ndarray, int]:
    """create the double difference matrix for use w/o singles

    Args:
        X: number of types of men
        Y: number of types of women

    Returns:
        an (r, XY) matrix and  its rank r
    """
    XY = X * Y
    D2_mat = np.ones((XY, XY)) / XY + np.eye(XY)
    for x in range(X):
        slice_x = slice(x * Y, x * Y + Y)
        D2_mat[slice_x, slice_x] -= 1.0 / Y
    for y in range(Y):
        slice_y = slice(y, XY, Y)
        D2_mat[slice_y, slice_y] -= 1.0 / X
    rank_D2 = np.linalg.matrix_rank(D2_mat)
    rng = np.random.default_rng(453)
    A_mat = rng.uniform(size=(rank_D2, XY))
    D2_mat = A_mat @ D2_mat
    rank_D2 = np.linalg.matrix_rank(D2_mat)
    print(f"\nThe rank of the double difference matrix D2 is {rank_D2}.\n")
    return D2_mat, rank_D2


def check_indep_phi_no_singles(D2_phi: np.ndarray, X: int, Y: int) -> None:
    """check that the double difference of the phi matrix has full column rank;
        if so, return it

    Args:
        D2_phi: an $(X*Y, K)$ matrix of double differences
        X: number of types of men
        Y: number of types of women

    Returns:
        nothing
    """
    K = D2_phi.shape[1]
    actual_rank = np.linalg.matrix_rank(D2_phi)  # Compute the matrix rank
    if actual_rank != D2_phi.shape[1]:
        bs_error_abort(
            f"We have {K} basis functions but phi_mat only has rank {actual_rank}."
        )


def make_hessian_mde(
    hessian_components_mumu: ThreeArrays, hessian_components_mur: TwoArrays
) -> np.ndarray:
    """reconstitute the Hessian of the entropy function from its components

    Args:
        hessian_components_mumu:  the components of the Hesssian wrt $(\\mu,\\mu)$
        hessian_components_mur: the components of the Hesssian wrt $(\\mu,r)$

    Returns:
        np.ndarray: _description_
    """
    hessian_mumu = fill_hessianMuMu_from_components(hessian_components_mumu)
    hessian_mur = fill_hessianMuR_from_components(hessian_components_mur)
    hessians_both = np.concatenate((hessian_mumu, hessian_mur), axis=1)
    return cast(np.ndarray, hessians_both)


def get_optimal_weighting_matrix(
    muhat: Matching,
    hessians_both: np.ndarray,
    no_singles: bool = False,
    D2_mat: np.ndarray | None = None,
) -> np.ndarray:
    """compute the $S^\ast$ matrix used in the second step of the MDE

    Args:
        muhat: the observed `Matching`
        hessians_both: the Hessian of the entropy function
    """
    var_muhat = variance_muhat(muhat)
    var_munm = var_muhat.var_munm
    var_entropy_gradient = hessians_both @ var_munm @ hessians_both.T
    if no_singles:
        if D2_mat is None:
            bs_error_abort("D2_mat should not be None when no_singles is True")
        else:
            var_entropy_gradient = D2_mat @ var_entropy_gradient @ D2_mat.T
    S_mat = spla.inv(var_entropy_gradient)
    return cast(np.ndarray, S_mat)


def compute_estimates(
    M: np.ndarray, S_mat: np.ndarray, d: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Returns the QGLS estimates and their variance-covariance.

    Args:
        M: an (XY,p) matrix
        S_mat: an (XY, XY) weighting matrix
        d: an XY-vector

    Returns:
        the p-vector of estimates and their estimated (p,p) variance
    """
    M_T = M.T
    M_S_d = M_T @ S_mat @ d
    M_S_M = M_T @ S_mat @ M
    est_coeffs = -spla.solve(M_S_M, M_S_d)
    varcov_coeffs = spla.inv(M_S_M)
    return est_coeffs, varcov_coeffs


@dataclass
class MDEResults:
    """
    The results from minimum-distance estimation and testing.

    Args:
        X: the number of types of men
        Y: the number of types of women
        K: the number of bases
        number_households: the number of households in the sample
        estimated_coefficients: the estimated coefficients
        varcov_coefficients: their eetimated var-covar
        stderrs_coefficients: their estimated stderrs
        estimated_Phi: the estimated joint surplus
        test_statistic: the value of the misspecification statistic
        test_pvalue: the p-value of the test
        ndf: the number of degrees of freedom
        parameterized_entropy: True if the derivative of the entropy has unknown parameters
    """

    X: int
    Y: int
    K: int
    number_households: int
    estimated_coefficients: np.ndarray
    varcov_coefficients: np.ndarray
    stderrs_coefficients: np.ndarray
    estimated_Phi: np.ndarray
    test_statistic: float
    test_pvalue: float
    ndf: int
    parameterized_entropy: bool | None = False

    def __str__(self):
        line_stars = "*" * 80 + "\n"
        if self.parameterized_entropy:
            n_alpha = self.estimated_coefficients.size - self.K
            entropy_str = f"     The entropy has {n_alpha} parameters."
        else:
            entropy_str = "     The entropy is parameter-free."
            n_alpha = 0
        model_str = f"The data has {self.number_households} households\n\n"
        model_str += f"The model has {self.X}x{self.Y} margins\n {entropy_str} \n"
        model_str += f"We use {self.K} basis functions.\n\n"
        repr_str = line_stars + model_str
        repr_str += "The estimated coefficients (and their standard errors) are\n\n"
        if self.parameterized_entropy:
            for i, coeff in enumerate(self.estimated_coefficients[:n_alpha]):
                repr_str += (
                    f"   alpha({i + 1}): {coeff: > 10.3f}  "
                    + f"({self.stderrs_coefficients[i]: .3f})\n"
                )
            repr_str += "\n"
        for i, coeff in enumerate(self.estimated_coefficients[n_alpha:]):
            repr_str += (
                f"   base {i + 1}: {coeff: > 10.3f} "
                + f"({self.stderrs_coefficients[n_alpha + i]: .3f})\n"
            )
        repr_str += "\nSpecification test:\n"
        repr_str += (
            f"   the value of the test statistic is {self.test_statistic: > 10.3f}\n"
        )
        repr_str += (
            f"     for a chi2({self.ndf}), the p-value is {self.test_pvalue: > 10.3f}\n"
        )
        return repr_str + line_stars

    def print_results(
        self, true_coeffs: np.ndarray | None = None, n_alpha: int = 0
    ) -> None | float:
        estimates = self.estimated_coefficients
        stderrs = self.stderrs_coefficients

        if true_coeffs is not None:
            repr_str = (
                "The true and estimated coefficients "
                + "(and their standard errors) are\n\n"
            )
            for i, coeff in enumerate(estimates[:n_alpha]):
                repr_str += f"   alpha({i + 1}): {true_coeffs[i]: > 10.3f}"
                repr_str += f"{coeff: > 10.3f}  ({stderrs[i]: > 10.3f})\n"
                repr_str += "\n"
            for i, coeff in enumerate(estimates[n_alpha:]):
                j = n_alpha + i
                repr_str += (
                    f"   base {i + 1}: {true_coeffs[j]: > 10.3f}  "
                    + f"{coeff: > 10.3f}  ({stderrs[j]: > 10.3f})\n"
                )
            print_stars(repr_str)
            discrepancy = npmaxabs(true_coeffs - estimates)
            print_stars(
                "The largest difference between true and estimated coefficients is"
                f" {discrepancy: .2e}"
            )
        else:
            repr_str = (
                "The estimated coefficients " + "(and their standard errors) are\n\n"
            )
            for i, coeff in enumerate(estimates[:n_alpha]):
                repr_str + f"{coeff: > 10.3f}  ({stderrs[i]: > 10.3f})\n"
                repr_str += "\n"
            for i, coeff in enumerate(estimates[n_alpha:]):
                j = n_alpha + i
                repr_str += f"{coeff: > 10.3f}  ({stderrs[j]: > 10.3f})\n"
            print_stars(repr_str)

        repr_str = "\nSpecification test:\n"
        repr_str += (
            "   the value of the test statistic is "
            + f"{self.test_statistic: > 10.3f}\n"
        )
        repr_str += (
            f"     for a chi2({self.ndf}), "
            + f"the p-value is {self.test_pvalue: > 10.3f}\n"
        )
        print_stars(repr_str)
        if true_coeffs is not None:
            return cast(float, discrepancy)
        return None

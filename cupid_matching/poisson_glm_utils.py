"""Utilities for Poisson GLM.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
from bs_python_utils.bsnputils import npmaxabs
from bs_python_utils.bsutils import bs_error_abort, print_stars

from cupid_matching.matching_utils import Matching, VarianceMatching, var_divide


@dataclass
class PoissonGLMResults:
    """Stores and formats the estimation results.

    Args:
        X: int
        Y: int
        K: int
        number_households: int
        number_individuals: int
        estimated_gamma: np.ndarray
        varcov_gamma: np.ndarray
        stderrs_gamma: np.ndarray
        estimated_beta: np.ndarray
        estimated_u: np.ndarray
        estimated_v: np.ndarray
        varcov_beta: np.ndarray
        stderrs_beta: np.ndarray
        stderrs_u: np.ndarray
        stderrs_v: np.ndarray
        estimated_Phi: np.ndarray
    """

    X: int
    Y: int
    K: int
    number_households: int
    number_individuals: int
    estimated_gamma: np.ndarray
    varcov_gamma: np.ndarray
    stderrs_gamma: np.ndarray
    estimated_beta: np.ndarray
    varcov_beta: np.ndarray
    estimated_u: np.ndarray
    estimated_v: np.ndarray
    stderrs_beta: np.ndarray
    stderrs_u: np.ndarray
    stderrs_v: np.ndarray
    estimated_Phi: np.ndarray

    def __str__(self):
        line_stars = "*" * 80 + "\n"
        print_stars("Estimating a Choo and Siow model by Poisson GLM.")
        model_str = f"The data has {self.number_households} households\n\n"
        model_str += f"We use {self.K} basis functions.\n\n"
        repr_str = line_stars + model_str
        repr_str += (
            "The estimated basis coefficients (and their standard errors) are\n\n"
        )
        for i in range(self.K):
            repr_str += (
                f"   base_{i + 1}: {self.estimated_beta[i]: > 10.3f}  "
                + f"({self.stderrs_beta[i]: .3f})\n"
            )
        repr_str += "The estimated utilities of men (and their standard errors) are\n\n"
        for i in range(self.X):
            repr_str += (
                f"   u_{i + 1}: {self.estimated_u[i]: > 10.3f}  "
                + f"({self.stderrs_u[i]: .3f})\n"
            )
        repr_str += (
            "The estimated utilities of women (and their standard errors) are\n\n"
        )
        for i in range(self.Y):
            repr_str += (
                f"   v {i + 1}: {self.estimated_v[i]: > 10.3f}  "
                + f"({self.stderrs_v[i]: .3f})\n"
            )
        return repr_str + line_stars

    def print_results(
        self,
        lambda_true: np.ndarray | None = None,
        u_true: np.ndarray | None = None,
        v_true: np.ndarray | None = None,
    ) -> float | None:
        estimates_beta = self.estimated_beta
        stderrs_beta = self.stderrs_beta

        if lambda_true is None:
            repr_str = "The  estimated coefficients "
            repr_str += "(and their standard errors) are\n\n"
            for i, coeff in enumerate(estimates_beta):
                repr_str += f" {coeff: > 10.3f}  ({stderrs_beta[i]: > 10.3f})\n"
            print_stars(repr_str)
        else:
            repr_str = "The  true and estimated coefficients "
            repr_str += "(and their standard errors) are\n\n"
            for i, coeff in enumerate(estimates_beta):
                repr_str += f"   base {i + 1}: {lambda_true[i]: > 10.3f} "
                repr_str += f" {coeff: > 10.3f}  ({stderrs_beta[i]: > 10.3f})\n"
            print_stars(repr_str)

        self.report_utilities("men", u_true)
        self.report_utilities("women", v_true)

        if lambda_true is None:
            return None
        else:
            discrepancy = npmaxabs(lambda_true - estimates_beta)
            print_stars(
                "The largest difference between true and estimated coefficients is"
                f" {discrepancy: .2e}"
            )
            return cast(float, discrepancy)

    def report_utilities(self, gender: str, utils_true: np.ndarray | None) -> None:
        if gender not in ["men", "women"]:
            bs_error_abort(f"gender can only be 'men' or 'women', not {gender}")
        utils_estimates = self.estimated_u if gender == "men" else self.estimated_v
        utils_stderrs = self.stderrs_u if gender == "men" else self.stderrs_v
        util_prefix = "u" if gender == "men" else "v"
        if utils_true is None:
            repr_str = f"The estimated utilities for {gender} "
            repr_str += "(and their standard errors) are:\n\n"
            for i, coeff in enumerate(utils_estimates):
                repr_str += f"   {util_prefix}_{i + 1}: "
                repr_str += f" {coeff: > 10.3f}  ({utils_stderrs[i]: > 10.3f})\n"
            print_stars(repr_str)
        else:
            repr_str = f"The true and estimated utilities for {gender} "
            repr_str += "(and their standard errors) are:\n\n"
            for i, coeff in enumerate(utils_estimates):
                repr_str += f"   {util_prefix}_{i + 1}: {utils_true[i]: > 10.3f} "
                repr_str += f" {coeff: > 10.3f}  ({utils_stderrs[i]: > 10.3f})\n"
            print_stars(repr_str)


def prepare_data(
    muhat: Matching,
    var_muhat: VarianceMatching,
    no_singles: bool = False,
) -> tuple[np.ndarray, VarianceMatching, int, int]:
    """Normalizes the matching patterns and stacks them.
    We rescale the data so that the total number of individuals is one.

    Args:
        muhat: the observed Matching
        var_muhat: the variance-covariance object for the observed matching
        no_singles: if True, we do not observe singles

    Returns:
        the stacked, normalized `muxy, mux0, mu0y` (the latter two are zero if `no_singles`)
        the corresponding variance-covariance matrix
        the number of households
        the number of individuals
    """
    muxy, mux0, mu0y, *_ = muhat.unpack()
    n_couples = np.sum(muxy)
    if no_singles:
        mux0 = np.zeros(mux0.shape)
        mu0y = np.zeros(mu0y.shape)
    n_households = n_couples + np.sum(mux0) + np.sum(mu0y)
    n_individuals = n_households + n_couples
    # rescale the data so that the total number of individuals is one
    muhat_norm = np.concatenate([muxy.flatten(), mux0, mu0y]) / n_individuals
    n_indivs2 = n_individuals * n_individuals
    var_muhat_norm = var_divide(var_muhat, n_indivs2)
    return (
        muhat_norm,
        var_muhat_norm,
        n_households,
        n_individuals,
    )

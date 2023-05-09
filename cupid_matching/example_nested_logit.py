""" example using a simple two-layer nested logit model
One nest on each side must consist of the 0 option.
The other nests are specified as nested lists.
E.g. [[1, 3], [2,4]] describes two nests, one with types 1 and 3,
and the other with types 2 and 4.
On each side, the nests are the same for each type, with the same parameters.
"""

from typing import cast

import numpy as np
from bs_python_utils.bsutils import print_stars

from cupid_matching.entropy import EntropyFunctions
from cupid_matching.min_distance import estimate_semilinear_mde
from cupid_matching.model_classes import Matching, NestedLogitPrimitives
from cupid_matching.nested_logit import setup_standard_nested_logit


def create_nestedlogit_population(
    X: int,
    Y: int,
    K: int,
    std_alphas: float = 0.5,
    std_betas: float = 1.0,
) -> tuple[
    NestedLogitPrimitives,
    np.ndarray,
    np.ndarray,
    EntropyFunctions,
    EntropyFunctions,
]:
    """
    we simulate a nested logit population
    with equal numbers of men and women of each type
    and random bases dunctions and coefficients

        Args:
         X: number of types of men
         Y: number of types of women
         K: random basis functions
         std_alphas: the nest parameters are drawn from a U[0, std_alphas] distribution
         std_betas: the coefficients of the bases are drawn from a centered normal
                     with this standard deviation

        Returns:
            a NestedLogitPrimitives instance, the basis functions, the true coefficients,
            and the entropy functions
    """
    X, Y, K = 10, 12, 5
    nests_for_each_x = [
        list(range(1, Y // 2 + 1)),
        list(range(Y // 2 + 1, Y + 1)),
    ]
    nests_for_each_y = [
        list(range(1, X // 2 + 1)),
        list(range(X // 2 + 1, X + 1)),
    ]

    n = np.ones(X)
    m = np.ones(Y)
    phi_bases = np.random.randn(X, Y, K)

    (
        entropy_nested_logit,
        entropy_nested_logit_numeric,
    ) = setup_standard_nested_logit(nests_for_each_x, nests_for_each_y)
    n_rhos, n_deltas = len(nests_for_each_x), len(nests_for_each_y)
    n_alphas = n_rhos + n_deltas

    betas_true = std_betas * np.random.randn(K)
    alphas_true = std_alphas * np.random.uniform(size=n_alphas)

    Phi = phi_bases @ betas_true
    nested_logit_instance = NestedLogitPrimitives(
        Phi, n, m, nests_for_each_x, nests_for_each_y, alphas_true
    )
    true_coeffs = np.concatenate((alphas_true, betas_true))
    return (
        nested_logit_instance,
        phi_bases,
        true_coeffs,
        entropy_nested_logit,
        entropy_nested_logit_numeric,
    )


def mde_estimate(
    mus_sim: Matching,
    phi_bases: np.ndarray,
    true_coeffs: np.ndarray,
    entropy: EntropyFunctions,
    title: str,
) -> float:
    """we estimate the parameters using the minimum distance estimator

    Args:
        mus_sim: a Choo and Siow Matching
        phi_bases: the basis functions
        true_coeffs: their true coefficients and  the nesting parameters
        entropy: the entropy functions we use
        title: the name of the estimator

    Returns:
        the largest absolute difference between the true and estimated coefficients
    """
    print_stars(f"    {title}")
    mde_results = estimate_semilinear_mde(
        mus_sim,
        phi_bases,
        entropy,
        additional_parameters=entropy.additional_parameters,
    )
    mde_discrepancy = mde_results.print_results(true_coeffs=true_coeffs)
    return cast(float, mde_discrepancy)


(
    nested_logit_instance,
    phi_bases,
    true_coeffs,
    entropy_nested_logit,
    entropy_nested_logit_numeric,
) = create_nestedlogit_population(20, 18, 6)
seed = 6475788
n_households = 1_000_000
mus_sim = nested_logit_instance.simulate(n_households, seed)
mde_discrepancy = mde_estimate(
    mus_sim,
    phi_bases,
    true_coeffs,
    entropy_nested_logit,
    "RESULTS FOR MDE WITH ANALYTICAL GRADIENT",
)
mde_discrepancy_numeric = mde_estimate(
    mus_sim,
    phi_bases,
    true_coeffs,
    entropy_nested_logit_numeric,
    "RESULTS FOR MDE WITH NUMERICAL GRADIENT",
)

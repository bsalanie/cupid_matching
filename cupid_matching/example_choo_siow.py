""" example using the Choo and Siow homoskedastic model"""

from typing import cast

import numpy as np
from bs_python_utils.bsutils import print_stars

from cupid_matching.choo_siow import (
    entropy_choo_siow,
    entropy_choo_siow_corrected,
    entropy_choo_siow_corrected_numeric,
    entropy_choo_siow_numeric,
)
from cupid_matching.entropy import EntropyFunctions
from cupid_matching.min_distance import estimate_semilinear_mde
from cupid_matching.model_classes import ChooSiowPrimitives, Matching
from cupid_matching.poisson_glm import choo_siow_poisson_glm


def create_choosiow_population(
    X: int, Y: int, K: int, std_betas: float
) -> tuple[ChooSiowPrimitives, np.ndarray, np.ndarray]:
    """
    we simulate a Choo and Siow population
    with equal numbers of men and women of each type
    and random bases functions and coefficients

        Args:
         X: number of types of men
         Y: number of types of women
         K: random basis functions
         std_betas: the coefficients are drawn from a centered normal
                     with this standard deviation

        Returns:
            a `ChooSiowPrimitives` instance, the basis functions, and the coefficients
    """
    betas_true = std_betas * np.random.randn(K)
    phi_bases = np.random.randn(X, Y, K)
    n = np.ones(X)
    m = np.ones(Y)
    Phi = phi_bases @ betas_true
    choo_siow_instance = ChooSiowPrimitives(Phi, n, m)
    return choo_siow_instance, phi_bases, betas_true


def mde_estimate(
    mus_sim: Matching,
    phi_bases: np.ndarray,
    betas_true: np.ndarray,
    entropy: EntropyFunctions,
    no_singles: bool = False,
    title: str | None = None,
    verbose: bool = False,
) -> float:
    """we estimate the parameters using the minimum distance estimator

    Args:
        mus_sim: a Choo and Siow Matching
        phi_bases: the basis functions
        betas_true: their true coefficients
        entropy: the entropy functions we use
        no_singles: if `True`, we use the no-singles version of the model
        title: the name of the estimator

    Returns:
        the largest absolute difference between the true and estimated coefficients
    """
    print_stars(f"    {title}")
    mde_results = estimate_semilinear_mde(
        mus_sim,
        phi_bases,
        entropy,
        no_singles=no_singles,
        verbose=verbose,
    )
    mde_discrepancy = mde_results.print_results(true_coeffs=betas_true)
    return cast(float, mde_discrepancy)


def demo_choo_siow(
    n_households: int, X: int, Y: int, K: int, std_betas: float = 1.0
) -> tuple[float, float, float, float, float]:
    """run four MDE estimators and the Poisson estimator
    on randomly generated data

    Args:
        n_households: number of households
        X: number of types of men
        Y: number of types of women
        K: number of basis functions
        std_betas: the standard errors of their coefficients

    Returns:
        the discrepancies of the five estimators
    """
    choo_siow_instance, phi_bases, betas_true = create_choosiow_population(
        X, Y, K, std_betas
    )
    mus_sim = choo_siow_instance.simulate(n_households)

    # we estimate using four variants of the minimum distance estimator
    mde_discrepancy = mde_estimate(
        mus_sim,
        phi_bases,
        betas_true,
        entropy_choo_siow,
        title="RESULTS FOR MDE WITH ANALYTICAL GRADIENT",
    )
    mde_discrepancy_numeric = mde_estimate(
        mus_sim,
        phi_bases,
        betas_true,
        entropy_choo_siow_numeric,
        title="RESULTS FOR MDE WITH NUMERICAL GRADIENT",
    )
    mde_discrepancy_corrected = mde_estimate(
        mus_sim,
        phi_bases,
        betas_true,
        entropy_choo_siow_corrected,
        title="RESULTS FOR THE CORRECTED MDE WITH ANALYTICAL GRADIENT",
    )
    mde_discrepancy_corrected_numeric = mde_estimate(
        mus_sim,
        phi_bases,
        betas_true,
        entropy_choo_siow_corrected_numeric,
        title="RESULTS FOR THE CORRECTED MDE WITH NUMERICAL GRADIENT",
    )

    # we also estimate using Poisson GLM
    print_stars("    RESULTS FOR POISSON   ")
    poisson_results = choo_siow_poisson_glm(mus_sim, phi_bases)
    _, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()
    poisson_discrepancy = poisson_results.print_results(
        betas_true,
        u_true=-np.log(mux0_sim / n_sim),
        v_true=-np.log(mu0y_sim / m_sim),
    )
    return (
        mde_discrepancy,
        mde_discrepancy_numeric,
        mde_discrepancy_corrected,
        mde_discrepancy_corrected_numeric,
        cast(float, poisson_discrepancy),
    )


if __name__ == "__main__":
    n_households = 1_000_000
    X, Y = 6, 4
    K = 3
    std_betas = 0.5
    (
        mde_discrepancy,
        mde_discrepancy_numeric,
        mde_discrepancy_corrected,
        mde_discrepancy_corrected_numeric,
        poisson_discrepancy,
    ) = demo_choo_siow(n_households, X, Y, K, std_betas=std_betas)

    print_stars(
        "Largest absolute differences between the true and estimated coefficients:"
    )
    print(f"MDE:                            {mde_discrepancy: .2e}")
    print(f"MDE numeric:                    {mde_discrepancy_numeric: .2e}")
    print(f"MDE corrected:                  {mde_discrepancy_corrected: .2e}")
    print(f"MDE corrected numeric:          {mde_discrepancy_corrected_numeric: .2e}")
    print(f"Poisson:                        {poisson_discrepancy: .2e}")

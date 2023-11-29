""" example using the Choo and Siow homoskedastic model without singles"""

from typing import cast

import numpy as np
from bs_python_utils.bsutils import print_stars

from cupid_matching.choo_siow_no_singles import (
    entropy_choo_siow_no_singles,
    entropy_choo_siow_no_singles_corrected,
    entropy_choo_siow_no_singles_corrected_numeric,
    entropy_choo_siow_no_singles_numeric,
)
from cupid_matching.example_choo_siow import mde_estimate
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import choo_siow_poisson_glm


def create_choosiow_no_singles_population(
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
            a `ChooSiowPrimitivesNoSingles` instance, the basis functions, and the coefficients
    """
    betas_true = std_betas * np.random.randn(K)
    phi_bases = np.random.randn(X, Y, K)
    n = np.ones(X)
    m = np.full(Y, X / Y)  # we need as many women as men overall
    Phi = phi_bases @ betas_true
    choo_siow_instance = ChooSiowPrimitives(Phi, n, m)
    return choo_siow_instance, phi_bases, betas_true


def demo_choo_siow_no_singles(
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
    choo_siow_instance, phi_bases, betas_true = create_choosiow_no_singles_population(
        X, Y, K, std_betas
    )
    mus_sim = choo_siow_instance.simulate(n_households)

    # we estimate using four variants of the minimum distance estimator
    mde_discrepancy = mde_estimate(
        mus_sim,
        phi_bases,
        betas_true,
        entropy_choo_siow_no_singles,
        no_singles=True,
        title="RESULTS FOR MDE WITH ANALYTICAL GRADIENT",
    )
    mde_discrepancy_numeric = mde_estimate(
        mus_sim,
        phi_bases,
        betas_true,
        entropy_choo_siow_no_singles_numeric,
        no_singles=True,
        title="RESULTS FOR MDE WITH NUMERICAL GRADIENT",
    )
    mde_discrepancy_corrected = mde_estimate(
        mus_sim,
        phi_bases,
        betas_true,
        entropy_choo_siow_no_singles_corrected,
        no_singles=True,
        title="RESULTS FOR THE CORRECTED MDE WITH ANALYTICAL GRADIENT",
    )
    mde_discrepancy_corrected_numeric = mde_estimate(
        mus_sim,
        phi_bases,
        betas_true,
        entropy_choo_siow_no_singles_corrected_numeric,
        no_singles=True,
        title="RESULTS FOR THE CORRECTED MDE WITH NUMERICAL GRADIENT",
    )

    # we also estimate using Poisson GLM
    print_stars("    RESULTS FOR POISSON   ")
    poisson_results = choo_siow_poisson_glm(mus_sim, phi_bases, no_singles=True)
    _, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()
    # we normalize u_1 = 0
    u_true = -np.log(mux0_sim / n_sim)
    v_true = -np.log(mu0y_sim / m_sim) + u_true[0]
    u_true -= u_true[0]
    poisson_discrepancy = poisson_results.print_results(
        betas_true,
        u_true,
        v_true,
    )
    print("\n   (we normalized the utility u_1 at 0)\n")
    return (
        mde_discrepancy,
        mde_discrepancy_numeric,
        mde_discrepancy_corrected,
        mde_discrepancy_corrected_numeric,
        cast(float, poisson_discrepancy),
    )


if __name__ == "__main__":
    n_households = 1_000_000
    X, Y = 10, 15
    K = 8
    std_betas = 0.5
    (
        mde_discrepancy,
        mde_discrepancy_numeric,
        mde_discrepancy_corrected,
        mde_discrepancy_corrected_numeric,
        poisson_discrepancy,
    ) = demo_choo_siow_no_singles(n_households, X, Y, K, std_betas=std_betas)

    print_stars(
        "Largest absolute differences between the true and estimated coefficients:"
    )
    print(f"MDE:                            {mde_discrepancy: .2e}")
    print(f"MDE numeric:                    {mde_discrepancy_numeric: .2e}")
    print(f"MDE corrected:                  {mde_discrepancy_corrected: .2e}")
    print(f"MDE corrected numeric:          {mde_discrepancy_corrected_numeric: .2e}")
    print(f"Poisson:                        {poisson_discrepancy: .2e}")

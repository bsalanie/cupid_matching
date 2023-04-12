""" example using the Choo and Siow homoskedastic model"""

from typing import Optional, Tuple

import numpy as np

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
from cupid_matching.utils import print_stars


def create_choosiow_population(
    X: int, Y: int, K: int, std_betas: Optional[float] = 1.0
) -> Tuple[ChooSiowPrimitives, np.ndarray, np.ndarray]:
    """
    we simulate a Choo and Siow population
    with equal numbers of men and women of each type
    and random bases dunctions and coefficients

        Args:
         X: number of types of men
         Y: number of types of women
         K: random basis functions
         std_betas: the coefficients are drawn from a centered normal
                     with this standard deviation

        Returns:
            a ChooSiowPrimitives instance, the basis functions, and the coefficients
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
    title: str,
) -> float:
    """we estimate the parameters using the minimum distance estimator

    Args:
        mus_sim: a Choo and Siow Matching
        phi_bases: the basis functions
        betas_true: their true coefficients
        entropy: the entropy functions we use
        title: the name of the estimator

    Returns:
        the largest absolute difference between the true and estimated coefficients
    """
    print_stars(f"    {title}")
    mde_results = estimate_semilinear_mde(mus_sim, phi_bases, entropy)
    mde_discrepancy = mde_results.print_results(true_coeffs=betas_true)
    return mde_discrepancy


choo_siow_instance, phi_bases, betas_true = create_choosiow_population(10, 8, 5)
mus_sim = choo_siow_instance.simulate(1e5)

# we estimate using forur variants of the minimum distance estimator
mde_discrepancy = mde_estimate(
    mus_sim,
    phi_bases,
    betas_true,
    entropy_choo_siow,
    "RESULTS FOR MDE WITH ANALYTICAL GRADIENT",
)
mde_discrepancy_numeric = mde_estimate(
    mus_sim,
    phi_bases,
    betas_true,
    entropy_choo_siow_numeric,
    "RESULTS FOR MDE WITH NUMERICAL GRADIENT",
)
mde_discrepancy_corrected = mde_estimate(
    mus_sim,
    phi_bases,
    betas_true,
    entropy_choo_siow_corrected,
    "RESULTS FOR THE CORRECTED MDE WITH ANALYTICAL GRADIENT",
)
mde_discrepancy_corrected_numeric = mde_estimate(
    mus_sim,
    phi_bases,
    betas_true,
    entropy_choo_siow_corrected_numeric,
    "RESULTS FOR THE CORRECTED MDE WITH NUMERICAL GRADIENT",
)

# we also estimate using Poisson GLM
print_stars("    RESULTS FOR POISSON   ")
poisson_results = choo_siow_poisson_glm(mus_sim, phi_bases)
muxy_sim, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()
poisson_discrepancy = poisson_results.print_results(
    betas_true,
    u_true=-np.log(mux0_sim / n_sim),
    v_true=-np.log(mu0y_sim / m_sim),
)

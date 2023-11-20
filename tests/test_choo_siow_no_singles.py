import numpy as np
from bs_python_utils.bsutils import print_stars

from cupid_matching.choo_siow_no_singles import (
    entropy_choo_siow_no_singles,
)
from cupid_matching.example_choo_siow import mde_estimate
from cupid_matching.model_classes import ChooSiowPrimitivesNoSingles


def create_choosiow_population_no_singles(
    X: int, Y: int, K: int, std_betas: float
) -> tuple[ChooSiowPrimitivesNoSingles, np.ndarray, np.ndarray]:
    """
    we simulate a Choo and Siow population w/o singles
    with random bases functions and coefficients

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
    phi_bases = np.zeros((X, Y, K))
    range_X, range_Y = np.arange(1, X + 1) / (X + 1), np.arange(1, Y + 1) / (Y + 1)
    for k in range(1, K + 1):
        phi_bases[:, :, k - 1] = np.outer(range_X**k, range_Y)
    n = np.ones(X)
    m = np.full(Y, X / float(Y))  # we want as many men as women
    Phi = phi_bases @ betas_true
    choo_siow_instance = ChooSiowPrimitivesNoSingles(Phi, n, m)
    return choo_siow_instance, phi_bases, betas_true


def demo_choo_siow_no_singles(
    n_households: int, X: int, Y: int, K: int, std_betas: float = 1.0
) -> tuple[float, float, float, float, float]:
    """run four MDE estimators and the Poisson estimator
    on randomly generated data w/o singles

    Args:
        n_households: number of households
        X: number of types of men
        Y: number of types of women
        K: number of basis functions
        std_betas: the standard errors of their coefficients

    Returns:
        the discrepancies of the five estimators
    """
    choo_siow_instance, phi_bases, betas_true = create_choosiow_population_no_singles(
        X, Y, K, std_betas
    )
    mus_sim = choo_siow_instance.simulate(n_households)
    # print(f"{(mus_sim.muxy**2)/np.exp(choo_siow_instance.Phi)=}")
    # print(f"{mus_sim.n=}")
    # print(f"{mus_sim.m=}")
    print(f"{mus_sim.muxy=}")

    # we estimate using four variants of the minimum distance estimator
    mde_discrepancy = mde_estimate(
        mus_sim,
        phi_bases,
        betas_true,
        entropy_choo_siow_no_singles,
        no_singles=True,
        title="RESULTS FOR MDE WITH ANALYTICAL GRADIENT",
    )
    # mde_discrepancy_numeric = mde_estimate(
    #     mus_sim,
    #     phi_bases,
    #     betas_true,
    #     entropy_choo_siow_no_singles_numeric,
    #     "RESULTS FOR MDE WITH NUMERICAL GRADIENT",
    # )
    # mde_discrepancy_corrected = mde_estimate(
    #     mus_sim,
    #     phi_bases,
    #     betas_true,
    #     entropy_choo_siow_no_singles_corrected,
    #     "RESULTS FOR THE CORRECTED MDE WITH ANALYTICAL GRADIENT",
    # )
    # mde_discrepancy_corrected_numeric = mde_estimate(
    #     mus_sim,
    #     phi_bases,
    #     betas_true,
    #     entropy_choo_siow_no_singles_corrected_numeric,
    #     "RESULTS FOR THE CORRECTED MDE WITH NUMERICAL GRADIENT",
    # )

    # # we also estimate using Poisson GLM
    # print_stars("    RESULTS FOR POISSON   ")
    # poisson_results = choo_siow_poisson_glm(mus_sim, phi_bases)
    # _, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()
    # poisson_discrepancy = poisson_results.print_results(
    #     betas_true,
    #     u_true=-np.log(mux0_sim / n_sim),
    #     v_true=-np.log(mu0y_sim / m_sim),
    # )
    return (
        mde_discrepancy,
        # mde_discrepancy_numeric,
        # mde_discrepancy_corrected,
        # mde_discrepancy_corrected_numeric,
        # cast(float, poisson_discrepancy),
    )


def test_choo_siow_no_singles():
    n_households = 100_000_000
    X, Y = 10, 15
    K = 2
    std_betas = 0.5
    TOL_CS = 1e-1

    (
        mde_discrepancy,
        # mde_discrepancy_numeric,
        # mde_discrepancy_corrected,
        # mde_discrepancy_corrected_numeric,
        # poisson_discrepancy,
    ) = demo_choo_siow_no_singles(n_households, X, Y, K, std_betas=std_betas)
    assert mde_discrepancy < TOL_CS
    # assert mde_discrepancy_numeric < TOL_CS
    # assert mde_discrepancy_corrected < TOL_CS
    # assert mde_discrepancy_corrected_numeric < TOL_CS
    # assert poisson_discrepancy < TOL_CS
    print_stars(
        "Largest absolute differences between the true and estimated coefficients:"
    )
    print(f"MDE:                            {mde_discrepancy: .2e}")
    # print(f"MDE numeric:                    {mde_discrepancy_numeric: .2e}")
    # print(f"MDE corrected:                  {mde_discrepancy_corrected: .2e}")
    # print(f"MDE corrected numeric:          {mde_discrepancy_corrected_numeric: .2e}")
    # print(f"Poisson:                        {poisson_discrepancy: .2e}")

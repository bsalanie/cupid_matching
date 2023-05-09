from cupid_matching.example_choo_siow import demo_choo_siow


def test_choo_siow():
    n_households = 100_000_000
    X, Y = 10, 15
    K = 5
    std_betas = 0.5
    TOL_CS = 1e-2
    (
        mde_discrepancy,
        mde_discrepancy_numeric,
        mde_discrepancy_corrected,
        mde_discrepancy_corrected_numeric,
        poisson_discrepancy,
    ) = demo_choo_siow(n_households, X, Y, K, std_betas=std_betas)
    assert mde_discrepancy < TOL_CS
    assert mde_discrepancy_numeric < TOL_CS
    assert mde_discrepancy_corrected < TOL_CS
    assert mde_discrepancy_corrected_numeric < TOL_CS
    assert poisson_discrepancy < TOL_CS

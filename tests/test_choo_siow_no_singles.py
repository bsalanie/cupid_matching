from cupid_matching.example_choo_siow_no_singles import demo_choo_siow_no_singles


def test_choo_siow_no_singles():
    n_households = 1_000_000
    X, Y = 10, 15
    K = 8
    std_betas = 0.5
    TOL_CS = 1e-1

    (
        mde_discrepancy,
        mde_discrepancy_numeric,
        mde_discrepancy_corrected,
        mde_discrepancy_corrected_numeric,
        poisson_discrepancy,
    ) = demo_choo_siow_no_singles(n_households, X, Y, K, std_betas=std_betas)
    assert mde_discrepancy < TOL_CS
    assert mde_discrepancy_numeric < TOL_CS
    assert mde_discrepancy_corrected < TOL_CS
    assert mde_discrepancy_corrected_numeric < TOL_CS
    assert poisson_discrepancy < TOL_CS


if __name__ == "__main__":
    test_choo_siow_no_singles()

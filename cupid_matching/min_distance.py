""" Estimates semilinear separable models with a given entropy function.
The entropy function and the surplus matrix must both be linear in the parameters.
"""

from typing import cast

import numpy as np
import scipy.stats as sts
from bs_python_utils.bsutils import print_stars

from cupid_matching.entropy import (
    EntropyFunctions,
    EntropyHessianMuMu,
    EntropyHessianMuR,
    EntropyHessians,
    numeric_hessian,
)
from cupid_matching.matching_utils import Matching, MatchingFunction
from cupid_matching.min_distance_utils import (
    MDEResults,
    check_args_mde,
    check_indep_phi_no_singles,
    compute_estimates,
    get_initial_weighting_matrix,
    get_optimal_weighting_matrix,
    make_D2_matrix,
    make_hessian_mde,
)
from cupid_matching.utils import make_XY_K_mat


def estimate_semilinear_mde(
    muhat: Matching,
    phi_bases: np.ndarray,
    entropy: EntropyFunctions,
    no_singles: bool = False,
    additional_parameters: list | None = None,
    initial_weighting_matrix: np.ndarray | None = None,
    verbose: bool = False,
) -> MDEResults:
    """
    Estimates the parameters of the distributions and of the base functions.

    Args:
        muhat: the observed `Matching`
        phi_bases: an (X, Y, K) array of bases
        entropy: an `EntropyFunctions` object
        no_singles: if `True`, only couples are observed
        additional_parameters: additional parameters of the distribution of errors,
            if any
        initial_weighting_matrix: if specified, used as the weighting matrix
            for the first step when `entropy.param_dependent` is `True`
        verbose: prints stuff if `True`

    Returns:
        an `MDEResults` instance

    Example:
        ```py
        # We simulate a Choo and Siow homoskedastic marriage market
        #  and we estimate a gender-heteroskedastic model on the simulated data.
        X, Y, K = 10, 20, 2
        n_households = int(1e6)
        lambda_true = np.random.randn(K)
        phi_bases = np.random.randn(X, Y, K)
        n = np.ones(X)
        m = np.ones(Y)
        Phi = phi_bases @ lambda_true
        choo_siow_instance = ChooSiowPrimitives(Phi, n, m)
        mus_sim = choo_siow_instance.simulate(n_households)
        choo_siow_instance.describe()

        entropy_model =  entropy_choo_siow_gender_heteroskedastic_numeric
        n_alpha = 1
        true_alpha = np.ones(n_alpha)
        true_coeffs = np.concatenate((true_alpha, lambda_true))

        print_stars(entropy_model.description)

        mde_results = estimate_semilinear_mde(
            mus_sim, phi_bases, entropy_model)

        mde_results.print_results(true_coeffs=true_coeffs, n_alpha=1)
        ```

    """
    X, Y, K = check_args_mde(muhat, phi_bases)
    XY = X * Y
    parameterized_entropy = entropy.parameter_dependent
    S_mat = get_initial_weighting_matrix(
        parameterized_entropy, initial_weighting_matrix, XY
    )

    phi_mat = make_XY_K_mat(phi_bases)

    # if there are no singles, we need to premultiply by the randomized double differencing matrix $D_2$
    if no_singles:
        D2_mat, rank_D2 = make_D2_matrix(X, Y)
        phi_mat = D2_mat @ phi_mat
        check_indep_phi_no_singles(phi_mat, X, Y)

    e0_vals = entropy.e0_fun(muhat, additional_parameters)
    e0_hat = e0_vals.ravel()

    # if there are no singles, we need to premultiply by the randomized double differencing matrix $D_2$
    if no_singles:
        e0_hat = D2_mat @ e0_hat

    if not parameterized_entropy:  # we only have e0(mu,r)
        n_pars = K
        hessian = entropy.hessian
        if hessian == "provided":  # we have the analytical hessian
            e0_derivative = cast(EntropyHessians, entropy.e0_derivative)
            hessian_components_mumu = e0_derivative[0](muhat, additional_parameters)
            hessian_components_mur = e0_derivative[1](muhat, additional_parameters)
        else:  # we use a numerical hessian
            hessian_components_mumu, hessian_components_mur = numeric_hessian(
                entropy,
                muhat,
                additional_parameters=additional_parameters,
            )

        hessians_both = make_hessian_mde(
            hessian_components_mumu, hessian_components_mur
        )

        # if there are no singles, we need to premultiply by the randomized double differencing matrix $D_2$
        if no_singles:
            S_mat = get_optimal_weighting_matrix(
                muhat, hessians_both, no_singles, D2_mat
            )
        else:
            S_mat = get_optimal_weighting_matrix(muhat, hessians_both)

        estimated_coefficients, varcov_coefficients = compute_estimates(
            phi_mat, S_mat, e0_hat
        )
        stderrs_coefficients = np.sqrt(np.diag(varcov_coefficients))
        est_Phi = phi_mat @ estimated_coefficients
        residuals = est_Phi + e0_hat
    else:  # parameterized entropy: e0(mu,r) + e(mu,r) . alpha
        e_fun = cast(MatchingFunction, entropy.e_fun)
        e_vals = e_fun(muhat, additional_parameters)
        e_hat = make_XY_K_mat(e_vals)

        # if there are no singles, we need to premultiply by the randomized double differencing matrix $D_2$
        if no_singles:
            e0_hat = D2_mat @ e0_hat

        F_hat = np.column_stack((e_hat, phi_mat))
        n_pars = e_hat.shape[1] + K

        # first pass with an initial weighting matrix
        first_coeffs, _ = compute_estimates(F_hat, cast(np.ndarray, S_mat), e0_hat)
        first_alpha = first_coeffs[:-K]

        if verbose:
            print_stars("First-stage estimates:")
            print(first_coeffs)

        # compute the efficient weighting matrix
        hessian = entropy.hessian
        if hessian == "provided":  # we have the analytical hessian
            e0_derivative = cast(EntropyHessians, entropy.e0_derivative)
            e_derivative = cast(EntropyHessians, entropy.e_derivative)
            e0_derivative_mumu = cast(EntropyHessianMuMu, e0_derivative[0])
            e0_derivative_mur = cast(EntropyHessianMuR, e0_derivative[1])
            e_derivative_mumu = cast(EntropyHessianMuMu, e_derivative[0])
            e_derivative_mur = cast(EntropyHessianMuR, e_derivative[1])
            hessian_components_mumu_e0 = e0_derivative_mumu(
                muhat, additional_parameters
            )
            hessian_components_mur_e0 = e0_derivative_mur(muhat, additional_parameters)
            hessian_components_mumu_e = e_derivative_mumu(muhat, additional_parameters)
            hessian_components_mur_e = e_derivative_mur(muhat, additional_parameters)

            if verbose:
                print_stars("First-stage estimates:")
                print(first_coeffs)

            hessian_components_mumu = (
                hessian_components_mumu_e0[i]
                + hessian_components_mumu_e[i] @ first_alpha
                for i in range(3)
            )
            hessian_components_mur = (
                hessian_components_mur_e0[i] + hessian_components_mur_e[i] @ first_alpha
                for i in range(3)
            )
        else:  # we use a numeric hessian
            hessian_components_mumu, hessian_components_mur = numeric_hessian(
                entropy,
                muhat,
                alpha=first_alpha,
                additional_parameters=additional_parameters,
            )
        hessians_both = make_hessian_mde(
            hessian_components_mumu, hessian_components_mur
        )

        # if there are no singles, we need to premultiply by the randomized double differencing matrix $D_2$
        if no_singles:
            S_mat = get_optimal_weighting_matrix(
                muhat, hessians_both, no_singles, D2_mat
            )
        else:
            S_mat = get_optimal_weighting_matrix(muhat, hessians_both)

        # second pass with the efficient weighting matrix
        estimated_coefficients, varcov_coefficients = compute_estimates(
            F_hat, S_mat, e0_hat
        )
        est_alpha, est_beta = (
            estimated_coefficients[:-K],
            estimated_coefficients[-K:],
        )
        stderrs_coefficients = np.sqrt(np.diag(varcov_coefficients))
        est_Phi = phi_mat @ est_beta
        residuals = est_Phi + e0_hat + e_hat @ est_alpha

    value_obj = residuals.T @ S_mat @ residuals
    ndf = rank_D2 - n_pars if no_singles else XY - n_pars
    test_stat = value_obj
    muxyhat, *_, nhat, mhat = muhat.unpack()
    n_individuals = np.sum(nhat) + np.sum(mhat)
    n_households = n_individuals - np.sum(muxyhat)

    est_Phi = est_Phi.reshape((X - 1, Y - 1)) if no_singles else est_Phi.reshape((X, Y))

    results = MDEResults(
        X=X,
        Y=Y,
        K=K,
        number_households=n_households,
        estimated_coefficients=estimated_coefficients,
        varcov_coefficients=varcov_coefficients,
        stderrs_coefficients=stderrs_coefficients,
        estimated_Phi=est_Phi,
        test_statistic=test_stat,
        ndf=ndf,
        test_pvalue=sts.chi2.sf(test_stat, ndf),
        parameterized_entropy=parameterized_entropy,
    )
    return results

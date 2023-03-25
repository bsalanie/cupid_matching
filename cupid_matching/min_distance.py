""" Estimates semilinear separable models with a given entropy function.
The entropy function and the surplus matrix must both be linear in the parameters.
"""
from typing import Optional, cast

import numpy as np
import scipy.linalg as spla
import scipy.stats as sts

from .entropy import (EntropyFunctions, EntropyHessianMuMu,
                      EntropyHessianMuMuParam, EntropyHessianMuR,
                      EntropyHessianMuRParam, EntropyHessians,
                      EntropyHessiansParam, _fill_hessianMuMu_from_components,
                      _fill_hessianMuR_from_components, _numeric_hessian)
from .matching_utils import (Matching, MatchingFunction, MatchingFunctionParam,
                             _make_XY_K_mat, _variance_muhat)
from .min_distance_utils import MDEResults, _compute_estimates
from .utils import bs_error_abort, print_stars


def estimate_semilinear_mde(
    muhat: Matching,
    phi_bases: np.ndarray,
    entropy: EntropyFunctions,
    additional_parameters: Optional[list] = None,
    initial_weights: Optional[np.ndarray] = None,
) -> MDEResults:
    """
    Estimates the parameters of the distributions and of the base functions.

    Args:
        muhat: the observed Matching
        phi_bases: an (X, Y, K) array of bases
        entropy: an `EntropyFunctions` object
        additional_parameters: additional parameters of the distribution of errors,
            if any
        initial_weights: if specified, used as the weighting matrix
            for the first step when `entropy.param_dependent` is `True`

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
    muxyhat, _, _, nhat, mhat = muhat.unpack()
    X, Y = muxyhat.shape
    XY = X * Y
    ndims_phi = phi_bases.ndim
    if ndims_phi != 3:
        bs_error_abort(f"phi_bases should have 3 dimensions, not {ndims_phi}")
    Xp, Yp, K = phi_bases.shape
    if Xp != X or Yp != Y:
        bs_error_abort(
            f"phi_bases should have shape ({X}, {Y}, {K}) not ({Xp}, {Yp}, {K})"
        )
    parameterized_entropy = entropy.parameter_dependent
    if parameterized_entropy:
        if initial_weights is None:
            print_stars(
                "Using the identity matrix as weighting matrix in the first step."
            )
            S_mat = np.eye(XY)
        else:
            S_mat = initial_weights

    phi_mat = _make_XY_K_mat(phi_bases)
    e0_fun = entropy.e0_fun
    if additional_parameters is None:
        e0_fun = cast(MatchingFunction, e0_fun)
        e0_vals = e0_fun(muhat)
    else:
        e0_fun = cast(MatchingFunctionParam, e0_fun)
        e0_vals = e0_fun(muhat, additional_parameters)
    e0_hat = e0_vals.ravel()

    if not parameterized_entropy:  # we only have e0(mu,r)
        n_pars = K
        hessian = entropy.hessian
        if hessian == "provided":
            e0_derivative = cast(EntropyHessians, entropy.e0_derivative)
            if additional_parameters is None:
                hessian_components_mumu = e0_derivative[0](muhat)
                hessian_components_mur = e0_derivative[1](muhat)
            else:
                e0_derivative1 = cast(EntropyHessiansParam, entropy.e0_derivative)
                hessian_components_mumu = e0_derivative1[0](
                    muhat, additional_parameters
                )
                hessian_components_mur = e0_derivative1[1](muhat, additional_parameters)
        else:
            if additional_parameters is None:
                hessian_components = _numeric_hessian(entropy, muhat)
            else:
                hessian_components = _numeric_hessian(
                    entropy,
                    muhat,
                    additional_parameters=additional_parameters,
                )
            (
                hessian_components_mumu,
                hessian_components_mur,
            ) = hessian_components
        hessian_mumu = _fill_hessianMuMu_from_components(hessian_components_mumu)
        hessian_mur = _fill_hessianMuR_from_components(hessian_components_mur)
        hessians_both = np.concatenate((hessian_mumu, hessian_mur), axis=1)

        _, var_munm = _variance_muhat(muhat)
        var_entropy_gradient = hessians_both @ var_munm @ hessians_both.T
        S_mat = spla.inv(var_entropy_gradient)
        estimated_coefficients, varcov_coefficients = _compute_estimates(
            phi_mat, S_mat, e0_hat
        )
        stderrs_coefficients = np.sqrt(np.diag(varcov_coefficients))
        est_Phi = phi_mat @ estimated_coefficients
        residuals = est_Phi + e0_hat
    else:  # parameterized entropy: e0(mu,r) + e(mu,r) . alpha
        # create the F matrix
        if additional_parameters is None:
            e_fun = cast(MatchingFunction, entropy.e_fun)
            e_vals = e_fun(muhat)
        else:
            e_fun1 = cast(MatchingFunctionParam, entropy.e_fun)
            e_vals = e_fun1(muhat, additional_parameters)
        e_hat = _make_XY_K_mat(e_vals)

        F_hat = np.column_stack((e_hat, phi_mat))
        n_pars = e_hat.shape[1] + K
        # first pass with an initial weighting matrix
        first_coeffs, _ = _compute_estimates(F_hat, S_mat, e0_hat)
        first_alpha = first_coeffs[:-K]

        # compute the efficient weighting matrix
        hessian = entropy.hessian
        if hessian == "provided":
            if additional_parameters is None:
                e0_derivative = cast(EntropyHessians, entropy.e0_derivative)
                e_derivative = cast(EntropyHessians, entropy.e_derivative)
                e0_derivative_mumu = cast(EntropyHessianMuMu, e0_derivative[0])
                hessian_components_mumu_e0 = e0_derivative_mumu(muhat)
                e0_derivative_mur = cast(EntropyHessianMuR, e0_derivative[1])
                hessian_components_mur_e0 = e0_derivative_mur(muhat)
                e_derivative_mumu = cast(EntropyHessianMuMu, e_derivative[0])
                hessian_components_mumu_e = e_derivative_mumu(muhat)
                e_derivative_mur = cast(EntropyHessianMuR, e_derivative[1])
                hessian_components_mur_e = e_derivative_mur(muhat)
            else:
                e0_derivative1 = cast(EntropyHessiansParam, entropy.e0_derivative)
                e_derivative1 = cast(EntropyHessiansParam, entropy.e_derivative)
                e0_derivative_mumu1 = cast(EntropyHessianMuMuParam, e0_derivative1[0])
                e0_derivative_mur1 = cast(EntropyHessianMuRParam, e0_derivative1[1])
                e_derivative_mumu1 = cast(EntropyHessianMuMuParam, e_derivative1[0])
                e_derivative_mur1 = cast(EntropyHessianMuRParam, e_derivative1[1])
                hessian_components_mumu_e0 = e0_derivative_mumu1(
                    muhat, additional_parameters
                )
                hessian_components_mur_e0 = e0_derivative_mur1(
                    muhat, additional_parameters
                )
                hessian_components_mumu_e = e_derivative_mumu1(
                    muhat, additional_parameters
                )
                hessian_components_mur_e = e_derivative_mur1(
                    muhat, additional_parameters
                )

            # print_stars("First-stage estimates:")
            # print(first_coeffs)

            hessian_components_mumu1 = (
                hessian_components_mumu_e0[0]
                + hessian_components_mumu_e[0] @ first_alpha,
                hessian_components_mumu_e0[1]
                + hessian_components_mumu_e[1] @ first_alpha,
                hessian_components_mumu_e0[2]
                + hessian_components_mumu_e[2] @ first_alpha,
            )
            hessian_components_mur1 = (
                hessian_components_mur_e0[0]
                + hessian_components_mur_e[0] @ first_alpha,
                hessian_components_mur_e0[1]
                + hessian_components_mur_e[1] @ first_alpha,
            )
            hessian_mumu = _fill_hessianMuMu_from_components(hessian_components_mumu1)
            hessian_mur = _fill_hessianMuR_from_components(hessian_components_mur1)
        else:  # numeric hessian
            if additional_parameters is None:
                hessian_components = _numeric_hessian(entropy, muhat, alpha=first_alpha)
            else:
                hessian_components = _numeric_hessian(
                    entropy,
                    muhat,
                    alpha=first_alpha,
                    additional_parameters=additional_parameters,
                )
            (
                hessian_components_mumu,
                hessian_components_mur,
            ) = hessian_components
            hessian_mumu = _fill_hessianMuMu_from_components(hessian_components_mumu)
            hessian_mur = _fill_hessianMuR_from_components(hessian_components_mur)

        hessians_both = np.concatenate((hessian_mumu, hessian_mur), axis=1)

        _, var_munm = _variance_muhat(muhat)
        var_entropy_gradient = hessians_both @ var_munm @ hessians_both.T
        S_mat = spla.inv(var_entropy_gradient)

        # second pass
        estimated_coefficients, varcov_coefficients = _compute_estimates(
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
    ndf = X * Y - n_pars
    test_stat = value_obj
    n_individuals = np.sum(nhat) + np.sum(mhat)
    n_households = n_individuals - np.sum(muxyhat)

    results = MDEResults(
        X=X,
        Y=Y,
        K=K,
        number_households=n_households,
        estimated_coefficients=estimated_coefficients,
        varcov_coefficients=varcov_coefficients,
        stderrs_coefficients=stderrs_coefficients,
        estimated_Phi=est_Phi.reshape((X, Y)),
        test_statistic=test_stat,
        ndf=ndf,
        test_pvalue=sts.chi2.sf(test_stat, ndf),
        parameterized_entropy=parameterized_entropy,
    )
    return results

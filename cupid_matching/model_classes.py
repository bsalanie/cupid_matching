from dataclasses import dataclass
from typing import cast

import numpy as np
from bs_python_utils.bsnputils import (
    ThreeArrays,
    check_matrix,
    check_vector,
    npexp,
    npmaxabs,
    nppow,
)
from bs_python_utils.bsutils import bs_error_abort, print_stars

from cupid_matching.ipfp_solvers import (
    IPFPNoGradientResults,
    ipfp_homoskedastic_no_singles_solver,
    ipfp_homoskedastic_solver,
)
from cupid_matching.matching_utils import (
    Matching,
    compute_margins,
    simulate_sample_from_mus,
)
from cupid_matching.utils import Nest, NestsList, change_indices, find_nest_of


@dataclass
class ChooSiowPrimitives:
    Phi: np.ndarray
    n: np.ndarray
    m: np.ndarray
    mus: Matching | None = None

    def __post_init__(self):
        X, Y = check_matrix(self.Phi)
        Xn = check_vector(self.n)
        Ym = check_vector(self.m)
        if Xn != X:
            bs_error_abort(f"Phi is a ({X}, {Y}) matrix but n has {Xn} elements.")
        if Ym != Y:
            bs_error_abort(f"Phi is a ({X}, {Y}) matrix but m has {Ym} elements.")

    def ipfp_solve(self) -> Matching:
        mus, *_ = cast(
            IPFPNoGradientResults,
            ipfp_homoskedastic_solver(self.Phi, self.n, self.m),
        )
        muxy, mux0, mu0y, _, _ = mus.unpack()
        n, m = compute_margins(muxy, mux0, mu0y)
        return Matching(muxy, n, m)

    def simulate(self, n_households: int, seed: int | None = None) -> Matching:
        self.n_households = n_households
        self.mus = self.ipfp_solve()
        mus_sim = simulate_sample_from_mus(self.mus, n_households, seed=seed)
        return mus_sim

    def describe(self):
        X, Y = self.Phi.shape
        print_stars("We are working with a Choo and Siow homoskedastic market")
        print(f"\t\t we have {int(X)} types of men and {int(Y)} types of women")
        print(f"\t\t and a total of {int(self.n_households):,d} households")


@dataclass
class ChooSiowPrimitivesNoSingles:
    Phi: np.ndarray
    n: np.ndarray
    m: np.ndarray
    mus: Matching | None = None

    def __post_init__(self):
        X, Y = check_matrix(self.Phi)
        Xn = check_vector(self.n)
        Ym = check_vector(self.m)
        if Xn != X:
            bs_error_abort(f"Phi is a ({X}, {Y}) matrix but n has {Xn} elements.")
        if Ym != Y:
            bs_error_abort(f"Phi is a ({X}, {Y}) matrix but m has {Ym} elements.")

    def ipfp_solve(self) -> Matching:
        muxy, *_ = cast(
            ThreeArrays,
            ipfp_homoskedastic_no_singles_solver(self.Phi, self.n, self.m),
        )
        return Matching(muxy, self.n, self.m, no_singles=True)

    def simulate(self, n_households: int, seed: int | None = None) -> Matching:
        self.n_households = n_households
        self.mus = self.ipfp_solve()
        mus_sim = simulate_sample_from_mus(
            self.mus, n_households, no_singles=True, seed=seed
        )
        return mus_sim

    def describe(self):
        X, Y = self.Phi.shape
        print_stars(
            "We are working with a Choo and Siow homoskedastic market w/o singles;"
        )
        print(f"\t\t we have {int(X)} types of men and {int(Y)} types of women")
        print(f"\t\t and a total of {int(self.n_households):,d} households")


@dataclass
class NestedLogitPrimitives:
    Phi: np.ndarray
    n: np.ndarray
    m: np.ndarray
    nests_for_each_x: (
        NestsList  # given by user, e.g. [[1, 3], [2,4]] has y=1 and y=3 in first nest
    )
    nests_for_each_y: NestsList
    nests_over_Y: (
        NestsList  # rebased to zero: the above example becomes [[0, 2], [1,3]]
    )
    nests_over_X: NestsList
    i_nest_of_x: Nest  # mapping x -> n'
    i_nest_of_y: Nest  # mapping y -> n
    n_alphas: int
    mus: Matching | None = None
    true_alphas: np.ndarray | None = None

    def __init__(
        self,
        Phi: np.ndarray,
        n: np.ndarray,
        m: np.ndarray,
        nests_for_each_x: NestsList,
        nests_for_each_y: NestsList,
        true_alphas: np.ndarray | None = None,
    ):
        """
        We only model two-level nested logit, with {0} as the first nest,
        and nests and nests parameters that do not depend on the type.

        Args:
            Phi: the (X,Y) joint surplus matrix
            n: the X-vector of men margins
            m: the X-vector of women margins
            nests_for_each_x: the composition of the nests over 1...Y, a list of r lists
            nests_for_each_y: the composition of the nests over 1...X, a list of d lists
            true_alphas: the true nest parameters, if any; should be an (r+d)-vector
        """
        X, Y = check_matrix(Phi)
        Xn = check_vector(n)
        Ym = check_vector(m)

        # we need to rebase the indices to zero
        self.nests_over_X = change_indices(nests_for_each_y)
        self.nests_over_Y = change_indices(nests_for_each_x)

        self.n_alphas = len(nests_for_each_y) + len(nests_for_each_x)

        if Xn != X:
            bs_error_abort(f"Phi is a ({X}, {Y}) matrix but n has {Xn} elements.")
        if Ym != Y:
            bs_error_abort(f"Phi is a ({X}, {Y}) matrix but m has {Ym} elements.")

        if true_alphas is not None:
            alpha_size = check_vector(true_alphas)
            if alpha_size != self.n_alphas:
                bs_error_abort(
                    f"true_alphas shoud have {self.n_alphas} elements, not {alpha_size}"
                )

        self.Phi = Phi
        self.n = n
        self.m = m
        self.true_alphas = true_alphas
        self.nests_for_each_x = nests_for_each_x
        self.nests_for_each_y = nests_for_each_y

        # check that every x is in a nest, and just once
        nests_check = []
        i_nest_of_x = np.zeros(X, int)
        for x in range(X):
            i_nest_of_x[x] = find_nest_of(self.nests_over_X, x)
            nests_check.append(i_nest_of_x[x])
        if -1 in nests_check or len(set(nests_check)) != len(nests_for_each_y):
            bs_error_abort("Check your nests_for_each_y")
        # check that every y is in a nest, and just once
        nests_check = []
        i_nest_of_y = np.zeros(Y, int)
        for y in range(Y):
            i_nest_of_y[y] = find_nest_of(self.nests_over_Y, y)
            nests_check.append(i_nest_of_y[y])
        if -1 in nests_check or len(set(nests_check)) != len(nests_for_each_x):
            bs_error_abort("Check your nests_for_each_x")

        self.i_nest_of_x = i_nest_of_x.tolist()
        self.i_nest_of_y = i_nest_of_y.tolist()

    def __str__(self):
        X, Y = self.Phi.shape
        nmen, nwomen = np.sum(self.n), np.sum(self.m)
        repr_str = (
            f"This is a 2-level nested logit with {nmen} men of {X} types"
            + f" and {nwomen} women of {Y} types.\n"
        )
        repr_str += (
            f" We have {self.n_nests_over_Y} nests over 1...Y "
            + f" and {self.n_nests_over_X} nests over 1...X,\n"
        )
        if self.true_alphas is None:
            repr_str += "     with unspecified nests parameters."
        else:
            alpha_vals = self.true_alphas
            repr_str += "     with respective nests parameters:\n"
            repr_str += f"    {alpha_vals[:self.n_nests_over_Y]}\n"
            repr_str += f" and {alpha_vals[self.n_nests_over_Y:]}\n"
        print_stars(repr_str)

    def ipfp_nested_logit_solver(
        self, tol: float = 1e-9, verbose: bool = False, maxiter: int = 1000
    ) -> tuple[Matching, np.ndarray, np.ndarray]:
        """Solves for equilibrium in a two-level nested logit market
        given systematic surplus and margins and nests parameters;
        does not compute_ the gradient of the matching patterns

        Args:
            tol: tolerance on change in solution
            verbose: if `True`, prints information
            maxiter: maximum number of iterations

        Returns:
             the matching patterns
             marg_err_x, marg_err_y: the errors on the margins
        """
        alphas = self.true_alphas
        if alphas is None:
            bs_error_abort("cannot solve without nest parameters")
        else:
            alphas = cast(np.ndarray, alphas)
            n_rhos = len(self.nests_over_Y)
            n_deltas = len(self.nests_over_X)
            rhos = alphas[:n_rhos]
            deltas = alphas[n_rhos:]

        #############################################################################
        # we solve the equilibrium equations
        #   starting with a reasonable initial point  muxy, mux0, mu0y = bigc
        #   it is important that it fit the number of individuals
        #############################################################################

        n, m = self.n, self.m
        X, Y = n.size, m.size

        nests_over_X, nests_over_Y = self.nests_over_X, self.nests_over_Y
        i_nest_of_x, i_nest_of_y = self.i_nest_of_x, self.i_nest_of_y

        rho_vals = rhos[i_nest_of_y]  # rho(n) for y in n in the paper
        delta_vals = deltas[i_nest_of_x]  # delta(n') for x in n' in the paper

        ephi = npexp(self.Phi / np.add.outer(delta_vals, rho_vals))

        # initial values
        nindivs = np.sum(n) + np.sum(m)
        bigc = nindivs / (X + Y + 2.0 * np.sum(ephi))

        mux0, mu0y, muxy = (
            np.full(X, bigc),
            np.full(Y, bigc),
            np.full((X, Y), bigc),
        )
        muxn = np.zeros((X, n_rhos))
        for i_nest_y, nest_y in enumerate(nests_over_Y):
            muxn[:, i_nest_y] = np.sum(muxy[:, nest_y], 1)
        muny = np.zeros((n_deltas, Y))
        for i_nest_x, nest_x in enumerate(nests_over_X):
            muny[i_nest_x, :] = np.sum(muxy[nest_x, :], 0)

        err_diff = bigc
        tol_diff = tol * bigc
        tol_newton = tol
        max_newton = 2000
        MIN_REST = 1e-4 * bigc  # used to bound mus below in the Newton iterations

        niter = 0
        while (err_diff > tol_diff) and (niter < maxiter):  # IPFP main loop
            # Newton iterates for men
            err_newton = bigc
            i_newton = 0
            while err_newton > tol_newton:
                gbar = np.zeros(
                    (X, n_rhos)
                )  # this will be the $\bar{G}^x_n$ of the note
                gbar_pow = np.zeros((X, n_rhos))
                biga = np.zeros(X)  # this will be the $A_x$ of the note
                for i_nest_x, nest_x in enumerate(nests_over_X):
                    # i_nest_x is n' in the paper
                    delta_x = deltas[i_nest_x]
                    muny_x = muny[i_nest_x, :]  # mu(n', :)
                    for x in nest_x:
                        ephi_x = ephi[x, :]
                        for i_nest_y, nest_y in enumerate(nests_over_Y):
                            # i_nest_y is n in the paper
                            mu_n = muny_x[nest_y]
                            mu0_n = mu0y[nest_y]
                            evec_n = ephi_x[nest_y]
                            rho_n = rhos[i_nest_y]
                            sum_rd = rho_n + delta_x
                            mun_term = nppow(mu_n, (delta_x - 1.0) / sum_rd)
                            mu0_term = nppow(mu0_n, 1.0 / sum_rd)
                            gbar[x, i_nest_y] = np.sum(mun_term * mu0_term * evec_n)
                            gbar_pow[x, i_nest_y] = nppow(
                                gbar[x, i_nest_y], sum_rd / (delta_x + 1.0)
                            )
                            biga[x] += gbar_pow[x, i_nest_y]

                # now we take one Newton step for all types of men
                delta_vals1 = 1.0 + delta_vals
                mux0_term = nppow(mux0, 1.0 / delta_vals1)
                bigb = mux0_term * biga  # this is the $B_x$ of the note
                numer = n * delta_vals1 - delta_vals * bigb
                lower_bound = np.full(X, MIN_REST)
                mux0_new = mux0 * np.maximum(
                    numer / (delta_vals1 * mux0 + bigb), lower_bound
                )
                muxn_new = gbar_pow * mux0_term.reshape((-1, 1))

                mux0 = mux0_new
                muxn = muxn_new
                errxi = mux0 + np.sum(muxn, 1) - n
                err_newton = npmaxabs(errxi)
                i_newton += 1
                if i_newton > max_newton:
                    bs_error_abort(
                        f"Newton solver failed for men after {max_newton} iterations"
                    )

            if verbose:
                print(
                    f"Newton error on men is {err_newton} after {i_newton} iterations"
                )

            # Newton iterates for women
            err_newton = bigc
            i_newton = 0
            while err_newton > tol_newton:
                gbar = np.zeros((Y, n_deltas))
                gbar_pow = np.zeros((Y, n_deltas))
                biga = np.zeros(Y)
                for i_nest_y, nest_y in enumerate(nests_over_Y):
                    # i_nest_y is n in the paper
                    rho_y = rhos[i_nest_y]
                    muxn_y = muxn[:, i_nest_y]  # mu(:, n)
                    for y in nest_y:
                        ephi_y = ephi[:, y]
                        for i_nest_x, nest_x in enumerate(nests_over_X):
                            mu_n = muxn_y[nest_x]
                            mu0_n = mux0[nest_x]
                            evec_n = ephi_y[nest_x]
                            delta_n = deltas[i_nest_x]
                            sum_rd = rho_y + delta_n
                            mun_term = nppow(mu_n, (rho_n - 1.0) / sum_rd)
                            mu0_term = nppow(mu0_n, 1.0 / sum_rd)
                            gbar[y, i_nest_x] = np.sum(mun_term * mu0_term * evec_n)
                            gbar_pow[y, i_nest_x] = nppow(
                                gbar[y, i_nest_x], sum_rd / (1.0 + rho_y)
                            )
                            biga[y] += gbar_pow[y, i_nest_x]

                # now we take one Newton step for all types of women
                rho_vals1 = 1.0 + rho_vals
                mu0y_term = nppow(mu0y, 1.0 / rho_vals1)
                bigb = mu0y_term * biga
                numer = m * rho_vals1 - rho_vals * bigb
                lower_bound = np.full(Y, MIN_REST)
                mu0y_new = mu0y * np.maximum(
                    numer / (rho_vals1 * mu0y + bigb), lower_bound
                )
                muny_new = gbar_pow.T * mu0y_term

                mu0y = mu0y_new
                muny = muny_new
                erryi = mu0y + np.sum(muny, 0) - m
                err_newton = npmaxabs(erryi)
                i_newton += 1
                if i_newton > max_newton:
                    bs_error_abort(
                        f"Newton solver failed for women after {max_newton} iterations"
                    )

            if verbose:
                print(
                    f"Newton error on women is {err_newton} after {i_newton} iterations"
                )

            muxy = np.zeros((X, Y))
            for x in range(X):
                i_nest_x = i_nest_of_x[x]  # n'
                ephi_x = ephi[x, :]
                mux0_x = mux0[x]
                muxn_x = muxn[x, :]
                delta_x = delta_vals[x]
                muny_x = muny[i_nest_x, :]
                for y in range(Y):
                    i_nest_y = i_nest_of_y[y]  # n
                    mu0y_y = mu0y[y]
                    rho_y = rho_vals[y]
                    muxn_xy = muxn_x[i_nest_y]
                    muny_xy = muny_x[y]
                    mu_term = (
                        mux0_x
                        * mu0y_y
                        * (muxn_xy ** (rho_y - 1.0))
                        * (muny_xy ** (delta_x - 1.0))
                    )
                    muxy[x, y] = ephi_x[y] * (mu_term ** (1.0 / (delta_x + rho_y)))

            n_sim, m_sim = compute_margins(muxy, mux0, mu0y)
            marg_err_x, marg_err_y = n_sim - n, m_sim - m

            if verbose:
                print(
                    f"Margin error on men is {marg_err_x} "
                    f" after {niter} IPFP iterations"
                )
                print(
                    f"Margin error on women is {marg_err_y} "
                    f" after {niter} IPFP iterations"
                )
            err_diff = npmaxabs(marg_err_x) + npmaxabs(marg_err_y)
            niter += 1

        n_sim, m_sim = compute_margins(muxy, mux0, mu0y)
        marg_err_x = n_sim - n
        marg_err_y = m_sim - m

        if verbose:
            print(
                f"Margin error on men is {npmaxabs(marg_err_x)} after {niter} IPFP"
                " iterations"
            )
            print(
                f"Margin error on women is {npmaxabs(marg_err_y)} after {niter} IPFP"
                " iterations"
            )

        return Matching(muxy, n, m), marg_err_x, marg_err_y

    def ipfp_solve(self) -> Matching:
        if self.true_alphas is None:
            bs_error_abort(
                "true_alphas must be specified to solve the nested logit by IPFP."
            )
        self.mus, err_x, err_y = self.ipfp_nested_logit_solver(verbose=False)
        return self.mus

    def simulate(self, n_households: int, seed: int | None = None) -> Matching:
        self.mus = self.ipfp_solve()
        mus_sim = simulate_sample_from_mus(self.mus, n_households, seed=seed)
        return mus_sim

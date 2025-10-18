# cupid_matching

![GitHub last commit](https://img.shields.io/github/last-commit/bsalanie/cupid_matching)
[![Build status](https://img.shields.io/github/actions/workflow/status/bsalanie/cupid_matching/main.yml?branch=main)](https://github.com/bsalanie/cupid_matching/actions/workflows/main.yml?query=branch%3Amain)
<!-- [![Release](https://img.shields.io/github/v/release/bsalanie/cupid_matching)](https://img.shields.io/github/v/release/bsalanie/cupid_matching) -->

<!-- https://img.shields.io/pypi/dm/cupid_matching -->

<!-- [![codecov](https://codecov.io/gh/bsalanie/cupid_matching/branch/main/graph/badge.svg)](https://codecov.io/gh/bsalanie/cupid_matching) -->

<!-- [![Commit activity](https://img.shields.io/github/commit-activity/m/bsalanie/cupid_matching)](https://img.shields.io/github/commit-activity/m/bsalanie/cupid_matching) -->

<!-- [![License](https://img.shields.io/github/license/bsalanie/cupid_matching)](https://img.shields.io/github/license/bsalanie/cupid_matching) -->

**A Python package to solve, simulate and estimate separable matching models**

-   Free software: MIT license
-   Documentation: <https://bsalanie.github.io/cupid_matching>
-   See also: [An interactive Streamlit app](http://3.84.215.135:8501)

## Installation

```         
pip install [-U] cupid_matching
```

The package relies on utilities from `bs_python_utils`; installing with `pip` pulls it in automatically, but you will need it available locally when running the test suite.

## Importing functions from the package

For instance:

``` py
from cupid_matching.min_distance import estimate_semilinear_mde
```

## How it works

The following only describes the general ideas. See   [here](https://github.com/bsalanie/cupid_matching/blob/main/CupidMatchingDoc.pdf) for more background and the API reference on this site for the technical documentation.

The `cupid_matching` package has code 

- to solve for the stable matching using our Iterative Projection Fitting Procedure (IPFP) in variants of the model of bipartite, one-to-one matching with perfectly transferable utility. It has IPFP solvers for variants of the [Choo and Siow 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) model with or without singles, homoskedastic and heteroskedastic; and also for a class of nested logit models
- to estimate the parameters of separable models with semilinear surplus and entropy using a minimum distance estimator
- to estimate the parameters of semilinear Choo and Siow models using a Poisson GLM estimator
- for a [Streamlit](https://www.streamlit.io/) interactive app that demonstrates solving and estimating the Choo and Siow model using the `cupid_matching` package. You can try it [here](http://3.84.215.135:8501).

Incidentally, my [ipfp_R](https://www.github.com/bsalanie/ipfp_R.git) Github repository contains R code to solve for equilibrium in (*only*) the basic version of the Choo and Siow model.

The package builds on the pioneering work of [Choo and Siow *JPE* 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) and on my work with Alfred Galichon, especially our [*REStud* 2022 paper](https://academic.oup.com/restud/article-abstract/89/5/2600/6478301) and [this working paper](https://econ.columbia.edu/working-paper/estimating-separable-matching-models/).

At this stage, it only deals with bipartite models. As the heterosexual marriage market is a leading example, I will refer to the two sides as *men* and *women*. Each man $m$ (resp. each women $w$) has an observed, discrete-valued type $x$ (resp. $y$), along with unobserved heterogeneity.

### The primitives

The primitives of this class of matching models are 

- the *margins*: the numbers $n_x$ of men of type $x=1,\ldots,X$ and the numbers $m_y$ of women of type $y=1,\ldots,Y$
- the *joint surplus* created by the match of a man $m$ of type $x$ and a woman $w$ of type $y$. We assume *separability*: this joint surplus takes the form 
$$
\Phi_{xy}+\varepsilon_{my} +\eta_{xw}.
$$ 
I will only describe here the case when the data also has singles, i.e., when $x=y=0$ is a possible type. The [fuller documentation](#) explains how to deal with the case when singles are not observed.


A single man has utility $\varepsilon_{m0}$ and a single woman has utility $\eta_{0w}$.

The modeler chooses the distributions of the vectors $(\varepsilon_{m0},\varepsilon_{m1},\ldots, \varepsilon_{mY})$ and $(\eta_{0w},\eta_{1w},\ldots,\eta_{Xw})$.

### The solution: the stable matching

We denote $\mu_{xy}$ the number of matches between men of type $x$ and women of type $y$, $\mu_{x0}$ the number of single men of type $x$, and $\mu_{0y}$ the number of single women of type $y$.

The total numbers must add up to the margins: 
$$
\sum_{y=1}^Y \mu_{xy}+\mu_{x0}=n_x \; \text{ and } \;
\sum_{x=1}^X \mu_{xy}+\mu_{0y}=m_y.
$$ 
The total number of individuals is $N_i=\sum_{x=1}^X n_x+ \sum_{y=1}^Y m_y$ and the total number of households is $N_h = N_i - \sum_{x=1}^X \sum_{y=1}^Y \mu_{xy}$.

Galichon-Salanié (*REStud* 2022) shows that in large markets, if the vectors $\varepsilon$ and $\eta$ have full support, the stable matching is the unique solution to the strictly convex program $$
\max_{\mu} \left(\sum_{x=1}^X \sum_{y=1}^Y \mu_{xy}\Phi_{xy} + \mathcal{E}(\mu; n, m)\right)
$$ where $\mathcal{E}$, the *generalized entropy* function, depends on the assumed distributions of the $\varepsilon$ and $\eta$ random vectors.

The files `choo_siow.py`, `choo_siow_gender_heteroskedastic`, `choo_siow_heteroskedastic`, and `nested_logit` provide `EntropyFunctions` objects that compute the generalized entropy and at least its first derivative for, respectively,

1.  the original Choo and Siow 2006 model, in which the $\varepsilon$ and $\eta$ terms are iid draws from a type I extreme value distribution
2.  the same model, without singles (to be used when only couples are observed)
3.  an extension of 1. that allows for a scale parameter $\tau$ for the distribution of $\eta$
4.  an extension of 3. that has type-dependent scale parameters $\sigma_x$ and $\tau_y$ (with $\sigma_1=1$
5.  a two-layer nested logit model in which singles (type 0) are in their own nest and the user chooses the structure of the other nests.

Users of the package are welcome to code `EntropyFunctions` objects for different distributions of the unobserved heterogeneity terms.

### Solving for the stable matching

Given any joint surplus matrix $\Phi$; margins $n$ and $m$; and a generalized entropy $\mathcal{E}$, one would like to compute the stable matching patterns $\mu$.

For the five classes of models above, this can be done efficiently using the IPFP algorithm in Galichon-Salanié (*REStud* 2022) , which is coded in `ipfp_solvers.py` for the four Choo and Siow variants and in `model_classes.py` for the nested logit.

Here is an example, given a Numpy array $\Phi$ that is an $(X,Y)$ matrix and a number $\tau>0$:

``` py
import numpy as np

from cupid_matching.ipfp_solvers import ipfp_gender_homoskedastic solver

solution = ipfp_gender_heteroskedastic_solver(Phi, n, m, tau)
mus, error_x, error_y = solution
muxy = mus.muxy
```

The `mus` above is an instance of a `Matching` object (defined in `matching_utils.py`). `mus.muxy` has the number of couples by $(x,y)$ cell at the stable matching; the vectors `mus.mux0` and `mus.mu0y` contain the numbers of single men and women of each type.

The vectors `error_x` and `error_y` are estimates of the precision of the solution (see the code in `ipfp_solvers.py`).

### Estimating the joint surplus

Given observed matching patterns $\mu$; a class of generalized entropy functions $(\mathcal{E}^{\alpha})$; and a class of joint surplus functions $(\Phi^{\beta})$, one would like to estimate the parameter vectors $\alpha$ and $\beta$.

The package provides two estimators, which are described extensively in [this paper](https://econ.columbia.edu/working-paper/estimating-separable-matching-models/):

- the minimum distance estimator in `min_distance.py` 
- the Poisson estimator in `poisson_glm.py`, which only applies to the Choo and Siow homoskedastic model.



At this stage `cupid_matching` only allows for linear models of the joint surplus:

$$
\Phi^{\beta}_{xy} = \sum_{k=1}^K \phi_{xy}^k \beta_k
$$


where the *basis functions* $(\phi^1,\ldots,\phi^K)$ are imposed by the analyst. 

The minimum distance estimator works as follows, given 

- an observed matching stored in a `Matching` object `mus`
- an `EntropyFunction` object `entropy_model` that allows for `p` parameters in $\alpha$
- an $(X,Y,K)$ Numpy array of basis functions `phi_bases`:
  
```py
mde_results = estimate_semilinear_mde(
    mus, phi_bases, entropy_model)
mde_results.print_results(n_alpha=p)
```
The `mde_results` object contains the estimated $\alpha$ and $\beta$, their estimated variance-covariance, and a specification test.

The Poisson-GLM estimator of the Choo and Siow homoskedastic model takes as input the obserrved matching and the basis functions, and returns the estimated $\beta$ and its estimated variance-covariance.
```py
poisson_results = choo_siow_poisson_glm(mus_sim, phi_bases)
    _, mux0_sim, mu0y_sim, n_sim, m_sim = mus_sim.unpack()
poisson_results.print_results()
```
The `poisson_results` object contains the estimated  $\beta$, the expected utilities $u_x$ and $v_y$, and their estimated variance-covariance.



## Examples 
The following can be found in the `examples` folder of the package:

-   [example_choosiow.py](https://github.com/bsalanie/cupid_matching/blob/main/cupid_matching/example_choo_siow.py) shows how to run minimum distance and Poisson estimators on a Choo and Siow homoskedastic model.
- [example_choosiow_no_singles.py](https://github.com/bsalanie/cupid_matching/blob/main/cupid_matching/example_choo_siow_no_singles.py) does the same for a model without singles.
-   [example_nested_logit.py](https://github.com/bsalanie/cupid_matching/blob/main/cupid_matching/example_nested_logit.py) shows how to run minimum distance estimators on a two-layer nested logit model.

## Warnings
-   many of these models (including all variants of Choo and Siow) rely heavily on logarithms and exponentials. It is easy to generate examples where numeric instability sets in.
-   as a consequence, the `numeric` versions of the minimum distance estimator (which use numerical derivatives) are not recommended.
-   the bias-corrected minimum distance estimator (`corrected`) may have a larger mean-squared error and/or introduce numerical instabilities.
-   the estimated variance of the estimators assumes that the observed matching was sampled at the household level, and that sampling weights are all equal.

## Release notes

{{< include releases.md >}}

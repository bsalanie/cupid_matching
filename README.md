# cupid_matching

![GitHub last commit](https://img.shields.io/github/last-commit/bsalanie/cupid_matching)

<!-- [![Release](https://img.shields.io/github/v/release/bsalanie/cupid_matching)](https://img.shields.io/github/v/release/bsalanie/cupid_matching) -->

[![Build status](https://img.shields.io/github/actions/workflow/status/bsalanie/cupid_matching/main.yml?branch=main)](https://github.com/bsalanie/cupid_matching/actions/workflows/main.yml?query=branch%3Amain)

<!-- [![codecov](https://codecov.io/gh/bsalanie/cupid_matching/branch/main/graph/badge.svg)](https://codecov.io/gh/bsalanie/cupid_matching) -->

<!-- [![Commit activity](https://img.shields.io/github/commit-activity/m/bsalanie/cupid_matching)](https://img.shields.io/github/commit-activity/m/bsalanie/cupid_matching) -->

[![License](https://img.shields.io/github/license/bsalanie/cupid_matching)](https://img.shields.io/github/license/bsalanie/cupid_matching)

**A Python package to solve, simulate and estimate separable matching models**

-   Free software: MIT license
-   Documentation: <https://bsalanie.github.io/cupid_matching>
-   See also: [An interactive Streamlit app](https://share.streamlit.io/bsalanie/cupid_matching_st/main/cupid_streamlit.py)

## Installation

```         
pip install [-U] cupid_matching
```

## Importing functions from the package

For instance:

``` py
from cupid_matching.min_distance import estimate_semilinear_mde
```

## Examples

-   `example_choosiow.py` shows how to run minimum distance and Poisson estimators on a Choo and Siow homoskedastic model.
-   `example_nestedlogit.py` shows how to run minimum distance estimators on a two-layer nested logit model.

## Warnings

-   many of these models (including all variants of Choo and Siow) rely heavily on logarithms and exponentials. It is easy to generate examples where numeric instability sets in.
-   as a consequence, the `numeric` versions of the minimum distance estimator (which use numerical derivatives) are not recommended.
-   the bias-corrected minimum distance estimator (`corrected`) may have a larger mean-squared error and/or introduce numerical instabilities. \## Release notes

### version 1.0.8

-   deleted spurious print statement.

### version 1.0.7

-   fixed error in bias-correction term.

### version 1.0.6

-   corrected typo.

### version 1.0.5

-   simplified the bias-correction for the minimum distance estimator in the Choo and Siow homoskedastic model.

### version 1.0.4

-   added an optional bias-correction for the minimum distance estimator in the Choo and Siow homoskedastic model, to help with cases when the matching patterns vary a lot across cells.
-   added two complete examples: `example_choosiow.py` and `example_nestedlogit.py`.
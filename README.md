# cupid_matching

**A Python package to solve, simulate and estimate separable matching models**

- Free software: MIT license
- Documentation: [https://bsalanie.github.io/cupid_matching](https://bsalanie.github.io/cupid_matching)
- See also: [An interactive Streamlit app](https://share.streamlit.io/bsalanie/cupid_matching_st/main/cupid_streamlit.py)

## Installation

```
pip install [-U] cupid_matching
```

## Importing functions from the package

For instance:

```py
from cupid_matching.min_distance import estimate_semilinear_mde
```

## Examples
* `example_choosiow.py` shows how to run minimum distance and Poisson estimators on a Choo and Siow homoskedastic model. 
* `example_nestedlogit.py` shows how to run minimum distance estimators on a two-layer nested logit model. 


## Warnings
* many of these models (including all Cho and Siow variants) rely heaviliy on logarithms and exponentials. It is easy ton generate examples where numeric instability sets in.
* as a consequence,  the `numeric` versions of the minimum distance estimator (which use numerical derivatives) are not recommended. 
* the bias-corrected minimum distance estimator (`corrected`) may have a larger mean-squared error and/or introduce numerical instabilities.
## Release notes
### version 1.0.3
* added an optional bias-correction for the minimum distance estimator in the Choo and Siow homoskedastic model, to help with cases when the matching patterns vary a lot across cells.
* added two complete examples: `example_choosiow.py` and `example_nestedlogit.py`


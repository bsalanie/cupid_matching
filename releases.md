
### version 1.3 (December 8, 2023)
- added `CupidMatchingDoc.pdf` on Github, with detailed explanations of the methods. 

### version 1.2 (November 29, 2023)
- incorporates models without singles for both MDE and Poisson; example in `example_choo_siow_no_singles.py`.

### version 1.1.3 (November 8, 2023)
- fixed URL of Streamlit app.

### version 1.1.2 (November 7, 2023)
-  improved the Streamlit app, now in two files: `cupid_streamlit.py` and  `cupid_streamlit_utils.py`.

### version 1.1.1
-  improved documentation
-  the package now relies on my utilities package `bs_python_utils`.  The `VarianceMatching` class in `matching_utils,py` is new; this should be transparent for the user.

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
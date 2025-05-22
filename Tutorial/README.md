# Tutorial on the change of basis technique

The following notebooks and Python script provide a step-by-step tutorial on how to use the change of basis introduced in the paper for any Hermitian matrix. 

* For an in-depth walkthrough, where the explicit matrices are built as well as their inverses, and then these are compared against their operator forms, we refer to notebooks [(A)]((A)%20Change%20of%20Basis%20-%20Understanding%20Q.ipynb) and [(B)]((B)%20Change%20of%20Basis%20-%20Understanding%20U.ipynb). The two notebooks also show how the operators can be applied in chain form to partially diagonalise a matrix for two eigenvectors that were found iteratively using Lanczos method.

* All routines have been packed in the script [`EigenDeflation.py`](EigenDeflation.py).

* For two quick examples that exemplify the module's functionality, refer to the notebooks [(C)]((C)%20Q%20-%20Example.ipynb) and [(D)]((D)%20U%20-%20Example.ipynb).

---
## Dependencies

The original code used the following standard libraries:
`NumPy 2.2.5`, `SciPy 1.15.1`, `SymPy 1.13.1`.

---
For better visualisation of the notebooks, the **codefolding for Jupyter** [extension](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/) is recommended.





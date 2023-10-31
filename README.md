## Diff-ICP : diffeomorphic ICP algorithm for registration of single and multiple point sets

### Summary

- Extends the classic iterative closest point (ICP) algorithm for point set registration.
- Registration functions are **diffeomorphic mappings**, following the *large deviation diffeomorphic metric mapping* (LDDMM) framework.
- Variations of the same algorithm can be used to register either:
  - One point set on a second point set (the classic ICP problem).
  - Multiple point sets to a common Gaussian Mixture Model which is inferred from the data.
  In the medical imaging literature, this is sometimes called *groupwise ICP*, and its result is an example of *statistical atlas*.
- The optimization problem to be solved is formulated as a classic Bayesian probabilistic inference, with a *prior* term implementing smoothness constraints on the registration mappings.
- The algorithm is documented in : *A. Wohrer*, **Diffeomorphic ICP registration for single and multiple point sets**, *Geometric Science of Information 2023* (proceedings)

### Implementation

- Written in Python
- Runs on CPU (default) or Nvidia GPU (if available), allowing for 10 to 100-fold speed gains
- Point sets and various operations on them are implemented with the **Torch** library
- Fast and robust kernel computations are implemented thanks to the **KeOps** library

### Dependencies

Aside from the classics (scipy, numpy, matplotlib), Python modules **torch** and **pykeops** should be installed (e.g., with pip).

### Contents

Directory **diffICP/core** contains the core models. Mainly:
  - *GMM.py* handles all Gaussian-Mixture-Model-related functions (e.g., EM algorithm for GMM clustering).
  - *LDDMM.py* handles the diffeomorphic registration functions.
  - *PSR.py* defines the full Point Set Registration algorithm, based on the interplay between the two above.
  - *affine.py* handles alternative registration functions based on affine (e.g., rigid, or more general linear) transformations. This can be used for comparison and/or preprocessing.
  - *PSR_standard.py* implements the 'standard' diffeomorphic Point Set Registration algorithm of Glaunès et al, for comparison.

Directory **diffICP/api** contains user-friendly interfaces for 
  - the two-point-set matching problem, with various algorithms.
  - building a groupwise point-set atlas, with various algorithms.
  - 
Directory **diffICP/examples** contains some use cases (somewhat obsolete now, better see **diffICP/api** instead)
  - *diffICP_basic.py* illustrates registration of a single point set to a known GMM model (Fig. 1 of the GSI article)
  - *diffICP_multi.py* illustrates registration of multiple point sets to a common GMM model that is inferred from the data (Fig. 2 of the GSI article)
  - *diffICP_full.py* illustrates a full model with multiple frames (e.g., patients) *and* multiple GMM models (e.g., brain structures) (not documented in the GSI article)

Directories **diffICP/tools** and **diffICP/visualization** contain a number of helper functions.

### Disclaimer

In its current state, this code is **experimental**, and does not pretend to have reached (yet) the status of a distributable module. The code is provided as is, for whoever may be interested in the algorithms.
It is only documented through comments in the code (70% english, 30% french :))

Diff-ICP : diffeomorphic ICP algorithms for registration of single and multiple point set

SUMMARY:
- Extends the classic iterative closest point (ICP) for point set registration
- Registration functions are diffeomorphic mappings, following the large deviation diffeomorphic metric mapping (LDDMM) framework
- The algorithms can be used to register
  - one point set on a second point set (classic ICP problem)
  - multiple points sets to a common Gaussian Mixture Model (groupwise ICP), which can then be interpreted as a statistical atlas of the multiple point sets
- The optimization problem to be solved is formulated as a well-posed probabilistic inference
- The algorithm is documented in : A. Wohrer, Diffeomorphic ICP registration for single and multiple point sets, Geometric Science of Information 2023 (proceedings)

IMPLEMENTATION:
- Written in Python
- Runs on CPU (default) or Nvidia GPU if avaiable
- Point sets and operations on them are implemented with the Torch library
- Fast kernel computations are implemented thanks to the KeOps library

DEPENDENCIES: 

Aside from the classics (scipy, numpy, matplotlib), Python modules **torch** and **pykeops** should be installed (e.g., with pip).

CONTENTS:
- Directory **diffICP/** defines a python module containing the core functions. Mainly:
  - *GMM.py* handles all Gaussian-Mixture-Model-related functionalities (e.g., EM algorithm for GMM clustering)
  - *LDDMM_logdet.py* handles the diffeomorphic registration functionalities
  - *PSR.py* defines the full Point Set Registration algorithm, based on the interplay between the two above.
  - *Affine_logdet.py* handles alternative registration functions based on affine (e.g., rigid, or general linear) transformations. This can be used for comparison and/or preprocessing.
  
  
- Directory **examples/** contains some use cases
  - *diffICP_basic.py* illustrates registration of a single point set to a known GMM model (Fig. 1 of the GSI article)
  - *diffICP_multi.py* illustrates registration of multiple point sets to a common GMM model that is inferred from the data (Fig. 2 of the GSI article)
  - *diffICP_full.py* illustrates a full model with multiple frames (e.g., patients) *and* multiple GMM models (e.g., brain structures) (not documented in the GSI article)
  
DISCLAIMER:

In the current state, this code is **experimental**, and does not pretend to have reached (yet) the status of a distributable module. The code is provided as is, for whoever may be interested in the algorithms.
It is only documented through comments in the code (70% english, 30% french :))

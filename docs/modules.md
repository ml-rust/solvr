# Module Reference

This page lists the algorithms in each solvr module. Each module lives under `src/<module>/`
and is re-exported from the crate root.

## signal

- Convolution and correlation
- STFT, inverse STFT, and spectrograms
- Spectral analysis, including Lomb-Scargle periodograms
- Continuous wavelet transform (CWT)
- FIR design (`firwin`) and IIR design, including Butterworth filters
- N-D filters and edge operations

## window

- Hann, Hamming, Blackman, and Kaiser windows, and related window generators

## interpolate

- 1D and N-D linear and cubic interpolation, and splines
- PCHIP and Akima interpolation
- Radial basis function (RBF) interpolation
- Geometric interpolation and transforms: affine, zoom, and rotate

## optimize

- Scalar minimization and root finding, including Brent's method
- Unconstrained minimization, including Newton methods
- Least squares
- Linear programming, quadratic programming, and conic programming (SDP, SOCP)
- Global methods: differential evolution, dual annealing, basin hopping

## integrate

- Numerical quadrature, including fixed-order, Romberg, and multi-dimensional quadrature
- ODE solvers, including Radau, LSODA, and BDF
- DAE solvers
- BVP solvers
- Sensitivity analysis with the adjoint method

## stats

- Continuous and discrete distributions
- Descriptive statistics
- Hypothesis tests
- Regression
- Information metrics

## spatial

- KDTree and BallTree
- Distance metrics
- Convex hull, Delaunay triangulation, and Voronoi diagrams
- Rotations
- Mesh operations
- Distance transforms

## morphology

- Binary and grayscale erosion, dilation, opening, and closing
- Connected-region labeling
- Region measurements

## cluster

- k-means, including mini-batch and bisecting variants
- DBSCAN, OPTICS, and HDBSCAN
- Gaussian mixture models (GMM) and Bayesian GMM
- Hierarchical clustering
- Spectral clustering

## linalg

- Matrix-equation solvers: Lyapunov, Riccati, and Sylvester
- Sparse QR (feature `sparse`)
- Advanced linear algebra helpers

## graph (feature `graph`)

- Shortest paths: Dijkstra and Bellman-Ford
- Connectivity
- Centrality: degree, closeness, betweenness, and eigenvector
- Minimum spanning tree
- Maximum flow and minimum-cost flow

## pde (feature `pde`)

- Finite-difference PDE tooling
- Finite-element PDE tooling
- Spectral PDE tooling

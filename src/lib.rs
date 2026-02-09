//! solvr - Advanced Computing Library for Real-World Problem Solving
//!
//! solvr provides production-ready algorithms for optimization, differential equations,
//! interpolation, statistics, signal processing, and spatial computing. Built on numr's
//! foundational math primitives, it works across all backends (CPU, CUDA, WebGPU).
//!
//! # When to Use solvr vs numr
//!
//! - **numr**: Foundational math (tensors, FFT, basic linalg). Most users only need this.
//! - **solvr**: Advanced algorithms for solving real-world problems (optimization, ODE,
//!   interpolation, statistical tests, etc.)
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │                      solvr                               │
//! │   (optimization, ODE, interpolation, stats, spatial)    │
//! └──────────────────────────┬──────────────────────────────┘
//!                            │ uses
//! ┌──────────────────────────▼──────────────────────────────┐
//! │                       numr                               │
//! │     (tensors, FFT, matmul, basic linalg, special fn)    │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Current Modules
//!
//! - [`signal`] - Signal processing (convolution, correlation, STFT, spectrogram, N-D filters, edge detection)
//! - [`window`] - Window functions (Hann, Hamming, Blackman, Kaiser)
//! - [`interpolate`] - Interpolation (linear, cubic, splines, geometric transforms: affine, zoom, rotate)
//! - [`optimize`] - Root finding, minimization, least squares, linear programming
//! - [`integrate`] - ODE solvers and numerical quadrature
//! - [`stats`] - Statistical distributions, hypothesis tests, descriptive stats
//! - [`spatial`] - KDTree, BallTree, distance metrics, convex hull, Delaunay, Voronoi, rotations, distance transforms
//! - [`morphology`] - Morphological operations (binary/grey erosion, dilation, opening, closing, connected components)
//!
//! # Planned Modules
//!
//!
//! # Backend Support
//!
//! solvr is generic over numr's `Runtime` trait. The same code works on:
//! - CPU (with SIMD acceleration)
//! - CUDA (NVIDIA GPUs)
//! - WebGPU (cross-platform GPU)
//!
//! # Feature Flags
//!
//! | Feature | Description | Dependencies |
//! |---------|-------------|--------------|
//! | `cuda`  | Enable CUDA GPU acceleration | CUDA 12.x, numr/cuda |
//! | `wgpu`  | Enable WebGPU cross-platform GPU | numr/wgpu |
//!
//! ## Usage
//!
//! ```toml
//! # CPU only (default)
//! solvr = "0.1"
//!
//! # With CUDA support
//! solvr = { version = "0.1", features = ["cuda"] }
//!
//! # With WebGPU support
//! solvr = { version = "0.1", features = ["wgpu"] }
//! ```
//!
//! ## Backend Limitations
//!
//! - **WebGPU**: Only supports F32 precision (no F64)
//! - **CUDA**: Requires CUDA 12.x toolkit installed
//!
//! # Example
//!
//! ```
//! # use numr::runtime::cpu::{CpuClient, CpuDevice};
//! # use numr::runtime::RuntimeClient;
//! # use numr::dtype::DType;
//! # use numr::ops::RandomOps;
//! use solvr::signal::{ConvolutionAlgorithms, ConvMode};
//! use solvr::window::WindowFunctions;
//! # let device = CpuDevice::new();
//! # let client = CpuClient::new(device.clone());
//! // Create a Hann window
//! # let window = client.hann_window(256, DType::F32, &device).unwrap();
//! // Perform convolution
//! # let signal = client.randn(&[256], DType::F32)?;
//! # let kernel = client.randn(&[32], DType::F32)?;
//! let result = client.convolve(&signal, &kernel, ConvMode::Same)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod cluster;
pub mod common;
#[cfg(feature = "graph")]
pub mod graph;
pub mod integrate;
pub mod interpolate;
pub mod linalg;
pub mod morphology;
pub mod optimize;
#[cfg(feature = "pde")]
pub mod pde;
pub mod signal;
pub mod spatial;
pub mod stats;
pub mod window;

// Re-export main types for convenience
#[cfg(feature = "graph")]
pub use graph::{
    AllPairsResult, CentralityAlgorithms, ComponentResult, ConnectivityAlgorithms,
    EigCentralityOptions, FlowAlgorithms, FlowResult, GraphData, GraphMatrixAlgorithms,
    MSTAlgorithms, MSTResult, MinCostFlowOptions, PageRankOptions, PathResult,
    ShortestPathAlgorithms, ShortestPathResult,
};
pub use integrate::{
    IntegrateError, IntegrateResult, IntegrationAlgorithms, ODEMethod, ODEOptions, ODEResultTensor,
    QuadOptions, QuadResult, RombergOptions, solve_ivp_impl,
};
pub use interpolate::{
    AkimaAlgorithms, BSplineCurve, BSplineCurveAlgorithms, BSplineSurface,
    BSplineSurfaceAlgorithms, BezierCurve, BezierCurveAlgorithms, BezierSurface,
    BezierSurfaceAlgorithms, CubicSplineAlgorithms, ExtrapolateMode, Interp1dAlgorithms,
    InterpMethod, InterpNdAlgorithms, InterpNdMethod, NurbsCurve, NurbsCurveAlgorithms,
    NurbsSurface, NurbsSurfaceAlgorithms, PchipAlgorithms, SplineBoundary,
};
pub use linalg::MatrixEquationAlgorithms;
pub use morphology::{
    BinaryMorphologyAlgorithms, GreyMorphologyAlgorithms, MeasurementAlgorithms, RegionProperties,
    StructuringElement,
};
pub use optimize::{OptimizeError, OptimizeResult, scalar::*};
#[cfg(feature = "pde")]
pub use pde::{
    BoundaryCondition, BoundarySide, BoundarySpec, FdmOptions, FdmResult, FemResult,
    FiniteDifferenceAlgorithms, FiniteElementAlgorithms, Grid2D, Grid3D, PdeError, PdeResult,
    SpectralAlgorithms, SpectralResult, TimeDependentOptions, TimeResult,
};
pub use signal::{ConvMode, ConvolutionAlgorithms, SpectrogramAlgorithms, StftAlgorithms};
pub use spatial::{
    BallTree,
    BallTreeAlgorithms,
    BallTreeOptions,
    // Computational geometry
    ConvexHull,
    ConvexHullAlgorithms,
    Delaunay,
    DelaunayAlgorithms,
    // Distance algorithms
    DistanceAlgorithms,
    DistanceMetric,
    EulerOrder,
    // Halfspace intersection
    HalfspaceIntersection,
    HalfspaceIntersectionAlgorithms,
    // Spatial trees
    KDTree,
    KDTreeAlgorithms,
    KDTreeOptions,
    KNNResult,
    // Mesh processing
    Mesh,
    MeshAlgorithms,
    ProcrustesAlgorithms,
    ProcrustesResult,
    RadiusResult,
    // Transforms
    Rotation,
    RotationAlgorithms,
    SimplificationMethod,
    SmoothingMethod,
    // Spherical Voronoi
    SphericalVoronoi,
    SphericalVoronoiAlgorithms,
    Voronoi,
    VoronoiAlgorithms,
};
pub use stats::{
    // Continuous distributions
    Beta,
    // Discrete distributions
    Binomial,
    Cauchy,
    ChiSquared,
    // Distribution traits
    ContinuousDistribution,
    // Runtime-generic traits for statistics operations (split into focused traits)
    DescriptiveStatisticsAlgorithms,
    DiscreteDistribution,
    DiscreteUniform,
    Distribution,
    Exponential,
    FDistribution,
    Gamma,
    Geometric,
    Gumbel,
    GumbelMin,
    Hypergeometric,
    HypothesisTestingAlgorithms,
    Laplace,
    LinregressResult,
    LogNormal,
    NegativeBinomial,
    Normal,
    Pareto,
    Poisson,
    RegressionAlgorithms,
    // Errors
    StatsError,
    StatsResult,
    StudentT,
    // Result types for tensor operations
    TensorDescriptiveStats,
    TensorTestResult,
    Uniform,
    Weibull,
};
pub use window::WindowFunctions;

// Re-export numr types that users will commonly need
pub use numr::dtype::DType;
pub use numr::error::{Error, Result};
pub use numr::runtime::{Runtime, RuntimeClient};
pub use numr::tensor::Tensor;

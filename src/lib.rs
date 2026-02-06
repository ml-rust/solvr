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
//! - [`signal`] - Signal processing (convolution, correlation, STFT, spectrogram)
//! - [`window`] - Window functions (Hann, Hamming, Blackman, Kaiser)
//! - [`interpolate`] - Interpolation methods (linear, cubic, splines)
//! - [`optimize`] - Root finding, minimization, least squares, linear programming
//! - [`integrate`] - ODE solvers and numerical quadrature
//! - [`stats`] - Statistical distributions, hypothesis tests, descriptive stats
//! - [`spatial`] - KDTree, BallTree, distance metrics, convex hull, Delaunay, Voronoi, rotations
//!
//! # Planned Modules
//!
//! - `ndimage` - N-dimensional image processing
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
//! solvr = "0.0"
//!
//! # With CUDA support
//! solvr = { version = "0.0", features = ["cuda"] }
//!
//! # With WebGPU support
//! solvr = { version = "0.0", features = ["wgpu"] }
//! ```
//!
//! ## Backend Limitations
//!
//! - **WebGPU**: Only supports F32 precision (no F64)
//! - **CUDA**: Requires CUDA 12.x toolkit installed
//!
//! # Example
//!
//! ```ignore
//! use solvr::signal::{ConvolutionAlgorithms, ConvMode};
//! use solvr::window::WindowFunctions;
//! use numr::runtime::cpu::{CpuClient, CpuDevice};
//! use numr::runtime::RuntimeClient;
//! use numr::dtype::DType;
//!
//! let device = CpuDevice::new();
//! let client = CpuClient::new(device.clone());
//!
//! // Create a Hann window
//! let window = client.hann_window(256, DType::F32, &device).unwrap();
//!
//! // Perform convolution
//! let signal = /* ... */;
//! let kernel = /* ... */;
//! let result = client.convolve(&signal, &kernel, ConvMode::Same).unwrap();
//! ```

pub mod cluster;
pub mod common;
pub mod integrate;
pub mod interpolate;
pub mod optimize;
pub mod signal;
pub mod spatial;
pub mod stats;
pub mod window;

// Re-export main types for convenience
pub use integrate::{
    IntegrateError, IntegrateResult, IntegrationAlgorithms, ODEMethod, ODEOptions, ODEResultTensor,
    QuadOptions, QuadResult, RombergOptions, solve_ivp_impl,
};
pub use interpolate::{
    AkimaAlgorithms, CubicSplineAlgorithms, ExtrapolateMode, Interp1dAlgorithms, InterpMethod,
    InterpNdAlgorithms, InterpNdMethod, PchipAlgorithms, SplineBoundary,
};
pub use optimize::{OptimizeError, OptimizeResult, scalar::*};
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
    // Spatial trees
    KDTree,
    KDTreeAlgorithms,
    KDTreeOptions,
    KNNResult,
    ProcrustesAlgorithms,
    ProcrustesResult,
    RadiusResult,
    // Transforms
    Rotation,
    RotationAlgorithms,
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

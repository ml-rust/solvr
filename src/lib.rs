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
//! - [`optimize`] - Root finding and minimization (scalar methods)
//!
//! # Planned Modules
//!
//! - `optimize` (expand) - Multivariate minimization, least squares, linear programming
//! - `integrate` - ODE/PDE solvers, numerical quadrature
//! - `stats` - Distributions, hypothesis tests, regression
//! - `spatial` - KDTree, distance metrics, geometric algorithms
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
//! use solvr::signal::{SignalProcessingAlgorithms, ConvMode};
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

pub mod interpolate;
pub mod optimize;
pub mod signal;
pub mod window;

// Re-export main types for convenience
pub use interpolate::{
    Akima1DInterpolator, CubicSpline, ExtrapolateMode, Interp1d, InterpMethod, InterpNdMethod,
    PchipInterpolator, RegularGridInterpolator, SplineBoundary,
};
pub use optimize::{OptimizeError, OptimizeResult, scalar::*};
pub use signal::{ConvMode, SignalProcessingAlgorithms};
pub use window::WindowFunctions;

// Re-export numr types that users will commonly need
pub use numr::dtype::DType;
pub use numr::error::{Error, Result};
pub use numr::runtime::{Runtime, RuntimeClient};
pub use numr::tensor::Tensor;

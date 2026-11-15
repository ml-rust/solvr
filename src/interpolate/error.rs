//! Error types for interpolation operations.

use crate::DType;
use std::fmt;

/// Result type for interpolation operations.
pub type InterpolateResult<T> = Result<T, InterpolateError>;

/// Errors that can occur during interpolation.
#[derive(Debug, Clone)]
pub enum InterpolateError {
    /// Input arrays have mismatched lengths.
    ShapeMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },

    /// Input array is too small for the requested operation.
    InsufficientData {
        required: usize,
        actual: usize,
        context: String,
    },

    /// Query point is outside the interpolation domain (1D).
    OutOfDomain {
        point: f64,
        min: f64,
        max: f64,
        context: String,
    },

    /// Query point is outside the interpolation domain (N-D).
    OutOfDomainNd {
        dimension: usize,
        point: f64,
        min: f64,
        max: f64,
        context: String,
    },

    /// Dimension mismatch in N-D operations.
    DimensionMismatch {
        expected: usize,
        actual: usize,
        context: String,
    },

    /// Input x values are not strictly increasing.
    NotMonotonic { context: String },

    /// Numerical computation failed (e.g., singular matrix).
    NumericalError { message: String },

    /// Diagnostic interpolation lacks a declared provider contract or detects unsupported execution-mode arithmetic.
    UnsupportedArithmetic {
        /// Selected input floating dtype.
        dtype: DType,
        /// Static `provider_contract` rejection or detected execution-mode probe categories.
        failed_probes: Vec<&'static str>,
    },

    /// Invalid parameter value.
    InvalidParameter { parameter: String, message: String },

    /// Error from underlying numr operation.
    NumrError(String),
}

impl fmt::Display for InterpolateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Shape mismatch in {}: expected {}, got {}",
                    context, expected, actual
                )
            }
            Self::InsufficientData {
                required,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Insufficient data for {}: need at least {}, got {}",
                    context, required, actual
                )
            }
            Self::OutOfDomain {
                point,
                min,
                max,
                context,
            } => {
                write!(
                    f,
                    "Point {} is outside interpolation domain [{}, {}] in {}",
                    point, min, max, context
                )
            }
            Self::OutOfDomainNd {
                dimension,
                point,
                min,
                max,
                context,
            } => {
                write!(
                    f,
                    "Point {} in dimension {} is outside domain [{}, {}] in {}",
                    point, dimension, min, max, context
                )
            }
            Self::DimensionMismatch {
                expected,
                actual,
                context,
            } => {
                write!(
                    f,
                    "Dimension mismatch in {}: expected {}, got {}",
                    context, expected, actual
                )
            }
            Self::NotMonotonic { context } => {
                write!(
                    f,
                    "Input x values must be strictly increasing in {}",
                    context
                )
            }
            Self::NumericalError { message } => {
                write!(f, "Numerical error: {}", message)
            }
            Self::UnsupportedArithmetic {
                dtype,
                failed_probes,
            } => {
                write!(
                    f,
                    "Unsupported Linear receipt arithmetic: dtype={dtype:?}, failed_probes={}. Select an IEEE-basic provider and an arithmetic mode supporting nearest-even rounding, gradual underflow, and Numr TwoSum witnesses",
                    failed_probes.join(", ")
                )
            }
            Self::InvalidParameter { parameter, message } => {
                write!(f, "Invalid parameter '{}': {}", parameter, message)
            }
            Self::NumrError(msg) => {
                write!(f, "numr error: {}", msg)
            }
        }
    }
}

impl std::error::Error for InterpolateError {}

impl From<numr::error::Error> for InterpolateError {
    fn from(err: numr::error::Error) -> Self {
        Self::NumrError(err.to_string())
    }
}

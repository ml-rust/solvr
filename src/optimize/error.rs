//! Error types for optimization operations.

use std::fmt;

/// Result type for optimization operations.
pub type OptimizeResult<T> = Result<T, OptimizeError>;

/// Errors that can occur during optimization.
#[derive(Debug, Clone)]
pub enum OptimizeError {
    /// The solver did not converge within the maximum iterations.
    DidNotConverge {
        iterations: usize,
        tolerance: f64,
        context: String,
    },

    /// Invalid interval provided (e.g., for bracketing methods).
    InvalidInterval { a: f64, b: f64, context: String },

    /// Function has the same sign at both bracket endpoints.
    SameSignBracket { fa: f64, fb: f64, context: String },

    /// Invalid parameter value.
    InvalidParameter { parameter: String, message: String },

    /// Numerical computation failed (e.g., division by zero).
    NumericalError { message: String },

    /// Invalid input array size or dimensions.
    InvalidInput { context: String },

    /// Error from underlying numr operation.
    NumrError(String),
}

impl fmt::Display for OptimizeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DidNotConverge {
                iterations,
                tolerance,
                context,
            } => {
                write!(
                    f,
                    "{}: did not converge after {} iterations (tolerance: {})",
                    context, iterations, tolerance
                )
            }
            Self::InvalidInterval { a, b, context } => {
                write!(
                    f,
                    "Invalid interval [{}, {}] in {}: bounds must satisfy a < b",
                    a, b, context
                )
            }
            Self::SameSignBracket { fa, fb, context } => {
                write!(
                    f,
                    "Function has same sign at bracket endpoints in {}: f(a)={}, f(b)={}",
                    context, fa, fb
                )
            }
            Self::InvalidParameter { parameter, message } => {
                write!(f, "Invalid parameter '{}': {}", parameter, message)
            }
            Self::NumericalError { message } => {
                write!(f, "Numerical error: {}", message)
            }
            Self::InvalidInput { context } => {
                write!(f, "Invalid input in {}", context)
            }
            Self::NumrError(msg) => {
                write!(f, "numr error: {}", msg)
            }
        }
    }
}

impl std::error::Error for OptimizeError {}

impl From<numr::error::Error> for OptimizeError {
    fn from(err: numr::error::Error) -> Self {
        Self::NumrError(err.to_string())
    }
}

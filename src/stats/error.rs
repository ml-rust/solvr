//! Error types for statistical operations.

use std::fmt;

/// Result type for statistics operations.
pub type StatsResult<T> = Result<T, StatsError>;

/// Errors that can occur during statistical operations.
#[derive(Debug, Clone)]
pub enum StatsError {
    /// Invalid parameter value for a distribution.
    InvalidParameter {
        name: String,
        value: f64,
        reason: String,
    },

    /// Input data is empty when non-empty data is required.
    EmptyData { context: String },

    /// Input data has insufficient length.
    InsufficientData {
        required: usize,
        got: usize,
        context: String,
    },

    /// Probability value out of range [0, 1].
    InvalidProbability { value: f64 },

    /// Value is out of the distribution's support.
    OutOfSupport { value: f64, support: String },

    /// Numerical computation failed.
    NumericalError { message: String },

    /// Iterative method did not converge.
    ConvergenceError { iterations: usize, context: String },

    /// Mismatched array lengths.
    LengthMismatch {
        expected: usize,
        got: usize,
        context: String,
    },
}

impl fmt::Display for StatsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameter {
                name,
                value,
                reason,
            } => {
                write!(f, "Invalid parameter '{}' = {}: {}", name, value, reason)
            }
            Self::EmptyData { context } => {
                write!(f, "Empty data in {}", context)
            }
            Self::InsufficientData {
                required,
                got,
                context,
            } => {
                write!(
                    f,
                    "Insufficient data in {}: need {} elements, got {}",
                    context, required, got
                )
            }
            Self::InvalidProbability { value } => {
                write!(f, "Invalid probability {}: must be in [0, 1]", value)
            }
            Self::OutOfSupport { value, support } => {
                write!(f, "Value {} is outside support {}", value, support)
            }
            Self::NumericalError { message } => {
                write!(f, "Numerical error: {}", message)
            }
            Self::ConvergenceError {
                iterations,
                context,
            } => {
                write!(
                    f,
                    "{} did not converge after {} iterations",
                    context, iterations
                )
            }
            Self::LengthMismatch {
                expected,
                got,
                context,
            } => {
                write!(
                    f,
                    "Length mismatch in {}: expected {}, got {}",
                    context, expected, got
                )
            }
        }
    }
}

impl std::error::Error for StatsError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = StatsError::InvalidParameter {
            name: "sigma".to_string(),
            value: -1.0,
            reason: "must be positive".to_string(),
        };
        assert!(err.to_string().contains("sigma"));
        assert!(err.to_string().contains("-1"));

        let err = StatsError::InvalidProbability { value: 1.5 };
        assert!(err.to_string().contains("1.5"));
        assert!(err.to_string().contains("[0, 1]"));
    }
}

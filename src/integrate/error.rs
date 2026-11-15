//! Error types for numerical integration operations.

use crate::DType;
use std::fmt;

/// Result type for integration operations.
pub type IntegrateResult<T> = Result<T, IntegrateError>;

/// Errors that can occur during numerical integration.
#[derive(Debug, Clone)]
pub enum IntegrateError {
    /// The solver did not converge within the specified tolerance.
    DidNotConverge {
        iterations: usize,
        tolerance: f64,
        context: String,
    },

    /// Invalid interval provided (e.g., a >= b).
    InvalidInterval { a: f64, b: f64, context: String },

    /// Invalid parameter value.
    InvalidParameter { parameter: String, message: String },

    /// Numerical computation failed (e.g., division by zero, NaN).
    NumericalError { message: String },

    /// Certified integration lacks the required arithmetic contract or execution mode.
    UnsupportedArithmetic {
        /// Selected floating-point input dtype.
        dtype: DType,
        /// Missing provider contract or execution-mode probe categories.
        failed_probes: Vec<&'static str>,
    },

    /// Certified integration cannot represent finite bounds at this calculation stage.
    ArithmeticRange { stage: &'static str },

    /// Invalid input array size or dimensions.
    InvalidInput { context: String },

    /// Step size became too small during adaptive integration.
    StepSizeTooSmall { step: f64, t: f64, context: String },

    /// Maximum number of steps exceeded.
    MaxStepsExceeded {
        steps: usize,
        t: f64,
        context: String,
    },

    /// Maximum number of subdivisions exceeded (for adaptive quadrature).
    MaxSubdivisionsExceeded {
        subdivisions: usize,
        context: String,
    },

    /// Integration encountered a singularity.
    SingularityDetected { x: f64, context: String },

    /// Error from underlying numr operation.
    NumrError(String),

    // ========================================================================
    // DAE-Specific Errors
    // ========================================================================
    /// Initial conditions are inconsistent with the DAE constraints.
    ///
    /// For a DAE F(t, y, y') = 0, the provided (y0, yp0) do not satisfy
    /// F(t0, y0, yp0) ≈ 0 and could not be refined within tolerance.
    InconsistentInitialConditions {
        residual_norm: f64,
        tolerance: f64,
        iterations: usize,
    },

    /// Newton iteration failed to converge for DAE step.
    DAENewtonFailed {
        t: f64,
        residual_norm: f64,
        iterations: usize,
    },

    /// Algebraic constraint was violated during integration.
    DAEConstraintViolation { t: f64, constraint_norm: f64 },
}

impl fmt::Display for IntegrateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DidNotConverge {
                iterations,
                tolerance,
                context,
            } => {
                write!(
                    f,
                    "{}: did not converge after {} iterations (tolerance: {:.2e})",
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
            Self::InvalidParameter { parameter, message } => {
                write!(f, "Invalid parameter '{}': {}", parameter, message)
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
                    "Certified Simpson arithmetic rejected dtype={dtype:?}, failed_probes={}. Select an IEEE-basic runtime and nearest-even arithmetic with gradual underflow and Numr TwoSum witnesses",
                    failed_probes.join(", ")
                )
            }
            Self::ArithmeticRange { stage } => {
                write!(
                    f,
                    "Certified Simpson cannot represent finite bounds at stage={stage}. Rescale the inputs or split the interval"
                )
            }
            Self::InvalidInput { context } => {
                write!(f, "Invalid input: {}", context)
            }
            Self::StepSizeTooSmall { step, t, context } => {
                write!(
                    f,
                    "{}: step size {:.2e} too small at t = {:.6}",
                    context, step, t
                )
            }
            Self::MaxStepsExceeded { steps, t, context } => {
                write!(
                    f,
                    "{}: exceeded maximum {} steps at t = {:.6}",
                    context, steps, t
                )
            }
            Self::MaxSubdivisionsExceeded {
                subdivisions,
                context,
            } => {
                write!(
                    f,
                    "{}: exceeded maximum {} subdivisions",
                    context, subdivisions
                )
            }
            Self::SingularityDetected { x, context } => {
                write!(f, "{}: singularity detected near x = {:.6}", context, x)
            }
            Self::NumrError(msg) => {
                write!(f, "numr error: {}", msg)
            }
            Self::InconsistentInitialConditions {
                residual_norm,
                tolerance,
                iterations,
            } => {
                write!(
                    f,
                    "DAE initial conditions inconsistent: residual norm {:.2e} > tolerance {:.2e} after {} iterations",
                    residual_norm, tolerance, iterations
                )
            }
            Self::DAENewtonFailed {
                t,
                residual_norm,
                iterations,
            } => {
                write!(
                    f,
                    "DAE Newton iteration failed at t = {:.6}: residual {:.2e} after {} iterations",
                    t, residual_norm, iterations
                )
            }
            Self::DAEConstraintViolation { t, constraint_norm } => {
                write!(
                    f,
                    "DAE constraint violated at t = {:.6}: norm = {:.2e}",
                    t, constraint_norm
                )
            }
        }
    }
}

impl std::error::Error for IntegrateError {}

impl From<numr::error::Error> for IntegrateError {
    fn from(err: numr::error::Error) -> Self {
        Self::NumrError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = IntegrateError::DidNotConverge {
            iterations: 100,
            tolerance: 1e-8,
            context: "quad".to_string(),
        };
        assert!(err.to_string().contains("did not converge"));
        assert!(err.to_string().contains("100"));

        let err = IntegrateError::InvalidInterval {
            a: 5.0,
            b: 3.0,
            context: "trapezoid".to_string(),
        };
        assert!(err.to_string().contains("Invalid interval"));

        let err = IntegrateError::StepSizeTooSmall {
            step: 1e-15,
            t: 0.5,
            context: "RK45".to_string(),
        };
        assert!(err.to_string().contains("too small"));
    }

    #[test]
    fn unsupported_arithmetic_names_dtype_probes_and_next_action() {
        let error = IntegrateError::UnsupportedArithmetic {
            dtype: DType::F64,
            failed_probes: vec!["provider_contract", "gradual_subnormal_inputs"],
        };
        assert_eq!(
            error.to_string(),
            "Certified Simpson arithmetic rejected dtype=F64, failed_probes=provider_contract, gradual_subnormal_inputs. Select an IEEE-basic runtime and nearest-even arithmetic with gradual underflow and Numr TwoSum witnesses"
        );
    }

    #[test]
    fn arithmetic_range_names_stage_and_next_action() {
        let error = IntegrateError::ArithmeticRange {
            stage: "panel remainder",
        };
        assert_eq!(
            error.to_string(),
            "Certified Simpson cannot represent finite bounds at stage=panel remainder. Rescale the inputs or split the interval"
        );
    }
}

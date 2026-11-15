//! Domain-independent arithmetic errors for thin solver adapters.

use crate::DType;
use std::fmt;

/// Result of checked interval arithmetic or arithmetic admission.
pub type ArithmeticResult<T> = Result<T, ArithmeticError>;

/// Arithmetic admission, input, or bound errors.
///
/// Read the operation stage and error details before changing the calculation.
#[derive(Debug)]
pub enum ArithmeticError {
    /// The selected dtype is outside the F32/F64 arithmetic domain.
    UnsupportedDType {
        /// Rejected dtype.
        dtype: DType,
    },
    /// A probe control tensor has an unsupported dtype.
    ProbeControlDType,
    /// A probe control value is not an exact integer mask in `0..=255`.
    InvalidProbeControl {
        /// Rejected control value.
        control: f64,
    },
    /// The provider declaration or execution-mode probes do not admit this arithmetic.
    UnsupportedArithmetic {
        /// Selected floating dtype.
        dtype: DType,
        /// Rejected provider declaration or execution-mode checks.
        failed_probes: Vec<&'static str>,
    },
    /// An arithmetic status mask has an unsupported dtype.
    ControlMaskDType,
    /// Positive error bounds contain nonfinite values or invalid witnesses.
    InvalidBounds,
    /// A completed calculation contains nonfinite, inverted, or invalid-domain bounds.
    Range {
        /// Caller-supplied calculation stage.
        stage: &'static str,
    },
    /// The underlying Numr operation returned an error.
    Numr(numr::error::Error),
}

impl From<numr::error::Error> for ArithmeticError {
    fn from(error: numr::error::Error) -> Self {
        Self::Numr(error)
    }
}

impl fmt::Display for ArithmeticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDType { dtype } => write!(
                formatter,
                "Interval dtype {dtype:?} is unsupported. Select F32 or F64."
            ),
            Self::ProbeControlDType => formatter.write_str(
                "Arithmetic probe control has an unsupported dtype. Select F32 or F64."
            ),
            Self::InvalidProbeControl { control } => write!(
                formatter,
                "Arithmetic probe control {control:?} requires an integer in 0..=255. Check the runtime arithmetic mode."
            ),
            Self::UnsupportedArithmetic { dtype, failed_probes } => write!(
                formatter,
                "Unsupported interval arithmetic for {dtype:?}: failed_probes={}. Select a declared provider and its admitted arithmetic mode.",
                failed_probes.join(", ")
            ),
            Self::ControlMaskDType => formatter.write_str(
                "Arithmetic status mask has an unsupported dtype. Use F32 or F64 control values."
            ),
            Self::InvalidBounds => formatter.write_str(
                "Positive error bounds contain nonfinite values or invalid witnesses. Rescale the calculation and check its inputs."
            ),
            Self::Range { stage } => write!(
                formatter,
                "Interval arithmetic rejected stage '{stage}': nonfinite or invalid bounds. Rescale inputs or check the operation domain."
            ),
            Self::Numr(error) => write!(formatter, "Numr arithmetic error: {error}"),
        }
    }
}

impl std::error::Error for ArithmeticError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Numr(error) => Some(error),
            _ => None,
        }
    }
}

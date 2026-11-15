//! Map shared arithmetic errors into integration errors.

use crate::common::arithmetic::error::ArithmeticError;
use crate::integrate::error::IntegrateError;

impl From<ArithmeticError> for IntegrateError {
    fn from(error: ArithmeticError) -> Self {
        match error {
            ArithmeticError::UnsupportedDType { dtype } => Self::InvalidParameter {
                parameter: "dtype".into(),
                message: format!("Certified Simpson dtype={dtype:?} requires F32 or F64. Select a supported dtype."),
            },
            ArithmeticError::UnsupportedArithmetic { dtype, failed_probes } => {
                Self::UnsupportedArithmetic { dtype, failed_probes }
            }
            ArithmeticError::Range { stage } => Self::ArithmeticRange { stage },
            ArithmeticError::ProbeControlDType | ArithmeticError::ControlMaskDType => {
                Self::NumericalError {
                    message: "Certified Simpson control dtype requires F32 or F64. Select a supported dtype.".into(),
                }
            }
            ArithmeticError::InvalidProbeControl { control } => Self::NumericalError {
                message: format!("Certified Simpson probe control={control} requires an exact eight-probe mask. Select supported arithmetic."),
            },
            ArithmeticError::InvalidBounds => Self::NumericalError {
                message: "Certified Simpson bounds contain invalid arithmetic. Rescale the interval or callback bounds.".into(),
            },
            ArithmeticError::Numr(error) => Self::from(error),
        }
    }
}

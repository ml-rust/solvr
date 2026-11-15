//! Preserve interpolation errors when shared arithmetic rejects a calculation.

use crate::common::arithmetic::error::ArithmeticError;
use crate::interpolate::error::InterpolateError;

impl From<ArithmeticError> for InterpolateError {
    fn from(error: ArithmeticError) -> Self {
        match error {
            ArithmeticError::UnsupportedDType { dtype } => Self::InvalidParameter {
                parameter: "dtype".into(),
                message: format!("{dtype:?}: Linear receipts require backend-supported F32 or F64"),
            },
            ArithmeticError::ProbeControlDType => Self::NumericalError {
                message: "Linear receipt probe control requires F32 or F64".into(),
            },
            ArithmeticError::InvalidProbeControl { control } => Self::NumericalError {
                message: format!(
                    "Linear receipt probe control={control}: require an exact eight-probe admission mask"
                ),
            },
            ArithmeticError::UnsupportedArithmetic { dtype, failed_probes } => {
                Self::UnsupportedArithmetic { dtype, failed_probes }
            },
            ArithmeticError::ControlMaskDType => Self::NumericalError {
                message: "interp1d control mask requires F32 or F64".into(),
            },
            ArithmeticError::InvalidBounds => Self::NumericalError {
                message: "Linear receipt contains nonfinite arithmetic, negative error, or mismatched TwoSum witnesses. Rescale the grid, values, or queries".into(),
            },
            ArithmeticError::Range { stage } => Self::NumericalError {
                message: format!("Interval arithmetic exceeds finite bounds during {stage}. Rescale the inputs."),
            },
            ArithmeticError::Numr(error) => Self::from(error),
        }
    }
}

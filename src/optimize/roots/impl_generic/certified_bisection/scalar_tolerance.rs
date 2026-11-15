//! Round scalar tolerances inward for the selected numerical dtype.

use crate::DType;
use crate::optimize::{OptimizeError, OptimizeResult};

/// Convert config metadata downward without relaxing the caller's tolerance.
pub(super) fn lower(dtype: DType, value: f64) -> OptimizeResult<f64> {
    if !value.is_finite() || value < 0.0 {
        return Err(OptimizeError::InvalidParameter {
            parameter: "tolerance".into(),
            message: format!("Received {value}. Use a finite nonnegative tolerance."),
        });
    }
    match dtype {
        DType::F64 => Ok(value),
        DType::F32 => {
            let rounded = value as f32;
            let directed = if f64::from(rounded) > value {
                rounded.next_down()
            } else {
                rounded
            };
            Ok(f64::from(directed))
        }
        _ => Err(OptimizeError::InvalidParameter {
            parameter: "dtype".into(),
            message: format!("Received {dtype:?}. Use F32 or F64 for certified roots."),
        }),
    }
}

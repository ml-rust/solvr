use super::cases::{F32_LIMITS, F32_VALUES, F64_LIMITS, F64_VALUES};
use numr::prelude::*;
use solvr::interpolate::{InterpolateError, InterpolateResult};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const LAYOUTS: [usize; 6] = [1, 31, 32, 33, 64, 65];

pub struct Case {
    pub id: &'static str,
    pub dtype: DType,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub query: Vec<f64>,
    pub legacy_error: Option<&'static str>,
    pub metadata: &'static [(usize, usize, bool, bool, bool)],
}

pub fn cases(dtype: DType) -> Vec<Case> {
    if dtype == DType::F32 {
        F32_VALUES
            .iter()
            .chain(F32_LIMITS)
            .map(|case| Case {
                id: case.id,
                dtype,
                x: case.x.iter().copied().map(f64::from).collect(),
                y: case.y.iter().copied().map(f64::from).collect(),
                query: case.query.iter().copied().map(f64::from).collect(),
                legacy_error: case.legacy_error,
                metadata: case.metadata,
            })
            .collect()
    } else {
        F64_VALUES
            .iter()
            .chain(F64_LIMITS)
            .map(|case| Case {
                id: case.id,
                dtype,
                x: case.x.to_vec(),
                y: case.y.to_vec(),
                query: case.query.to_vec(),
                legacy_error: case.legacy_error,
                metadata: case.metadata,
            })
            .collect()
    }
}

pub fn tensor<R: Runtime<DType = DType>>(
    values: &[f64],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Tensor<R>> {
    if dtype == DType::F32 {
        Tensor::from_slice(
            &values.iter().map(|value| *value as f32).collect::<Vec<_>>(),
            shape,
            device,
        )
    } else {
        Tensor::from_slice(values, shape, device)
    }
}

pub fn values<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> numr::error::Result<Vec<f64>> {
    if tensor.dtype() == DType::F32 {
        Ok(tensor
            .try_to_vec::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect())
    } else {
        tensor.try_to_vec::<f64>()
    }
}

pub fn bits(value: f64, dtype: DType) -> String {
    if dtype == DType::F32 {
        format!("{:08x}", (value as f32).to_bits())
    } else {
        format!("{:016x}", value.to_bits())
    }
}

pub fn error_label(error: &InterpolateError) -> &'static str {
    match error {
        InterpolateError::OutOfDomain { .. } => "out_of_domain",
        InterpolateError::NumericalError { .. } => "numerical",
        InterpolateError::NotMonotonic { .. } => "not_monotonic",
        InterpolateError::UnsupportedArithmetic { .. } => "unsupported_arithmetic",
        _ => "other_error",
    }
}

pub fn arithmetic_limit(error: &InterpolateError, dtype: DType) -> InterpolateResult<bool> {
    let InterpolateError::UnsupportedArithmetic {
        dtype: actual,
        failed_probes,
    } = error
    else {
        return Ok(false);
    };
    let known = [
        "provider_contract",
        "nearest_even_positive_ties",
        "nearest_even_negative_ties",
        "gradual_subnormal_inputs",
        "gradual_multiplication_underflow",
        "gradual_division_underflow",
        "two_sum_positive_witness",
        "two_sum_negative_witness",
        "two_sum_cancellation_witness",
    ];
    if *actual != dtype
        || failed_probes.is_empty()
        || failed_probes.iter().any(|probe| !known.contains(probe))
    {
        return Err(InterpolateError::NumericalError {
            message: format!("Malformed arithmetic limitation for {dtype:?}: {error}"),
        });
    }
    Ok(true)
}

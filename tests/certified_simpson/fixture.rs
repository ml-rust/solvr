use numr::prelude::*;
use solvr::integrate::{CertifiedSimpsonOptions, IntegrateError};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Clone, Copy)]
pub enum Function {
    Polynomial {
        coefficients: &'static [u64],
        scale: u64,
        shift: u64,
        offset: u64,
    },
    Sqrt,
}
#[derive(Clone, Copy, PartialEq)]
pub enum Admission {
    Required,
    Either,
    Reject,
}
pub struct Lane {
    pub a: u64,
    pub b: u64,
    pub truth: (u64, u64),
    pub remainder: (u64, u64),
    pub active: bool,
}
pub struct Case {
    pub id: &'static str,
    pub dtype: DType,
    pub panels: usize,
    pub function: Function,
    pub m4: u64,
    pub admission: Admission,
    pub max_points: usize,
    pub function_ceiling: usize,
    pub derivative_ceiling: usize,
    pub width_gate: Option<(f64, f64)>,
    pub lanes: &'static [Lane],
}
impl Case {
    pub fn options(&self) -> CertifiedSimpsonOptions {
        CertifiedSimpsonOptions {
            panels: self.panels,
            max_points: self.max_points,
        }
    }
}
pub fn cases() -> impl Iterator<Item = &'static Case> {
    super::f32_values::CASES
        .iter()
        .chain(super::f32_limits::CASES)
        .chain(super::f64_values::CASES)
        .chain(super::f64_limits::CASES)
        .chain(super::f64_precision::CASES)
}
pub fn decode(bits: u64, dtype: DType) -> f64 {
    if dtype == DType::F32 {
        f64::from(f32::from_bits(bits as u32))
    } else {
        f64::from_bits(bits)
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
            &values.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            shape,
            device,
        )
    } else {
        Tensor::from_slice(values, shape, device)
    }
}
pub fn read<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> numr::error::Result<Vec<f64>> {
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
pub fn read_bits<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> numr::error::Result<Vec<u64>> {
    if tensor.dtype() == DType::F32 {
        Ok(tensor
            .try_to_vec::<f32>()?
            .into_iter()
            .map(|v| u64::from(v.to_bits()))
            .collect())
    } else {
        Ok(tensor
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect())
    }
}
pub fn range(stage: &'static str) -> IntegrateError {
    IntegrateError::ArithmeticRange { stage }
}

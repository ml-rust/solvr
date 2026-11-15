use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps};
use numr::prelude::*;
use solvr::optimize::roots::CertifiedBisectionOptions;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

pub trait Client<R: Runtime<DType = DType>>:
    TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + ScalarOps<R> + RuntimeClient<R>
{
}
impl<R, C> Client<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
}

pub struct Case {
    pub id: &'static str,
    pub dtype: DType,
    pub function: &'static str,
    pub oracle: &'static str,
    pub lower: u64,
    pub upper: u64,
    pub kappa: u64,
    pub root_lower: u64,
    pub root_upper: u64,
    pub atol: u64,
    pub rtol: u64,
    pub max_iterations: usize,
    pub max_points: usize,
    pub status: i64,
    pub certified: bool,
}
impl Case {
    pub fn options(&self) -> CertifiedBisectionOptions {
        CertifiedBisectionOptions {
            atol: f64::from_bits(self.atol),
            rtol: f64::from_bits(self.rtol),
            max_iterations: self.max_iterations,
            max_points: self.max_points,
        }
    }
}
pub fn cases() -> impl Iterator<Item = &'static Case> {
    super::f32_values::CASES
        .iter()
        .chain(super::f64_values::CASES)
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
    match dtype {
        DType::F32 => Tensor::from_slice(
            &values.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            shape,
            device,
        ),
        DType::F64 => Tensor::from_slice(values, shape, device),
        _ => Err(numr::error::Error::UnsupportedDType {
            dtype,
            op: "root fixture",
        }),
    }
}
pub fn read<R: Runtime<DType = DType>>(value: &Tensor<R>) -> numr::error::Result<Vec<f64>> {
    match value.dtype() {
        DType::F32 => Ok(value
            .try_to_vec::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect()),
        DType::F64 => value.try_to_vec::<f64>(),
        dtype => Err(numr::error::Error::UnsupportedDType {
            dtype,
            op: "root test read",
        }),
    }
}
pub fn bits<R, C>(client: &C, value: &Tensor<R>) -> numr::error::Result<Vec<u64>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let owned = Tensor::empty(value.shape(), value.dtype(), client.device())?;
    client.copy_into(&owned, value)?;
    match value.dtype() {
        DType::F32 => Ok(owned
            .try_to_vec::<f32>()?
            .into_iter()
            .map(|x| u64::from(x.to_bits()))
            .collect()),
        DType::F64 => Ok(owned
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect()),
        dtype => Err(numr::error::Error::UnsupportedDType {
            dtype,
            op: "root test bits",
        }),
    }
}
pub fn constant<R, C>(client: &C, like: &Tensor<R>, value: f64) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    Tensor::full_scalar(like.shape(), like.dtype(), value, client.device())
}
pub fn sentinel() -> numr::error::Error {
    numr::error::Error::InvalidArgument {
        arg: "certified root callback",
        reason: "callback sentinel".into(),
    }
}

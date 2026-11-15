use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps};
use numr::prelude::*;
use solvr::optimize::roots::CertifiedRootValues;

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

pub fn tensor<R: Runtime<DType = DType>>(
    values: &[f64],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Tensor<R>> {
    if dtype == DType::F32 {
        Tensor::from_slice(
            &values.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[values.len()],
            device,
        )
    } else {
        Tensor::from_slice(values, &[values.len()], device)
    }
}

pub fn input<R: Runtime<DType = DType>>(
    values: &[f64],
    dtype: DType,
    device: &R::Device,
    strided: bool,
) -> numr::error::Result<Tensor<R>> {
    if !strided {
        return tensor(values, dtype, device);
    }
    let paired = values.iter().flat_map(|&x| [19.0, x]).collect::<Vec<_>>();
    Ok(tensor::<R>(&paired, dtype, device)?
        .reshape(&[values.len(), 2])?
        .narrow(1, 1, 1)?
        .squeeze(Some(1)))
}

pub fn read<R: Runtime<DType = DType>>(value: &Tensor<R>) -> numr::error::Result<Vec<f64>> {
    if value.dtype() == DType::F32 {
        Ok(value
            .try_to_vec::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect())
    } else {
        value.try_to_vec::<f64>()
    }
}

pub fn constant<R, C>(client: &C, like: &Tensor<R>, value: f64) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    Tensor::full_scalar(like.shape(), like.dtype(), value, client.device())
}

pub fn affine<R, C>(
    client: &C,
    x: &Tensor<R>,
    root: &Tensor<R>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let residual = client.sub(x, root)?;
    let exact = client.eq(x, root)?;
    let zero = constant(client, x, 0.0)?;
    // Outward neighbors enclose subtraction even when root and point exponents differ.
    Ok(CertifiedRootValues {
        lower: client.where_cond(&exact, &zero, &client.next_down(&residual)?)?,
        upper: client.where_cond(&exact, &zero, &client.next_up(&residual)?)?,
    })
}

pub fn sentinel() -> numr::error::Error {
    numr::error::Error::InvalidArgument {
        arg: "upper bracket callback",
        reason: "callback sentinel".into(),
    }
}

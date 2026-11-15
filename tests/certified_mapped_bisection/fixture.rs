use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedRootDomainQuery, CertifiedRootMappedValues, CertifiedRootQuery, CertifiedRootResult,
    CertifiedRootValues,
};

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
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Tensor<R>> {
    match dtype {
        DType::F32 => Tensor::from_slice(
            &values.iter().map(|&value| value as f32).collect::<Vec<_>>(),
            shape,
            device,
        ),
        DType::F64 => Tensor::from_slice(values, shape, device),
        _ => Err(numr::error::Error::UnsupportedDType {
            dtype,
            op: "mapped-root fixture",
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
            op: "mapped-root test read",
        }),
    }
}

pub fn bits<R: Runtime<DType = DType>>(value: &Tensor<R>) -> numr::error::Result<Vec<u64>> {
    match value.dtype() {
        DType::F32 => Ok(value
            .try_to_vec::<f32>()?
            .into_iter()
            .map(|x| u64::from(x.to_bits()))
            .collect()),
        DType::F64 => Ok(value
            .try_to_vec::<f64>()?
            .into_iter()
            .map(f64::to_bits)
            .collect()),
        dtype => Err(numr::error::Error::UnsupportedDType {
            dtype,
            op: "mapped-root test bits",
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

pub fn identity<R: Runtime<DType = DType>>(
    query: &CertifiedRootDomainQuery<R>,
) -> numr::error::Result<CertifiedRootMappedValues<R>> {
    Ok(CertifiedRootMappedValues {
        lower: query.lower.clone(),
        upper: query.upper.clone(),
    })
}

pub fn affine<R, C>(
    client: &C,
    point: &Tensor<R>,
    root: f64,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let value = client.fused_mul_add_scalar(point, 1.0, -root)?;
    let zero = constant(client, point, 0.0)?;
    let exact = client.eq(point, &constant(client, point, root)?)?;
    Ok(CertifiedRootValues {
        lower: client.where_cond(&exact, &zero, &client.next_down(&value)?)?,
        upper: client.where_cond(&exact, &zero, &client.next_up(&value)?)?,
    })
}

pub fn quadratic<R, C>(
    client: &C,
    query: &CertifiedRootQuery<R>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let value = client.fused_mul_add(
        &query.point,
        &query.point,
        &constant(client, &query.point, -2.0)?,
    )?;
    Ok(CertifiedRootValues {
        lower: client.next_down(&value)?,
        upper: client.next_up(&value)?,
    })
}

pub fn uncertain_affine<R, C>(
    client: &C,
    query: &CertifiedRootQuery<R>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let at_half = client.eq(&query.point, &constant(client, &query.point, 0.5)?)?;
    let radius = client.where_cond(
        &at_half,
        &constant(client, &query.point, 0.125)?,
        &constant(client, &query.point, 0.25)?,
    )?;
    let value = affine(client, &query.point, 0.375)?;
    Ok(CertifiedRootValues {
        lower: client.next_down(&client.sub(&value.lower, &radius)?)?,
        upper: client.next_up(&client.add(&value.upper, &radius)?)?,
    })
}

#[derive(Debug, PartialEq, Eq)]
pub struct Snapshot {
    pub status: Vec<i64>,
    pub owners: Vec<i64>,
    pub lower: Vec<u64>,
    pub upper: Vec<u64>,
    pub iterations: usize,
    pub points: usize,
}

pub fn snapshot<R: Runtime<DType = DType>>(result: &CertifiedRootResult<R>) -> Result<Snapshot> {
    Ok(Snapshot {
        status: result.status().try_to_vec::<i64>()?,
        owners: result.root_owner().try_to_vec::<i64>()?,
        lower: bits(result.root_lower())?,
        upper: bits(result.root_upper())?,
        iterations: result.iterations(),
        points: result.evaluated_points(),
    })
}

pub fn contains<R: Runtime<DType = DType>>(
    result: &CertifiedRootResult<R>,
    roots: &[f64],
) -> Result {
    let lower = read(result.root_lower())?;
    let upper = read(result.root_upper())?;
    assert_eq!(lower.len(), roots.len());
    for ((lower, upper), root) in lower.into_iter().zip(upper).zip(roots) {
        assert!(
            lower <= *root && *root <= upper,
            "[{lower:e}, {upper:e}] excludes {root:e}"
        );
    }
    Ok(())
}

pub fn sentinel() -> numr::error::Error {
    numr::error::Error::InvalidArgument {
        arg: "mapped root callback",
        reason: "mapped callback sentinel".into(),
    }
}

use super::super::super::certified::native::RootClient;
pub(super) use crate::DType;
pub(super) use crate::optimize::roots::traits::{
    CertifiedBisectionOptions, CertifiedMappedBisectionAlgorithms, CertifiedMappedRootProgress,
    CertifiedRootAdvanceAllowance, CertifiedRootContinuation, CertifiedRootDomainQuery,
    CertifiedRootMappedValues, CertifiedRootQuery, CertifiedRootTolerance, CertifiedRootValues,
};
pub(super) use numr::runtime::{Runtime, RuntimeClient};
pub(super) use numr::tensor::Tensor;

pub(super) type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub(super) trait Client<R: Runtime<DType = DType>>:
    RootClient<R> + CertifiedMappedBisectionAlgorithms<R>
{
}
impl<R: Runtime<DType = DType>, C: RootClient<R> + CertifiedMappedBisectionAlgorithms<R>> Client<R>
    for C
{
}

pub(super) fn tensor<R: Runtime<DType = DType>>(
    values: &[f64],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Tensor<R>> {
    match dtype {
        DType::F32 => Tensor::from_slice(
            &values.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[values.len()],
            device,
        ),
        DType::F64 => Tensor::from_slice(values, &[values.len()], device),
        _ => Err(sentinel()),
    }
}
pub(super) fn roots<R, C>(
    client: &C,
    query: &CertifiedRootQuery<R>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let minus_two = Tensor::full_scalar(
        query.point.shape(),
        query.point.dtype(),
        -2.0,
        client.device(),
    )?;
    let value = client.fused_mul_add(&query.point, &query.point, &minus_two)?;
    Ok(CertifiedRootValues {
        lower: client.next_down(&value)?,
        upper: client.next_up(&value)?,
    })
}
pub(super) fn mapping<R: Runtime<DType = DType>>(
    query: &CertifiedRootDomainQuery<R>,
) -> numr::error::Result<CertifiedRootMappedValues<R>> {
    Ok(CertifiedRootMappedValues {
        lower: query.lower.clone(),
        upper: query.upper.clone(),
    })
}
pub(super) fn options(iterations: usize, points: usize) -> CertifiedBisectionOptions {
    CertifiedBisectionOptions {
        atol: 2.0,
        rtol: 0.0,
        max_iterations: iterations,
        max_points: points,
    }
}
pub(super) fn tolerance() -> CertifiedRootTolerance {
    CertifiedRootTolerance {
        atol: 0.0,
        rtol: 0.0,
    }
}
pub(super) fn allowance(iterations: usize, points: usize) -> CertifiedRootAdvanceAllowance {
    CertifiedRootAdvanceAllowance {
        max_iterations: iterations,
        max_points: points,
    }
}
pub(super) fn begin<R, C>(
    client: &C,
    dtype: DType,
    options: &CertifiedBisectionOptions,
) -> Result<CertifiedMappedRootProgress<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let lower = tensor::<R>(&[1.0, 3.0, 1.0, 1.0], dtype, client.device())?;
    let upper = tensor::<R>(&[2.0, 4.0, 2.0, 2.0], dtype, client.device())?;
    // One is a conservative derivative lower bound for x*x-2 on every supplied domain.
    let derivative = tensor::<R>(&[1.0; 4], dtype, client.device())?;
    Ok(client.bisect_certified_mapped(
        |q| roots(client, q),
        mapping,
        &lower,
        &upper,
        &derivative,
        options,
    )?)
}
pub(super) fn owners<R: Runtime<DType = DType>>(
    client: &impl RuntimeClient<R>,
    values: &[i64],
) -> numr::error::Result<Tensor<R>> {
    Tensor::from_slice(values, &[values.len()], client.device())
}
pub(super) fn guard<R: Runtime<DType = DType>>(
    owners: &Tensor<R>,
    allowed: &[i64],
) -> numr::error::Result<()> {
    for owner in owners.try_to_vec::<i64>()? {
        assert!(
            allowed.contains(&owner),
            "Undemanded callback owner {owner}"
        );
    }
    Ok(())
}
pub(super) fn bits<R: Runtime<DType = DType>>(value: &Tensor<R>) -> numr::error::Result<Vec<u64>> {
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
        _ => Err(sentinel()),
    }
}
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Snapshot {
    pub(super) status: Vec<i64>,
    pub(super) owners: Vec<i64>,
    pub(super) fields: Vec<Vec<u64>>,
    pub(super) counters: [usize; 5],
}
pub(super) fn snapshot<R: Runtime<DType = DType>>(
    c: &CertifiedRootContinuation<R>,
) -> Result<Snapshot> {
    let s = &c.state;
    Ok(Snapshot {
        status: s.status.try_to_vec::<i64>()?,
        owners: s.owner.try_to_vec::<i64>()?,
        fields: [
            &s.lower,
            &s.upper,
            &s.derivative,
            &s.previous_point,
            &s.sampled,
            &s.active,
            &c.floor,
        ]
        .into_iter()
        .map(bits)
        .collect::<numr::error::Result<_>>()?,
        counters: [
            c.iterations,
            s.evaluated,
            c.mapped_points,
            c.max_iterations,
            c.max_points,
        ],
    })
}
pub(super) fn untouched(before: &Snapshot, after: &Snapshot, original: &[usize]) {
    assert_eq!(before.owners, after.owners);
    assert_eq!(before.counters[3..], after.counters[3..]);
    for &owner in original {
        assert_eq!(before.status[owner], after.status[owner]);
        if let Some(index) = before
            .owners
            .iter()
            .position(|&x| usize::try_from(x) == Ok(owner))
        {
            for (old, new) in before.fields.iter().zip(&after.fields) {
                assert_eq!(
                    old[index], new[index],
                    "Private field changed for undemanded owner {owner}"
                );
            }
        }
    }
}
pub(super) fn contains<R: Runtime<DType = DType>>(c: &CertifiedRootContinuation<R>) -> Result {
    let read = |value: &Tensor<R>| -> numr::error::Result<Vec<f64>> {
        if value.dtype() == DType::F32 {
            Ok(value
                .try_to_vec::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect())
        } else {
            value.try_to_vec::<f64>()
        }
    };
    for (lower, upper) in read(&c.state.lower)?.into_iter().zip(read(&c.state.upper)?) {
        assert!(lower <= 2.0_f64.sqrt() && 2.0_f64.sqrt() <= upper);
    }
    Ok(())
}
pub(super) fn sentinel() -> numr::error::Error {
    numr::error::Error::InvalidArgument {
        arg: "selected root callback",
        reason: "Undemanded callback executed".into(),
    }
}

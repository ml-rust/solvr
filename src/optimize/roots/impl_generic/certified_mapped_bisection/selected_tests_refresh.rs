use super::support::*;
use std::cell::Cell;

fn residual<R, C>(
    client: &C,
    query: &CertifiedRootQuery<R>,
    radius: f64,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let one = Tensor::ones(query.point.shape(), query.point.dtype(), client.device())?;
    let value = client.sub(&query.point, &one)?;
    let singleton_owner = Tensor::full_scalar(&[], DType::I64, 2.0, client.device())?;
    let exact = client.eq(&query.owner, &singleton_owner)?;
    let radius = Tensor::full_scalar(
        query.point.shape(),
        query.point.dtype(),
        radius,
        client.device(),
    )?;
    let zero = Tensor::zeros(query.point.shape(), query.point.dtype(), client.device())?;
    let radius = client.where_cond(&exact, &zero, &radius)?;
    Ok(CertifiedRootValues {
        lower: client.sub(&value, &radius)?,
        upper: client.add(&value, &radius)?,
    })
}

fn begin<R, C>(
    client: &C,
    dtype: DType,
    max_iterations: usize,
    max_points: usize,
) -> Result<CertifiedRootContinuation<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let lower = tensor::<R>(&[0.0, 3.0, 1.0, 0.0], dtype, client.device())?;
    let upper = tensor::<R>(&[2.0, 4.0, 1.0, 2.0], dtype, client.device())?;
    let derivative = tensor::<R>(&[1.0; 4], dtype, client.device())?;
    let progress = client.bisect_certified_mapped(
        |query| residual(client, query, 0.125),
        mapping,
        &lower,
        &upper,
        &derivative,
        &CertifiedBisectionOptions {
            atol: 0.0,
            rtol: 0.0,
            max_iterations,
            max_points,
        },
    )?;
    Ok(progress.into_parts().1)
}

pub(super) fn policy<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let continuation = begin(client, dtype, 32, 128)?;
    let initial = snapshot(&continuation)?;
    assert_eq!(initial.status, [3, 1, 0, 3]);
    let index = initial
        .owners
        .iter()
        .position(|owner| *owner == 0)
        .ok_or_else(sentinel)?;
    assert_ne!(initial.fields[0][index], initial.fields[1][index]);
    assert_ne!(initial.fields[4][index], 0);
    assert_ne!(initial.fields[6][index], 0);
    let points = Cell::new(0);
    let ordinary = client.resume_certified_mapped_selected(
        |_| {
            points.set(points.get() + 1);
            Err(sentinel())
        },
        |query| {
            guard(&query.owner, &[0, 2])?;
            mapping(query)
        },
        continuation,
        &owners(client, &[2, 0, 1])?,
        &tolerance(),
        &allowance(32, 128),
    )?;
    assert_eq!(points.get(), 0);
    let (_, continuation) = ordinary.into_parts();
    let before = snapshot(&continuation)?;
    assert_eq!(before.fields, initial.fields);
    assert_eq!(before.status, initial.status);
    assert_eq!(before.counters[..2], initial.counters[..2]);
    let refreshed = client.refresh_certified_mapped_selected(
        |query| {
            guard(&query.owner, &[0])?;
            points.set(points.get() + query.owner.numel());
            residual(client, query, 0.0)
        },
        |query| {
            guard(&query.owner, &[0, 2])?;
            mapping(query)
        },
        continuation,
        &owners(client, &[2, 1, 0])?,
        &tolerance(),
        &allowance(32, 128),
    )?;
    let (_, continuation) = refreshed.into_parts();
    let after = snapshot(&continuation)?;
    untouched(&before, &after, &[1, 2, 3]);
    assert_eq!(after.status[0], 0);
    assert_eq!(after.fields[0][index], after.fields[1][index]);
    let one = tensor::<R>(&[1.0], dtype, client.device())?;
    assert_eq!(after.fields[0][index], bits(&one)?[0]);
    assert_eq!(points.get(), 1);
    assert_eq!(after.counters[0], before.counters[0] + 1);
    assert_eq!(after.counters[1], before.counters[1] + 1);
    assert_eq!(after.counters[3..], before.counters[3..]);
    let empty = client.refresh_certified_mapped_selected(
        |_| Err(sentinel()),
        |_| Err(sentinel()),
        continuation,
        &owners(client, &[])?,
        &tolerance(),
        &allowance(usize::MAX, usize::MAX),
    )?;
    assert_eq!(snapshot(&empty.into_parts().1)?, after);
    Ok(())
}

pub(super) fn exhausted<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    for (max_iterations, max_points) in [(1, 128), (32, 8)] {
        let continuation = begin(client, dtype, max_iterations, max_points)?;
        let before = snapshot(&continuation)?;
        assert_eq!(before.counters[0], 1);
        assert!(before.counters[0] == max_iterations || before.counters[1] == max_points);
        let refreshed = client.refresh_certified_mapped_selected(
            |_| Err(sentinel()),
            |query| {
                guard(&query.owner, &[0])?;
                mapping(query)
            },
            continuation,
            &owners(client, &[0])?,
            &tolerance(),
            &allowance(usize::MAX, usize::MAX),
        )?;
        let (_, continuation) = refreshed.into_parts();
        let after = snapshot(&continuation)?;
        assert_eq!(after.counters[..2], before.counters[..2]);
        assert_eq!(after.counters[3..], before.counters[3..]);
        assert_eq!(after.fields[..3], before.fields[..3]);
        untouched(&before, &after, &[1, 2, 3]);
    }
    Ok(())
}

pub(super) fn admission<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let invalid = [
        owners(client, &[0, 0])?,
        owners(client, &[-1])?,
        owners(client, &[4])?,
        owners(client, &[0])?.reshape(&[1, 1])?,
        tensor::<R>(&[0.0], dtype, client.device())?,
    ];
    for demand in invalid {
        let continuation = begin(client, dtype, 32, 128)?;
        let called = Cell::new(false);
        let result = client.refresh_certified_mapped_selected(
            |_| {
                called.set(true);
                Err(sentinel())
            },
            |_| {
                called.set(true);
                Err(sentinel())
            },
            continuation,
            &demand,
            &tolerance(),
            &allowance(32, 128),
        );
        assert!(result.is_err());
        assert!(!called.get());
    }
    let point = tensor::<R>(&[1.0], dtype, client.device())?;
    let derivative = tensor::<R>(&[1.0], dtype, client.device())?;
    let progress = client.bisect_certified_mapped(
        |query| residual(client, query, 0.125),
        mapping,
        &point,
        &point,
        &derivative,
        &options(8, 16),
    )?;
    let (_, continuation) = progress.into_parts();
    let before = snapshot(&continuation)?;
    assert_eq!(before.status, [2]);
    assert!(before.owners.is_empty());
    let refreshed = client.refresh_certified_mapped_selected(
        |_| Err(sentinel()),
        |_| Err(sentinel()),
        continuation,
        &owners(client, &[0])?,
        &tolerance(),
        &allowance(8, 16),
    )?;
    assert_eq!(snapshot(&refreshed.into_parts().1)?, before);
    Ok(())
}

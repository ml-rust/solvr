use super::support::*;
use std::cell::Cell;

pub(super) fn invalid<R, C>(client: &C, dtype: DType, demand: &Tensor<R>) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let (_, continuation) = begin(client, dtype, &options(16, 64))?.into_parts();
    let called = Cell::new(false);
    let result = client.resume_certified_mapped_selected(
        |_| {
            called.set(true);
            Err(sentinel())
        },
        |_| {
            called.set(true);
            Err(sentinel())
        },
        continuation,
        demand,
        &tolerance(),
        &allowance(8, 16),
    );
    assert!(result.is_err(), "Invalid selected demand accepted");
    assert!(!called.get());
    Ok(())
}
pub(super) fn run<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    for values in [
        &[0_i64, 0][..],
        &[3_i64, 0, 3][..],
        &[-1_i64][..],
        &[4_i64][..],
        &[0_i64, 1, 2, 3, 0][..],
    ] {
        invalid(client, dtype, &owners(client, values)?)?;
    }
    let rank_two = owners(client, &[0])?.reshape(&[1, 1])?;
    invalid(client, dtype, &rank_two)?;
    let wrong_dtype = tensor::<R>(&[0.0], dtype, client.device())?;
    invalid(client, dtype, &wrong_dtype)?;
    // Rejected original rows are valid demand, but supply no compact root callbacks.
    let (_, continuation) = begin(client, dtype, &options(16, 64))?.into_parts();
    let before = snapshot(&continuation)?;
    let progress = client.resume_certified_mapped_selected(
        |_| Err(sentinel()),
        |_| Err(sentinel()),
        continuation,
        &owners(client, &[1])?,
        &tolerance(),
        &allowance(8, 16),
    )?;
    let (_, continuation) = progress.into_parts();
    assert_eq!(snapshot(&continuation)?, before);
    Ok(())
}

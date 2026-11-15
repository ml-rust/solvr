use super::support::{self, Client, Result, enclosed, tensor};
use numr::prelude::*;
use solvr::common::arithmetic::with_interval_arithmetic;

fn input<R: Runtime<DType = DType>>(
    values: &[f64],
    dtype: DType,
    device: &R::Device,
    strided: bool,
) -> numr::error::Result<Tensor<R>> {
    if strided {
        let paired: Vec<_> = values.iter().flat_map(|&value| [value, 91.0]).collect();
        Ok(tensor::<R>(&paired, &[values.len(), 2], dtype, device)?
            .narrow(1, 0, 1)?
            .squeeze(Some(1)))
    } else {
        tensor(values, &[values.len()], dtype, device)
    }
}

pub fn run<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    for strided in [false, true] {
        let lower = input::<R>(&[1.0, 3.0], dtype, client.device(), strided)?;
        let upper = input::<R>(&[2.0, 4.0], dtype, client.device(), strided)?;
        let lower_before = support::input_bits(client, &lower)?;
        let upper_before = support::input_bits(client, &upper)?;
        let captured =
            with_interval_arithmetic::<R, _, _, _>(client, dtype, "owned bounds", |context| {
                assert!(std::ptr::eq(context.client(), client));
                context.bounds(&lower, &upper)
            })?;
        enclosed(&captured, &[1.0, 3.0], &[2.0, 4.0])?;
        assert_eq!(support::input_bits(client, &lower)?, lower_before);
        assert_eq!(support::input_bits(client, &upper)?, upper_before);
        for endpoint in [captured.lower(), captured.upper()] {
            assert!(!endpoint.shares_storage_with(&lower));
            assert!(!endpoint.shares_storage_with(&upper));
        }
        let replacement = tensor::<R>(&[17.0, 17.0], &[2], dtype, client.device())?;
        client.copy_into(&lower, &replacement)?;
        client.copy_into(&upper, &replacement)?;
        enclosed(&captured, &[1.0, 3.0], &[2.0, 4.0])?;

        let points = input::<R>(&[7.0, 11.0], dtype, client.device(), strided)?;
        let doubled =
            with_interval_arithmetic::<R, _, _, _>(client, dtype, "point snapshot", |context| {
                let captured = context.point(&points)?;
                client.copy_into(&points, &replacement)?;
                context.add(&captured, &captured)
            })?;
        enclosed(&doubled, &[14.0, 22.0], &[14.0, 22.0])?;
    }
    let empty = Tensor::<R>::empty(&[2, 0, 3], dtype, client.device())?;
    let captured =
        with_interval_arithmetic::<R, _, _, _>(client, dtype, "empty interval", |context| {
            context.point(&empty)
        })?;
    for endpoint in [captured.lower(), captured.upper()] {
        assert_eq!(endpoint.shape(), &[2, 0, 3]);
        assert_eq!(endpoint.dtype(), dtype);
        assert!(endpoint.device().is_same(client.device()));
        assert!(support::read(endpoint)?.is_empty());
    }
    Ok(())
}

use super::fixture::{self, Client, Result, input, tensor, values};
use numr::prelude::*;
use solvr::common::arithmetic::with_interval_arithmetic;

pub fn run<R, C>(client: &C, dtype: DType) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    for strided in [false, true] {
        let high = input::<R>(&[-0.0, 7.0, 1.0], dtype, client.device(), strided)?;
        let low = input::<R>(&[-0.0, -0.0, 1.0], dtype, client.device(), strided)?;
        let error = input::<R>(&[0.0, 0.0, 0.125], dtype, client.device(), strided)?;
        let high_before = fixture::bits(client, &high)?;
        let low_before = fixture::bits(client, &low)?;
        let error_before = fixture::bits(client, &error)?;
        let (captured, point) = with_interval_arithmetic::<R, _, _, _>(
            client,
            dtype,
            "owned expansion components",
            |context| {
                assert!(std::ptr::eq(context.client(), client));
                Ok((
                    context.expansion_bounds(&high, &low, &error)?,
                    context.expansion_point(&high)?,
                ))
            },
        )?;
        let captured_before = [
            fixture::bits(client, captured.high())?,
            fixture::bits(client, captured.low())?,
            fixture::bits(client, captured.abs_error())?,
        ];
        assert_eq!(&captured_before[0][..2], &high_before[..2]);
        assert_eq!(&captured_before[1][..2], &low_before[..2]);
        assert_eq!(values(client, captured.high())?[2], 2.0);
        assert_eq!(values(client, captured.low())?[2], 0.0);
        assert_eq!(captured_before[2], error_before);
        assert_eq!(fixture::bits(client, point.high())?, high_before);
        assert_eq!(fixture::bits(client, &high)?, high_before);
        assert_eq!(fixture::bits(client, &low)?, low_before);
        for result in [&captured, &point] {
            for component in [result.high(), result.low(), result.abs_error()] {
                assert!(component.is_contiguous());
                assert_eq!(component.shape(), &[3]);
                assert_eq!(component.dtype(), dtype);
                assert!(component.device().is_same(client.device()));
                for source in [&high, &low, &error] {
                    assert!(!component.shares_storage_with(source));
                }
            }
        }
        let replacement = tensor::<R>(&[17.0; 3], &[3], dtype, client.device())?;
        for source in [&high, &low, &error] {
            client.copy_into(source, &replacement)?;
        }
        for (component, expected) in [captured.high(), captured.low(), captured.abs_error()]
            .into_iter()
            .zip(&captured_before)
        {
            assert_eq!(&fixture::bits(client, component)?, expected);
        }
        assert_eq!(fixture::bits(client, point.high())?, high_before);
    }
    Ok(())
}

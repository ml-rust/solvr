use super::admission::ones;
use super::callback::Client;
use super::fixture::{self, Result};
use numr::prelude::*;
use solvr::integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions};
use std::cell::RefCell;

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        let left = fixture::tensor::<R>(&[0.0], &[1], dtype, client.device())?;
        let right = fixture::tensor::<R>(&[1.0], &[1], dtype, client.device())?;
        let retained = RefCell::new(Vec::new());
        let receipt = client.certified_simpson(
            &left,
            &right,
            CertifiedSimpsonOptions {
                panels: 8,
                max_points: 32,
            },
            |query| {
                let output = ones(client, query)?;
                retained.borrow_mut().extend([
                    output.value.clone(),
                    output.lower.clone(),
                    output.upper.clone(),
                ]);
                Ok(output)
            },
            |query| {
                // Function parameters stay fixed. Only aliases of returned callback storage change.
                for alias in retained.borrow().iter() {
                    let replacement =
                        Tensor::full_scalar(alias.shape(), dtype, 7.0, client.device())?;
                    client.copy_into(alias, &replacement)?;
                }
                Ok(Tensor::zeros(query.lower.shape(), dtype, client.device())?)
            },
        )?;
        let value = fixture::read(receipt.value())?[0];
        let lower = fixture::read(receipt.lower())?[0];
        let upper = fixture::read(receipt.upper())?[0];
        assert!(
            lower <= 1.0 && upper >= 1.0,
            "callback alias changed mathematical receipt"
        );
        assert!((value - 1.0).abs() < 1e-5 && upper - lower < 1e-4);
        assert!(upper < 2.0);
        let width = if dtype == DType::F32 {
            f64::from(f32::from_bits(0x4e800000))
        } else {
            f64::from_bits(0x52b0000000000000)
        };
        let wide_right = fixture::tensor::<R>(&[width], &[1], dtype, client.device())?;
        let wide = client.certified_simpson(
            &left,
            &wide_right,
            CertifiedSimpsonOptions {
                panels: 8,
                max_points: 32,
            },
            |query| ones(client, query),
            |query| Ok(Tensor::zeros(query.lower.shape(), dtype, client.device())?),
        )?;
        assert!(
            fixture::read(wide.lower())?[0] <= width && fixture::read(wide.upper())?[0] >= width
        );
        assert_eq!(fixture::read(wide.quadrature_error_bound())?[0], 0.0);
        let before = fixture::read_bits(receipt.value())?;
        let replacement = Tensor::zeros(left.shape(), dtype, client.device())?;
        for alias in retained.borrow().iter() {
            let zeros = Tensor::zeros(alias.shape(), dtype, client.device())?;
            client.copy_into(alias, &zeros)?;
        }
        client.copy_into(&right, &replacement)?;
        assert_eq!(fixture::read_bits(receipt.value())?, before);
    }
    Ok(())
}

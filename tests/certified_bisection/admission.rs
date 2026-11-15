use super::fixture::{self, Client, Result, constant, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedBisectionAlgorithms, CertifiedBisectionOptions, CertifiedRootValues,
};
use std::cell::Cell;

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedBisectionAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        let zero = tensor::<R>(&[0.0], &[1], dtype, client.device())?;
        let one = tensor::<R>(&[1.0], &[1], dtype, client.device())?;
        let options = CertifiedBisectionOptions::default();
        for (lo, hi, derivative) in [
            (f64::NAN, 1.0, 1.0),
            (0.0, f64::INFINITY, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, -1.0),
            (0.0, 1.0, f64::NAN),
            (0.0, 1.0, f64::INFINITY),
        ] {
            let called = Cell::new(false);
            let result = client.bisect_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &tensor(&[lo], &[1], dtype, client.device())?,
                &tensor(&[hi], &[1], dtype, client.device())?,
                &tensor(&[derivative], &[1], dtype, client.device())?,
                &options,
            );
            assert!(result.is_err());
            assert!(!called.get());
        }
        let other = if dtype == DType::F32 {
            DType::F64
        } else {
            DType::F32
        };
        let wrong = [
            tensor::<R>(&[1.0], &[1, 1], dtype, client.device())?,
            tensor::<R>(&[1.0, 1.0], &[2], dtype, client.device())?,
            tensor::<R>(&[1.0], &[1], other, client.device())?,
        ];
        for value in &wrong {
            let called = Cell::new(false);
            let result = client.bisect_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &zero,
                value,
                &one,
                &options,
            );
            assert!(result.is_err());
            assert!(!called.get());
        }
        for malformed in 0..8 {
            let called = Cell::new(0usize);
            let result = client.bisect_certified(
                |query| {
                    called.set(called.get() + 1);
                    let value = constant(client, &query.point, -1.0)?;
                    let upper = match malformed {
                        0 => Tensor::empty(&[query.point.numel(), 1], dtype, client.device())?,
                        1 => Tensor::empty(&[query.point.numel() + 1], dtype, client.device())?,
                        2 => Tensor::empty(query.point.shape(), other, client.device())?,
                        3 => constant(client, &query.point, f64::NAN)?,
                        4 => constant(client, &query.point, f64::INFINITY)?,
                        5 => constant(client, &query.point, -2.0)?,
                        6 => return Err(fixture::sentinel()),
                        _ => {
                            return Ok(CertifiedRootValues {
                                lower: constant(client, &query.point, f64::NEG_INFINITY)?,
                                upper: value,
                            });
                        }
                    };
                    Ok(CertifiedRootValues {
                        lower: value,
                        upper,
                    })
                },
                &zero,
                &one,
                &one,
                &options,
            );
            assert!(result.is_err(), "malformed callback {malformed} accepted");
            assert!(called.get() > 0);
            if malformed == 6 {
                match result {
                    Err(error) => assert!(error.to_string().contains("callback sentinel")),
                    Ok(_) => return Err("callback error accepted".into()),
                }
            }
        }
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            for relative in [false, true] {
                let mut invalid_options = options.clone();
                if relative {
                    invalid_options.rtol = invalid;
                } else {
                    invalid_options.atol = invalid;
                }
                let called = Cell::new(false);
                assert!(
                    client
                        .bisect_certified(
                            |_| {
                                called.set(true);
                                Err(fixture::sentinel())
                            },
                            &zero,
                            &one,
                            &one,
                            &invalid_options
                        )
                        .is_err()
                );
                assert!(!called.get());
            }
        }
        let empty = Tensor::<R>::empty(&[0], dtype, client.device())?;
        let called = Cell::new(false);
        let result = client.bisect_certified(
            |_| {
                called.set(true);
                Err(fixture::sentinel())
            },
            &empty,
            &empty,
            &empty,
            &CertifiedBisectionOptions {
                max_points: 0,
                ..options.clone()
            },
        )?;
        assert!(!called.get());
        assert_eq!(result.status().shape(), &[0]);
        assert_eq!(result.root_owner().shape(), &[0]);
        assert_eq!(result.root_lower().shape(), &[0]);
        assert_eq!(result.root_upper().shape(), &[0]);
        assert_eq!((result.iterations(), result.evaluated_points()), (0, 0));
    }
    Ok(())
}

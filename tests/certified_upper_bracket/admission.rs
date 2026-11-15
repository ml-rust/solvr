use super::fixture::{self, Client, Result, constant, tensor};
use numr::prelude::*;
use solvr::optimize::roots::{
    CertifiedRootValues, CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions,
};
use std::cell::Cell;

pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedUpperBracketAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        let zero = tensor::<R>(&[0.0], dtype, client.device())?;
        let one = tensor::<R>(&[1.0], dtype, client.device())?;
        let two = tensor::<R>(&[2.0], dtype, client.device())?;
        let options = CertifiedUpperBracketOptions::default();
        if usize::BITS > 63 {
            let called = Cell::new(false);
            let result = client.bracket_upper_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &zero,
                &one,
                &two,
                &CertifiedUpperBracketOptions {
                    max_expansions: usize::MAX,
                    ..Default::default()
                },
            );
            assert!(result.is_err());
            assert!(!called.get());
        }
        for (lo, seed, cap) in [
            (f64::NAN, 1.0, 2.0),
            (0.0, f64::NAN, 2.0),
            (0.0, 1.0, f64::NAN),
            (f64::NEG_INFINITY, 1.0, 2.0),
            (0.0, f64::INFINITY, 2.0),
            (0.0, 1.0, f64::INFINITY),
            (2.0, 1.0, 3.0),
            (0.0, 2.0, 1.0),
        ] {
            let called = Cell::new(false);
            let result = client.bracket_upper_certified(
                |_| {
                    called.set(true);
                    Err(fixture::sentinel())
                },
                &tensor(&[lo], dtype, client.device())?,
                &tensor(&[seed], dtype, client.device())?,
                &tensor(&[cap], dtype, client.device())?,
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
            one.reshape(&[1, 1])?,
            tensor::<R>(&[1.0, 1.0], dtype, client.device())?,
            tensor::<R>(&[1.0], other, client.device())?,
            Tensor::<R>::from_slice(&[1i64], &[1], client.device())?,
        ];
        for malformed in &wrong {
            for position in 0..3 {
                let args = [&zero, &one, &two];
                let called = Cell::new(false);
                let result = client.bracket_upper_certified(
                    |_| {
                        called.set(true);
                        Err(fixture::sentinel())
                    },
                    if position == 0 { malformed } else { args[0] },
                    if position == 1 { malformed } else { args[1] },
                    if position == 2 { malformed } else { args[2] },
                    &options,
                );
                assert!(result.is_err());
                assert!(!called.get());
            }
        }
        for malformed in 0..9 {
            let called = Cell::new(false);
            let result = client.bracket_upper_certified(
                |query| {
                    called.set(true);
                    let lower = constant(client, &query.point, -1.0)?;
                    let upper = match malformed {
                        0 => Tensor::empty(&[query.point.numel(), 1], dtype, client.device())?,
                        1 => Tensor::empty(&[query.point.numel() + 1], dtype, client.device())?,
                        2 => Tensor::empty(query.point.shape(), other, client.device())?,
                        3 => constant(client, &query.point, f64::NAN)?,
                        4 => constant(client, &query.point, f64::INFINITY)?,
                        5 => constant(client, &query.point, -2.0)?,
                        6 => return Err(fixture::sentinel()),
                        7 => {
                            return Ok(CertifiedRootValues {
                                lower: constant(client, &query.point, f64::NEG_INFINITY)?,
                                upper: lower,
                            });
                        }
                        _ => {
                            return Ok(CertifiedRootValues {
                                lower: lower.reshape(&[query.point.numel(), 1])?,
                                upper: lower,
                            });
                        }
                    };
                    Ok(CertifiedRootValues { lower, upper })
                },
                &zero,
                &one,
                &two,
                &options,
            );
            assert!(called.get());
            assert!(result.is_err(), "malformed callback {malformed} accepted");
            if malformed == 6 {
                match result {
                    Err(error) => assert!(error.to_string().contains("callback sentinel")),
                    Ok(_) => return Err("callback sentinel accepted".into()),
                }
            }
        }
        let empty = Tensor::<R>::empty(&[0], dtype, client.device())?;
        let called = Cell::new(false);
        let result = client.bracket_upper_certified(
            |_| {
                called.set(true);
                Err(fixture::sentinel())
            },
            &empty,
            &empty,
            &empty,
            &CertifiedUpperBracketOptions {
                max_points: 0,
                max_expansions: 0,
            },
        )?;
        assert!(!called.get());
        assert_eq!(result.callback_points(), 0);
        assert_eq!(result.expansion_rounds(), 0);
        for value in [
            result.lower(),
            result.upper(),
            result.status(),
            result.expansions(),
            &result.lower_residual().lower,
            &result.lower_residual().upper,
            &result.upper_residual().lower,
            &result.upper_residual().upper,
        ] {
            assert_eq!(value.shape(), &[0]);
        }
    }
    Ok(())
}

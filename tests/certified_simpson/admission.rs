use super::callback::Client;
use super::fixture::{self, Result};
use numr::prelude::*;
use solvr::integrate::{
    CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, CertifiedSimpsonQuery,
    CertifiedSimpsonValues, IntegrateError, IntegrateResult,
};

pub fn ones<R, C>(
    client: &C,
    query: &CertifiedSimpsonQuery<R>,
) -> IntegrateResult<CertifiedSimpsonValues<R>>
where
    R: Runtime<DType = DType>,
    C: Client<R>,
{
    let value = Tensor::ones(query.point.shape(), query.point.dtype(), client.device())?;
    Ok(CertifiedSimpsonValues {
        lower: value.clone(),
        upper: value.clone(),
        value,
    })
}
fn input_error(error: &IntegrateError) -> bool {
    matches!(
        error,
        IntegrateError::InvalidInput { .. } | IntegrateError::InvalidParameter { .. }
    )
}
pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        let a = fixture::tensor::<R>(&[0.0], &[1], dtype, client.device())?;
        let b = fixture::tensor::<R>(&[1.0], &[1], dtype, client.device())?;
        let normal = CertifiedSimpsonOptions {
            panels: 8,
            max_points: 32,
        };
        for options in [
            CertifiedSimpsonOptions {
                panels: 0,
                max_points: 32,
            },
            CertifiedSimpsonOptions {
                panels: 3,
                max_points: 32,
            },
            CertifiedSimpsonOptions {
                panels: 8,
                max_points: 0,
            },
            CertifiedSimpsonOptions {
                panels: 8,
                max_points: 24,
            },
            CertifiedSimpsonOptions {
                panels: 1usize << (usize::BITS - 1),
                max_points: usize::MAX,
            },
        ] {
            let error = client
                .certified_simpson(
                    &a,
                    &b,
                    options,
                    |_| panic!("invalid config reaches value callback"),
                    |_| panic!("invalid config reaches derivative callback"),
                )
                .err()
                .expect("invalid config accepted");
            assert!(input_error(&error), "{error}");
        }
        let wrong_shape = b.reshape(&[1, 1])?;
        let mixed = client.cast(
            &b,
            if dtype == DType::F32 {
                DType::F64
            } else {
                DType::F32
            },
        )?;
        let reversed = fixture::tensor::<R>(&[-1.0], &[1], dtype, client.device())?;
        let nan = fixture::tensor::<R>(&[f64::NAN], &[1], dtype, client.device())?;
        let infinity = fixture::tensor::<R>(&[f64::INFINITY], &[1], dtype, client.device())?;
        for right in [&wrong_shape, &mixed, &reversed, &nan, &infinity] {
            assert!(
                client
                    .certified_simpson(
                        &a,
                        right,
                        normal,
                        |_| panic!("invalid endpoints reach value callback"),
                        |_| panic!("invalid endpoints reach derivative callback")
                    )
                    .is_err()
            );
        }
        for kind in 0..10 {
            let result = client.certified_simpson(
                &a,
                &b,
                normal,
                |query| {
                    let mut output = ones(client, query)?;
                    match kind {
                        0 => {
                            output.value =
                                Tensor::ones(&[query.point.numel() + 1], dtype, client.device())?
                        }
                        1 => {
                            output.value = client.cast(
                                &output.value,
                                if dtype == DType::F32 {
                                    DType::F64
                                } else {
                                    DType::F32
                                },
                            )?
                        }
                        2 => {
                            output.value = Tensor::full_scalar(
                                query.point.shape(),
                                dtype,
                                f64::NAN,
                                client.device(),
                            )?
                        }
                        3 => {
                            output.lower = Tensor::full_scalar(
                                query.point.shape(),
                                dtype,
                                2.0,
                                client.device(),
                            )?
                        }
                        4 => {
                            output.value =
                                Tensor::zeros(query.point.shape(), dtype, client.device())?
                        }
                        5 => {
                            output.upper = Tensor::full_scalar(
                                query.point.shape(),
                                dtype,
                                f64::INFINITY,
                                client.device(),
                            )?
                        }
                        _ => {}
                    }
                    Ok(output)
                },
                |query| {
                    let shape = if kind == 8 {
                        vec![query.lower.numel() + 1]
                    } else {
                        query.lower.shape().to_vec()
                    };
                    let value = match kind {
                        6 => -1.0,
                        7 => f64::NAN,
                        _ => 0.0,
                    };
                    let output = Tensor::full_scalar(&shape, dtype, value, client.device())?;
                    Ok(if kind == 9 {
                        client.cast(
                            &output,
                            if dtype == DType::F32 {
                                DType::F64
                            } else {
                                DType::F32
                            },
                        )?
                    } else {
                        output
                    })
                },
            );
            let error = result.err().expect("malformed callback accepted");
            assert!(
                input_error(&error),
                "callback kind{kind} wrong rejection for {dtype:?}: {error}"
            );
        }
        let mut derivative_calls = 0;
        let error = client
            .certified_simpson(
                &a,
                &b,
                normal,
                |_| {
                    Err(IntegrateError::InvalidParameter {
                        parameter: "value sentinel".into(),
                        message: "preserve payload".into(),
                    })
                },
                |_| {
                    derivative_calls += 1;
                    panic!("value error reaches derivative callback")
                },
            )
            .err()
            .expect("value error swallowed");
        assert_eq!(derivative_calls, 0);
        assert!(
            matches!(error, IntegrateError::InvalidParameter { parameter, message } if parameter == "value sentinel" && message == "preserve payload")
        );
        let error = client
            .certified_simpson(
                &a,
                &b,
                normal,
                |query| ones(client, query),
                |_| {
                    Err(IntegrateError::InvalidInput {
                        context: "derivative sentinel".into(),
                    })
                },
            )
            .err()
            .expect("derivative error swallowed");
        assert!(
            matches!(error, IntegrateError::InvalidInput { context } if context == "derivative sentinel")
        );
    }
    let input = Tensor::<R>::from_slice(&[0i64], &[1], client.device())?;
    assert!(
        client
            .certified_simpson(
                &input,
                &input,
                CertifiedSimpsonOptions {
                    panels: 1,
                    max_points: 1
                },
                |_| panic!("integer input reaches callback"),
                |_| panic!("integer input reaches derivative")
            )
            .is_err()
    );
    Ok(())
}

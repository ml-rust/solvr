//! Shared exact public-API observations for CPU and GPU execution.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::InterpolateResult;
use solvr::interpolate::{Interp1dAlgorithms, InterpMethod};

pub fn tensor<R: Runtime<DType = DType>>(
    values: &[f64],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Tensor<R>> {
    if dtype == DType::F32 {
        let values: Vec<f32> = values.iter().map(|v| *v as f32).collect();
        Tensor::from_slice(&values, shape, device)
    } else {
        Tensor::from_slice(values, shape, device)
    }
}

pub fn values<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> numr::error::Result<Vec<f64>> {
    if tensor.dtype() == DType::F32 {
        Ok(tensor
            .try_to_vec::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect())
    } else {
        tensor.try_to_vec::<f64>()
    }
}

pub fn observations<R, C>(
    client: &C,
    device: &R::Device,
    dtype: DType,
) -> InterpolateResult<Vec<f64>>
where
    R: Runtime<DType = DType>,
    C: Interp1dAlgorithms<R>,
{
    let make = |data: &[f64], shape: &[usize]| tensor(data, shape, dtype, device);
    let mut observed = Vec::new();
    let x = make(&[0.0, 1e-14], &[2])?;
    let y = make(&[0.0, 1.0], &[2])?;
    let q = make(&[0.0, 0.5e-14, 1e-14, 2e-14], &[4])?;
    let output = client.interp1d(&x, &y, &q, InterpMethod::Linear)?;
    assert_eq!(output.dtype(), dtype);
    let actual = values(&output)?;
    assert_eq!(actual, vec![0.0, 0.5, 1.0, 2.0]);
    observed.extend(actual);
    let x = make(&[0.0, 1.0, 3.0], &[3])?;
    let y = make(&[2.0, 4.0, 8.0], &[3])?;
    let q = make(&[-1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0], &[7])?;
    let output = client.interp1d(&x, &y, &q, InterpMethod::Linear)?;
    let actual = values(&output)?;
    assert_eq!(actual, vec![0.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0]);
    observed.extend(actual);
    let grid = make(&[99.0, 0.0, 99.0, 1.0, 99.0, 3.0], &[3, 2])?;
    let ordinate = make(&[99.0, 2.0, 99.0, 4.0, 99.0, 8.0], &[3, 2])?;
    let query = make(&[99.0, 0.0, 99.0, 0.5, 99.0, 3.0], &[3, 2])?;
    let grid = grid.narrow(1, 1, 1)?.squeeze(Some(1));
    let ordinate = ordinate.narrow(1, 1, 1)?.squeeze(Some(1));
    let query = query.narrow(1, 1, 1)?.squeeze(Some(1));
    assert!(!grid.is_contiguous());
    let output = client.interp1d(&grid, &ordinate, &query, InterpMethod::Linear)?;
    let actual = values(&output)?;
    assert_eq!(actual, vec![2.0, 3.0, 8.0]);
    observed.extend(actual);
    let (small, large) = if dtype == DType::F32 {
        (1e-30, f64::from(f32::MAX))
    } else {
        (1e-300, f64::MAX)
    };
    let axis = make(&[0.0, small, 1.0], &[3])?;
    let ordinate = make(&[7.0, 7.0, 9.0], &[3])?;
    let query = make(&[-1e20, 0.5, 1.0], &[3])?;
    let output = client.interp1d(&axis, &ordinate, &query, InterpMethod::Linear)?;
    let actual = values(&output)?;
    assert_eq!(actual, vec![7.0, 8.0, 9.0]);
    observed.extend(actual);
    let axis = make(&[-1.0, 0.0, large], &[3])?;
    let ordinate = make(&[5.0, 7.0, 7.0], &[3])?;
    let query = make(&[small, -0.5, large], &[3])?;
    let output = client.interp1d(&axis, &ordinate, &query, InterpMethod::Linear)?;
    let actual = values(&output)?;
    assert_eq!(actual, vec![7.0, 6.0, 7.0]);
    observed.extend(actual);
    let empty = Tensor::<R>::empty(&[0], dtype, device)?;
    let output = client.interp1d(&x, &y, &empty, InterpMethod::Linear)?;
    assert_eq!(output.shape(), &[0]);
    assert_eq!(output.dtype(), dtype);
    assert!(values(&output)?.is_empty());
    Ok(observed)
}

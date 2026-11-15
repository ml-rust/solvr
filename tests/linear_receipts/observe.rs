use super::fixture::{self, Case};
use numr::prelude::*;
use solvr::interpolate::{
    Interp1dAlgorithms, InterpMethod, InterpolateError, InterpolateResult,
    LinearInterp1dDiagnostics, LinearInterp1dReceipt,
};

pub struct Projected {
    pub values: Vec<f64>,
    pub absolute_error: Vec<f64>,
    pub lower: Vec<i64>,
    pub upper: Vec<i64>,
    pub at_lower: Vec<f64>,
    pub at_upper: Vec<f64>,
    pub constant: Vec<f64>,
}

pub struct Observation {
    pub source_queries: Vec<usize>,
    pub legacy: InterpolateResult<Vec<f64>>,
    pub diagnostic: InterpolateResult<Projected>,
}

fn indices<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> InterpolateResult<Vec<i64>> {
    match tensor.dtype() {
        DType::I64 => Ok(tensor.try_to_vec::<i64>()?),
        DType::I32 => Ok(tensor
            .try_to_vec::<i32>()?
            .into_iter()
            .map(i64::from)
            .collect()),
        dtype => Err(InterpolateError::NumericalError {
            message: format!("Unexpected Linear receipt bracket dtype {dtype:?}"),
        }),
    }
}

pub fn project<R: Runtime<DType = DType>>(
    receipt: &LinearInterp1dReceipt<R>,
    length: usize,
    dtype: DType,
) -> InterpolateResult<Projected> {
    for tensor in [
        receipt.values(),
        receipt.absolute_error(),
        receipt.lower_indices(),
        receipt.upper_indices(),
        receipt.at_lower(),
        receipt.at_upper(),
        receipt.constant(),
    ] {
        if tensor.shape() != [length] {
            return Err(InterpolateError::NumericalError {
                message: format!("Receipt shape={:?}, expected=[{length}]", tensor.shape()),
            });
        }
    }
    for tensor in [
        receipt.values(),
        receipt.absolute_error(),
        receipt.at_lower(),
        receipt.at_upper(),
        receipt.constant(),
    ] {
        if tensor.dtype() != dtype {
            return Err(InterpolateError::NumericalError {
                message: format!("Receipt dtype={:?}, expected={dtype:?}", tensor.dtype()),
            });
        }
    }
    Ok(Projected {
        values: fixture::values(receipt.values())?,
        absolute_error: fixture::values(receipt.absolute_error())?,
        lower: indices(receipt.lower_indices())?,
        upper: indices(receipt.upper_indices())?,
        at_lower: fixture::values(receipt.at_lower())?,
        at_upper: fixture::values(receipt.at_upper())?,
        constant: fixture::values(receipt.constant())?,
    })
}

pub fn run<R, C>(
    client: &C,
    device: &R::Device,
    case: &Case,
    layout: usize,
) -> InterpolateResult<Observation>
where
    R: Runtime<DType = DType>,
    C: Interp1dAlgorithms<R> + LinearInterp1dDiagnostics<R>,
{
    let source_queries = if case.query.is_empty() {
        Vec::new()
    } else {
        (0..layout)
            .map(|lane| (lane + layout) % case.query.len())
            .collect::<Vec<_>>()
    };
    let query = source_queries
        .iter()
        .map(|index| case.query[*index])
        .collect::<Vec<_>>();
    let x = fixture::tensor(&case.x, &[case.x.len()], case.dtype, device)?;
    let y = fixture::tensor(&case.y, &[case.y.len()], case.dtype, device)?;
    let q = fixture::tensor(&query, &[query.len()], case.dtype, device)?;
    let legacy = client
        .interp1d(&x, &y, &q, InterpMethod::Linear)
        .and_then(|tensor| Ok(fixture::values(&tensor)?));
    let diagnostic = client
        .interp1d_linear_with_receipts(&x, &y, &q)
        .and_then(|receipt| project(&receipt, query.len(), case.dtype));
    Ok(Observation {
        source_queries,
        legacy,
        diagnostic,
    })
}

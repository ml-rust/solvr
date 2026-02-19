//! Structuring element generation.

use crate::morphology::traits::binary::StructuringElement;
use numr::dtype::DType;
use numr::error::Result;
use numr::ops::TypeConversionOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Generate a structuring element tensor for the given ndim.
///
/// - Cross: only immediate neighbors along each axis (connectivity=1)
/// - Full: all neighbors including diagonals (connectivity=ndim)
pub fn generate_structuring_element<R, C>(
    client: &C,
    kind: StructuringElement,
    ndim: usize,
    dtype: DType,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TypeConversionOps<R> + RuntimeClient<R>,
{
    let size = 3usize; // 3x3x...x3
    let total = size.pow(ndim as u32);
    let shape: Vec<usize> = vec![3; ndim];

    let device = client.device();
    match kind {
        StructuringElement::Full => {
            // All ones
            let data = vec![1.0f64; total];
            let tensor = Tensor::from_slice(&data, &shape, device);
            client.cast(&tensor, dtype)
        }
        StructuringElement::Cross => {
            // Only center and axis-aligned neighbors
            let mut data = vec![0.0f64; total];
            let center = total / 2;
            data[center] = 1.0;
            // For each axis, set the two neighbors
            for ax in 0..ndim {
                let stride: usize = size.pow((ndim - 1 - ax) as u32);
                data[center - stride] = 1.0;
                data[center + stride] = 1.0;
            }
            let tensor = Tensor::from_slice(&data, &shape, device);
            client.cast(&tensor, dtype)
        }
    }
}

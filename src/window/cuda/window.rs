//! CUDA backend implementation for window functions.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
use numr::tensor::Tensor;

use crate::window::impl_generic::{
    generate_blackman_f64, generate_hamming_f64, generate_hann_f64, generate_kaiser_f64,
};
use crate::window::traits::{WindowFunctions, validate_window_dtype, validate_window_size};

impl WindowFunctions<CudaRuntime> for CudaClient {
    fn hann_window(
        &self,
        size: usize,
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_window_size(size, "hann_window")?;
        validate_window_dtype(dtype, "hann_window")?;
        let values = generate_hann_f64(size);
        create_window_tensor_cuda(values, dtype, device)
    }

    fn hamming_window(
        &self,
        size: usize,
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_window_size(size, "hamming_window")?;
        validate_window_dtype(dtype, "hamming_window")?;
        let values = generate_hamming_f64(size);
        create_window_tensor_cuda(values, dtype, device)
    }

    fn blackman_window(
        &self,
        size: usize,
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_window_size(size, "blackman_window")?;
        validate_window_dtype(dtype, "blackman_window")?;
        let values = generate_blackman_f64(size);
        create_window_tensor_cuda(values, dtype, device)
    }

    fn kaiser_window(
        &self,
        size: usize,
        beta: f64,
        dtype: DType,
        device: &CudaDevice,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_window_size(size, "kaiser_window")?;
        validate_window_dtype(dtype, "kaiser_window")?;
        let values = generate_kaiser_f64(size, beta);
        create_window_tensor_cuda(values, dtype, device)
    }
}

fn create_window_tensor_cuda(
    values: Vec<f64>,
    dtype: DType,
    device: &CudaDevice,
) -> Result<Tensor<CudaRuntime>> {
    let size = values.len();
    match dtype {
        DType::F32 => {
            let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            Ok(Tensor::<CudaRuntime>::from_slice(
                &values_f32,
                &[size],
                device,
            ))
        }
        DType::F64 => Ok(Tensor::<CudaRuntime>::from_slice(&values, &[size], device)),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "window",
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::dtype::DType;

    #[test]
    #[ignore] // CUDA tests require GPU
    fn test_hann_window_cuda() {
        let device = CudaDevice::new(0).expect("CUDA device not available");
        let client = CudaClient::new(device.clone());
        let window = client
            .hann_window(8, DType::F64, &device)
            .expect("hann_window failed");
        assert_eq!(window.shape(), &[8]);
    }
}

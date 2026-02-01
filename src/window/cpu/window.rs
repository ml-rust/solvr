//! CPU backend implementation for window functions.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

use crate::window::impl_generic::{
    generate_blackman_f64, generate_hamming_f64, generate_hann_f64, generate_kaiser_f64,
};
use crate::window::traits::{WindowFunctions, validate_window_dtype, validate_window_size};

impl WindowFunctions<CpuRuntime> for CpuClient {
    fn hann_window(
        &self,
        size: usize,
        dtype: DType,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_window_size(size, "hann_window")?;
        validate_window_dtype(dtype, "hann_window")?;
        let values = generate_hann_f64(size);
        create_window_tensor(values, dtype, device)
    }

    fn hamming_window(
        &self,
        size: usize,
        dtype: DType,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_window_size(size, "hamming_window")?;
        validate_window_dtype(dtype, "hamming_window")?;
        let values = generate_hamming_f64(size);
        create_window_tensor(values, dtype, device)
    }

    fn blackman_window(
        &self,
        size: usize,
        dtype: DType,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_window_size(size, "blackman_window")?;
        validate_window_dtype(dtype, "blackman_window")?;
        let values = generate_blackman_f64(size);
        create_window_tensor(values, dtype, device)
    }

    fn kaiser_window(
        &self,
        size: usize,
        beta: f64,
        dtype: DType,
        device: &<CpuRuntime as Runtime>::Device,
    ) -> Result<Tensor<CpuRuntime>> {
        validate_window_size(size, "kaiser_window")?;
        validate_window_dtype(dtype, "kaiser_window")?;
        let values = generate_kaiser_f64(size, beta);
        create_window_tensor(values, dtype, device)
    }
}

fn create_window_tensor(
    values: Vec<f64>,
    dtype: DType,
    device: &<CpuRuntime as Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    let size = values.len();
    match dtype {
        DType::F32 => {
            let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
            Ok(Tensor::<CpuRuntime>::from_slice(
                &values_f32,
                &[size],
                device,
            ))
        }
        DType::F64 => Ok(Tensor::<CpuRuntime>::from_slice(&values, &[size], device)),
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
    use numr::runtime::cpu::CpuDevice;

    #[test]
    fn test_hann_window_cpu() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let window = client
            .hann_window(8, DType::F64, &device)
            .expect("hann_window failed");
        assert_eq!(window.shape(), &[8]);
    }

    #[test]
    fn test_kaiser_window_cpu() {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let window = client
            .kaiser_window(8, 5.0, DType::F32, &device)
            .expect("kaiser_window failed");
        assert_eq!(window.shape(), &[8]);
    }
}

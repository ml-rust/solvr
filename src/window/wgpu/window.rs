//! WebGPU backend implementation for window functions.

use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
use numr::tensor::Tensor;

use crate::window::impl_generic::{
    generate_blackman_f64, generate_hamming_f64, generate_hann_f64, generate_kaiser_f64,
};
use crate::window::traits::{WindowFunctions, validate_window_size};

impl WindowFunctions<WgpuRuntime> for WgpuClient {
    fn hann_window(
        &self,
        size: usize,
        dtype: DType,
        device: &WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_window_size(size, "hann_window")?;
        // WebGPU only supports F32
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "hann_window (WebGPU only supports F32)",
            });
        }
        let values = generate_hann_f64(size);
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        Ok(Tensor::<WgpuRuntime>::from_slice(
            &values_f32,
            &[size],
            device,
        ))
    }

    fn hamming_window(
        &self,
        size: usize,
        dtype: DType,
        device: &WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_window_size(size, "hamming_window")?;
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "hamming_window (WebGPU only supports F32)",
            });
        }
        let values = generate_hamming_f64(size);
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        Ok(Tensor::<WgpuRuntime>::from_slice(
            &values_f32,
            &[size],
            device,
        ))
    }

    fn blackman_window(
        &self,
        size: usize,
        dtype: DType,
        device: &WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_window_size(size, "blackman_window")?;
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "blackman_window (WebGPU only supports F32)",
            });
        }
        let values = generate_blackman_f64(size);
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        Ok(Tensor::<WgpuRuntime>::from_slice(
            &values_f32,
            &[size],
            device,
        ))
    }

    fn kaiser_window(
        &self,
        size: usize,
        beta: f64,
        dtype: DType,
        device: &WgpuDevice,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_window_size(size, "kaiser_window")?;
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "kaiser_window (WebGPU only supports F32)",
            });
        }
        let values = generate_kaiser_f64(size, beta);
        let values_f32: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        Ok(Tensor::<WgpuRuntime>::from_slice(
            &values_f32,
            &[size],
            device,
        ))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore] // WebGPU tests require GPU
    fn test_hann_window_wgpu() {
        // WebGPU initialization is async, so this test is skipped by default
    }
}

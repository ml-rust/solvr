//! CPU implementation of geometric transform algorithms.

use crate::interpolate::error::InterpolateResult;
use crate::interpolate::impl_generic::geometric::{
    affine_transform_impl, map_coordinates_impl, rotate_impl, shift_impl, zoom_impl,
};
use crate::interpolate::traits::geometric::{GeometricTransformAlgorithms, InterpolationOrder};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl GeometricTransformAlgorithms<CpuRuntime> for CpuClient {
    fn map_coordinates(
        &self,
        input: &Tensor<CpuRuntime>,
        coordinates: &Tensor<CpuRuntime>,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        map_coordinates_impl(self, input, coordinates, order, cval)
    }

    fn affine_transform(
        &self,
        input: &Tensor<CpuRuntime>,
        matrix: &Tensor<CpuRuntime>,
        offset: &Tensor<CpuRuntime>,
        output_shape: Option<&[usize]>,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        affine_transform_impl(self, input, matrix, offset, output_shape, order, cval)
    }

    fn zoom(
        &self,
        input: &Tensor<CpuRuntime>,
        zoom: &[f64],
        order: InterpolationOrder,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        zoom_impl(self, input, zoom, order)
    }

    fn rotate(
        &self,
        input: &Tensor<CpuRuntime>,
        angle: f64,
        axes: (usize, usize),
        reshape: bool,
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        rotate_impl(self, input, angle, axes, reshape, order, cval)
    }

    fn shift(
        &self,
        input: &Tensor<CpuRuntime>,
        shift: &[f64],
        order: InterpolationOrder,
        cval: f64,
    ) -> InterpolateResult<Tensor<CpuRuntime>> {
        shift_impl(self, input, shift, order, cval)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_zoom_2x() {
        let (client, device) = setup();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_slice(&data, &[2, 2], &device).unwrap();
        let result = client
            .zoom(&input, &[2.0, 2.0], InterpolationOrder::Nearest)
            .unwrap();
        assert_eq!(result.shape(), &[4, 4]);
    }

    #[test]
    fn test_rotate_90() {
        let (client, device) = setup();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_slice(&data, &[2, 2], &device).unwrap();
        let result = client
            .rotate(
                &input,
                90.0,
                (0, 1),
                false,
                InterpolationOrder::Nearest,
                0.0,
            )
            .unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[test]
    fn test_shift_identity() {
        let (client, device) = setup();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::from_slice(&data, &[5], &device).unwrap();
        let result = client
            .shift(&input, &[0.0], InterpolationOrder::Linear, 0.0)
            .unwrap();
        let out: Vec<f64> = result.to_vec();
        for (a, b) in data.iter().zip(out.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}

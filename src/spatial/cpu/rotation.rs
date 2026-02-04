//! CPU implementation of rotation algorithms.

use crate::spatial::impl_generic::{
    rotation_apply_impl, rotation_as_euler_impl, rotation_as_matrix_impl, rotation_as_quat_impl,
    rotation_as_rotvec_impl, rotation_compose_impl, rotation_from_axis_angle_impl,
    rotation_from_euler_impl, rotation_from_matrix_impl, rotation_from_quat_impl,
    rotation_from_rotvec_impl, rotation_identity_impl, rotation_inverse_impl,
    rotation_magnitude_impl, rotation_random_impl, rotation_slerp_impl,
};
use crate::spatial::traits::rotation::{EulerOrder, Rotation, RotationAlgorithms};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl RotationAlgorithms<CpuRuntime> for CpuClient {
    fn rotation_from_quat(&self, quaternion: &Tensor<CpuRuntime>) -> Result<Rotation<CpuRuntime>> {
        rotation_from_quat_impl(self, quaternion)
    }

    fn rotation_from_matrix(&self, matrix: &Tensor<CpuRuntime>) -> Result<Rotation<CpuRuntime>> {
        rotation_from_matrix_impl(self, matrix)
    }

    fn rotation_from_euler(
        &self,
        angles: &Tensor<CpuRuntime>,
        order: EulerOrder,
    ) -> Result<Rotation<CpuRuntime>> {
        rotation_from_euler_impl(self, angles, order)
    }

    fn rotation_from_axis_angle(
        &self,
        axis: &Tensor<CpuRuntime>,
        angle: &Tensor<CpuRuntime>,
    ) -> Result<Rotation<CpuRuntime>> {
        rotation_from_axis_angle_impl(self, axis, angle)
    }

    fn rotation_from_rotvec(&self, rotvec: &Tensor<CpuRuntime>) -> Result<Rotation<CpuRuntime>> {
        rotation_from_rotvec_impl(self, rotvec)
    }

    fn rotation_as_quat(&self, rot: &Rotation<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        rotation_as_quat_impl(self, rot)
    }

    fn rotation_as_matrix(&self, rot: &Rotation<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        rotation_as_matrix_impl(self, rot)
    }

    fn rotation_as_euler(
        &self,
        rot: &Rotation<CpuRuntime>,
        order: EulerOrder,
    ) -> Result<Tensor<CpuRuntime>> {
        rotation_as_euler_impl(self, rot, order)
    }

    fn rotation_as_rotvec(&self, rot: &Rotation<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        rotation_as_rotvec_impl(self, rot)
    }

    fn rotation_apply(
        &self,
        rot: &Rotation<CpuRuntime>,
        vectors: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        rotation_apply_impl(self, rot, vectors)
    }

    fn rotation_compose(
        &self,
        r1: &Rotation<CpuRuntime>,
        r2: &Rotation<CpuRuntime>,
    ) -> Result<Rotation<CpuRuntime>> {
        rotation_compose_impl(self, r1, r2)
    }

    fn rotation_inverse(&self, rot: &Rotation<CpuRuntime>) -> Result<Rotation<CpuRuntime>> {
        rotation_inverse_impl(self, rot)
    }

    fn rotation_slerp(
        &self,
        r1: &Rotation<CpuRuntime>,
        r2: &Rotation<CpuRuntime>,
        t: f64,
    ) -> Result<Rotation<CpuRuntime>> {
        rotation_slerp_impl(self, r1, r2, t)
    }

    fn rotation_identity(&self, n: Option<usize>) -> Result<Rotation<CpuRuntime>> {
        rotation_identity_impl::<CpuRuntime, Self>(self, n)
    }

    fn rotation_random(&self, n: Option<usize>) -> Result<Rotation<CpuRuntime>> {
        rotation_random_impl(self, n)
    }

    fn rotation_magnitude(&self, rot: &Rotation<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        rotation_magnitude_impl(self, rot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;
    use std::f64::consts::PI;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_rotation_identity() {
        let (client, _device) = setup();

        let rot = client.rotation_identity(None).unwrap();
        let quat = client.rotation_as_quat(&rot).unwrap();

        let data: Vec<f64> = quat.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-6); // w = 1
        assert!(data[1].abs() < 1e-6); // x = 0
        assert!(data[2].abs() < 1e-6); // y = 0
        assert!(data[3].abs() < 1e-6); // z = 0
    }

    #[test]
    fn test_rotation_from_euler_xyz() {
        let (client, device) = setup();

        // 90 degree rotation around Z
        let angles = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, PI / 2.0], &[3], &device);

        let rot = client
            .rotation_from_euler(&angles, EulerOrder::XYZ)
            .unwrap();
        let matrix = client.rotation_as_matrix(&rot).unwrap();

        let mat_data: Vec<f64> = matrix.to_vec();
        // 90 deg Z rotation: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        assert!(mat_data[0].abs() < 1e-6); // cos(90) = 0
        assert!((mat_data[1] + 1.0).abs() < 1e-6); // -sin(90) = -1
    }

    #[test]
    fn test_rotation_compose() {
        let (client, _device) = setup();

        let r1 = client.rotation_identity(None).unwrap();
        let r2 = client.rotation_identity(None).unwrap();

        let r_composed = client.rotation_compose(&r1, &r2).unwrap();
        let quat = client.rotation_as_quat(&r_composed).unwrap();

        let data: Vec<f64> = quat.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_inverse() {
        let (client, device) = setup();

        let angles = Tensor::<CpuRuntime>::from_slice(&[0.1, 0.2, 0.3], &[3], &device);
        let rot = client
            .rotation_from_euler(&angles, EulerOrder::XYZ)
            .unwrap();

        let rot_inv = client.rotation_inverse(&rot).unwrap();
        let composed = client.rotation_compose(&rot, &rot_inv).unwrap();

        // Should be identity
        let quat = client.rotation_as_quat(&composed).unwrap();
        let data: Vec<f64> = quat.to_vec();
        assert!((data[0].abs() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rotation_slerp() {
        let (client, _device) = setup();

        let r1 = client.rotation_identity(None).unwrap();
        let r2 = client.rotation_identity(None).unwrap();

        // Slerp at t=0.5 between two identical rotations should give the same
        let result = client.rotation_slerp(&r1, &r2, 0.5).unwrap();
        let quat = client.rotation_as_quat(&result).unwrap();

        let data: Vec<f64> = quat.to_vec();
        assert!((data[0] - 1.0).abs() < 1e-6);
    }
}

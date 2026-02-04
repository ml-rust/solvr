//! WebGPU implementation of rotation algorithms.

use crate::spatial::impl_generic::{
    rotation_apply_impl, rotation_as_euler_impl, rotation_as_matrix_impl, rotation_as_quat_impl,
    rotation_as_rotvec_impl, rotation_compose_impl, rotation_from_axis_angle_impl,
    rotation_from_euler_impl, rotation_from_matrix_impl, rotation_from_quat_impl,
    rotation_from_rotvec_impl, rotation_identity_impl, rotation_inverse_impl,
    rotation_magnitude_impl, rotation_random_impl, rotation_slerp_impl,
};
use crate::spatial::traits::rotation::{EulerOrder, Rotation, RotationAlgorithms};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl RotationAlgorithms<WgpuRuntime> for WgpuClient {
    fn rotation_from_quat(
        &self,
        quaternion: &Tensor<WgpuRuntime>,
    ) -> Result<Rotation<WgpuRuntime>> {
        rotation_from_quat_impl(self, quaternion)
    }

    fn rotation_from_matrix(&self, matrix: &Tensor<WgpuRuntime>) -> Result<Rotation<WgpuRuntime>> {
        rotation_from_matrix_impl(self, matrix)
    }

    fn rotation_from_euler(
        &self,
        angles: &Tensor<WgpuRuntime>,
        order: EulerOrder,
    ) -> Result<Rotation<WgpuRuntime>> {
        rotation_from_euler_impl(self, angles, order)
    }

    fn rotation_from_axis_angle(
        &self,
        axis: &Tensor<WgpuRuntime>,
        angle: &Tensor<WgpuRuntime>,
    ) -> Result<Rotation<WgpuRuntime>> {
        rotation_from_axis_angle_impl(self, axis, angle)
    }

    fn rotation_from_rotvec(&self, rotvec: &Tensor<WgpuRuntime>) -> Result<Rotation<WgpuRuntime>> {
        rotation_from_rotvec_impl(self, rotvec)
    }

    fn rotation_as_quat(&self, rot: &Rotation<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        rotation_as_quat_impl(self, rot)
    }

    fn rotation_as_matrix(&self, rot: &Rotation<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        rotation_as_matrix_impl(self, rot)
    }

    fn rotation_as_euler(
        &self,
        rot: &Rotation<WgpuRuntime>,
        order: EulerOrder,
    ) -> Result<Tensor<WgpuRuntime>> {
        rotation_as_euler_impl(self, rot, order)
    }

    fn rotation_as_rotvec(&self, rot: &Rotation<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        rotation_as_rotvec_impl(self, rot)
    }

    fn rotation_apply(
        &self,
        rot: &Rotation<WgpuRuntime>,
        vectors: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        rotation_apply_impl(self, rot, vectors)
    }

    fn rotation_compose(
        &self,
        r1: &Rotation<WgpuRuntime>,
        r2: &Rotation<WgpuRuntime>,
    ) -> Result<Rotation<WgpuRuntime>> {
        rotation_compose_impl(self, r1, r2)
    }

    fn rotation_inverse(&self, rot: &Rotation<WgpuRuntime>) -> Result<Rotation<WgpuRuntime>> {
        rotation_inverse_impl(self, rot)
    }

    fn rotation_slerp(
        &self,
        r1: &Rotation<WgpuRuntime>,
        r2: &Rotation<WgpuRuntime>,
        t: f64,
    ) -> Result<Rotation<WgpuRuntime>> {
        rotation_slerp_impl(self, r1, r2, t)
    }

    fn rotation_identity(&self, n: Option<usize>) -> Result<Rotation<WgpuRuntime>> {
        rotation_identity_impl::<WgpuRuntime, Self>(self, n)
    }

    fn rotation_random(&self, n: Option<usize>) -> Result<Rotation<WgpuRuntime>> {
        rotation_random_impl(self, n)
    }

    fn rotation_magnitude(&self, rot: &Rotation<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        rotation_magnitude_impl(self, rot)
    }
}

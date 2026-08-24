//! CPU implementation of halfspace intersection.

use crate::spatial::impl_generic::halfspace_intersection_impl;
use crate::spatial::traits::halfspace_intersection::{
    HalfspaceIntersection, HalfspaceIntersectionAlgorithms,
};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl HalfspaceIntersectionAlgorithms<CpuRuntime> for CpuClient {
    fn halfspace_intersection(
        &self,
        halfspaces: &Tensor<CpuRuntime>,
        interior_point: &Tensor<CpuRuntime>,
    ) -> Result<HalfspaceIntersection<CpuRuntime>> {
        halfspace_intersection_impl(self, halfspaces, interior_point)
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
    fn test_halfspace_2d_triangle() {
        let (client, device) = setup();

        // Triangle: x >= 0, y >= 0, x + y <= 1
        // As n·x + b ≤ 0:
        // -x ≤ 0       → [-1, 0, 0]
        // -y ≤ 0       → [0, -1, 0]
        // x + y - 1 ≤ 0 → [1, 1, -1]
        #[rustfmt::skip]
        let halfspaces = Tensor::<CpuRuntime>::from_slice(&[
            -1.0,  0.0,  0.0,
             0.0, -1.0,  0.0,
             1.0,  1.0, -1.0,
        ], &[3, 3], &device).unwrap();

        let interior = Tensor::<CpuRuntime>::from_slice(&[0.25, 0.25], &[2], &device).unwrap();

        let result = client
            .halfspace_intersection(&halfspaces, &interior)
            .unwrap();

        // Should have 3 vertices: (0,0), (1,0), (0,1)
        assert_eq!(result.intersections.shape()[1], 2);
        let n_verts = result.intersections.shape()[0];
        assert_eq!(
            n_verts, 3,
            "Triangle should have 3 vertices, got {}",
            n_verts
        );

        let verts: Vec<f64> = result.intersections.to_vec();
        // Verify all expected vertices are present
        let expected = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        for (ex, ey) in &expected {
            let found = (0..n_verts).any(|i| {
                let vx = verts[i * 2];
                let vy = verts[i * 2 + 1];
                (vx - ex).abs() < 1e-6 && (vy - ey).abs() < 1e-6
            });
            assert!(found, "Expected vertex ({}, {}) not found", ex, ey);
        }
    }

    #[test]
    fn test_halfspace_2d_square() {
        let (client, device) = setup();

        // Unit square: 0 ≤ x ≤ 1, 0 ≤ y ≤ 1
        // As n·x + b ≤ 0:
        // -x ≤ 0     → [-1, 0, 0]
        // x - 1 ≤ 0  → [1, 0, -1]
        // -y ≤ 0     → [0, -1, 0]
        // y - 1 ≤ 0  → [0, 1, -1]
        #[rustfmt::skip]
        let halfspaces = Tensor::<CpuRuntime>::from_slice(&[
            -1.0,  0.0,  0.0,
             1.0,  0.0, -1.0,
             0.0, -1.0,  0.0,
             0.0,  1.0, -1.0,
        ], &[4, 3], &device).unwrap();

        let interior = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[2], &device).unwrap();

        let result = client
            .halfspace_intersection(&halfspaces, &interior)
            .unwrap();

        assert_eq!(result.intersections.shape()[0], 4);
        assert_eq!(result.intersections.shape()[1], 2);

        let verts: Vec<f64> = result.intersections.to_vec();
        let expected = vec![(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)];
        for (ex, ey) in &expected {
            let found = (0..4)
                .any(|i| (verts[i * 2] - ex).abs() < 1e-6 && (verts[i * 2 + 1] - ey).abs() < 1e-6);
            assert!(found, "Expected vertex ({}, {}) not found", ex, ey);
        }
    }

    #[test]
    fn test_halfspace_3d_cube() {
        let (client, device) = setup();

        // Unit cube: 6 halfspaces
        #[rustfmt::skip]
        let halfspaces = Tensor::<CpuRuntime>::from_slice(&[
            -1.0,  0.0,  0.0,  0.0,  // -x ≤ 0
             1.0,  0.0,  0.0, -1.0,  // x - 1 ≤ 0
             0.0, -1.0,  0.0,  0.0,  // -y ≤ 0
             0.0,  1.0,  0.0, -1.0,  // y - 1 ≤ 0
             0.0,  0.0, -1.0,  0.0,  // -z ≤ 0
             0.0,  0.0,  1.0, -1.0,  // z - 1 ≤ 0
        ], &[6, 4], &device).unwrap();

        let interior = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5, 0.5], &[3], &device).unwrap();

        let result = client
            .halfspace_intersection(&halfspaces, &interior)
            .unwrap();

        // Cube has 8 vertices
        assert_eq!(result.intersections.shape()[0], 8);
        assert_eq!(result.intersections.shape()[1], 3);
    }

    #[test]
    fn test_halfspace_invalid_interior() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let halfspaces = Tensor::<CpuRuntime>::from_slice(&[
            -1.0, 0.0, 0.0,
             0.0,-1.0, 0.0,
             1.0, 1.0,-1.0,
        ], &[3, 3], &device).unwrap();

        // Point outside the halfspaces
        let interior = Tensor::<CpuRuntime>::from_slice(&[2.0, 2.0], &[2], &device).unwrap();

        assert!(
            client
                .halfspace_intersection(&halfspaces, &interior)
                .is_err()
        );
    }
}

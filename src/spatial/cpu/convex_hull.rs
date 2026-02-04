//! CPU implementation of convex hull algorithms.

use crate::spatial::impl_generic::{convex_hull_contains_impl, convex_hull_impl};
use crate::spatial::traits::convex_hull::{ConvexHull, ConvexHullAlgorithms};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl ConvexHullAlgorithms<CpuRuntime> for CpuClient {
    fn convex_hull(&self, points: &Tensor<CpuRuntime>) -> Result<ConvexHull<CpuRuntime>> {
        convex_hull_impl(self, points)
    }

    fn convex_hull_contains(
        &self,
        hull: &ConvexHull<CpuRuntime>,
        points: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        convex_hull_contains_impl(self, hull, points)
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
    fn test_convex_hull_2d_square() {
        let (client, device) = setup();

        // Square corners
        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[4, 2],
            &device,
        );

        let hull = client.convex_hull(&points).unwrap();

        // All 4 corners should be on hull
        assert_eq!(hull.vertices.shape()[0], 4);

        // 4 edges
        assert_eq!(hull.simplices.shape()[0], 4);

        // Area should be 1.0
        assert!((hull.volume - 1.0).abs() < 1e-6);

        // Perimeter should be 4.0
        assert!((hull.area - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_convex_hull_2d_with_interior_point() {
        let (client, device) = setup();

        // Square corners plus interior point
        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.5],
            &[5, 2],
            &device,
        );

        let hull = client.convex_hull(&points).unwrap();

        // Interior point should not be on hull
        assert_eq!(hull.vertices.shape()[0], 4);
    }

    #[test]
    fn test_convex_hull_contains_2d() {
        let (client, device) = setup();

        // Square
        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[4, 2],
            &device,
        );

        let hull = client.convex_hull(&points).unwrap();

        // Test points: (0.5, 0.5) inside, (2.0, 2.0) outside
        let test_points = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5, 2.0, 2.0], &[2, 2], &device);

        let result = client.convex_hull_contains(&hull, &test_points).unwrap();
        let data: Vec<f64> = result.to_vec();

        assert!(data[0] > 0.5); // Inside
        assert!(data[1] < 0.5); // Outside
    }

    #[test]
    fn test_convex_hull_3d_tetrahedron() {
        let (client, device) = setup();

        // Tetrahedron
        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[4, 3],
            &device,
        );

        let hull = client.convex_hull(&points).unwrap();

        // 4 vertices
        assert_eq!(hull.vertices.shape()[0], 4);

        // 4 triangular faces
        assert_eq!(hull.simplices.shape()[0], 4);
        assert_eq!(hull.simplices.shape()[1], 3);

        // Volume of tetrahedron with unit edges from origin = 1/6
        assert!((hull.volume - 1.0 / 6.0).abs() < 1e-6);
    }
}

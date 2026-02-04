//! CPU implementation of Delaunay triangulation.

use crate::spatial::impl_generic::{
    delaunay_find_simplex_impl, delaunay_impl, delaunay_vertex_neighbors_impl,
};
use crate::spatial::traits::delaunay::{Delaunay, DelaunayAlgorithms};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl DelaunayAlgorithms<CpuRuntime> for CpuClient {
    fn delaunay(&self, points: &Tensor<CpuRuntime>) -> Result<Delaunay<CpuRuntime>> {
        delaunay_impl(self, points)
    }

    fn delaunay_find_simplex(
        &self,
        tri: &Delaunay<CpuRuntime>,
        query: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        delaunay_find_simplex_impl(self, tri, query)
    }

    fn delaunay_vertex_neighbors(
        &self,
        tri: &Delaunay<CpuRuntime>,
    ) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
        delaunay_vertex_neighbors_impl(self, tri)
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
    fn test_delaunay_square() {
        let (client, device) = setup();

        // Square
        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[4, 2],
            &device,
        );

        let tri = client.delaunay(&points).unwrap();

        // Square should have 2 triangles
        assert_eq!(tri.simplices.shape()[0], 2);
        assert_eq!(tri.simplices.shape()[1], 3);
    }

    #[test]
    fn test_delaunay_find_simplex() {
        let (client, device) = setup();

        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[4, 2],
            &device,
        );

        let tri = client.delaunay(&points).unwrap();

        // Query point inside the triangulation
        let query = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.5], &[1, 2], &device);

        let result = client.delaunay_find_simplex(&tri, &query).unwrap();
        let indices: Vec<i64> = result.to_vec();

        // Should find a valid simplex
        assert!(indices[0] >= 0);
    }

    #[test]
    fn test_delaunay_vertex_neighbors() {
        let (client, device) = setup();

        let points = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            &[4, 2],
            &device,
        );

        let tri = client.delaunay(&points).unwrap();
        let (_indices, indptr) = client.delaunay_vertex_neighbors(&tri).unwrap();

        // indptr should have n+1 entries
        assert_eq!(indptr.shape()[0], 5);

        // Each vertex should have neighbors
        let indptr_data: Vec<i64> = indptr.to_vec();
        for i in 0..4 {
            assert!(indptr_data[i + 1] > indptr_data[i]);
        }
    }

    #[test]
    fn test_delaunay_convex_hull() {
        let (client, device) = setup();

        // Triangle
        let points =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0], &[3, 2], &device);

        let tri = client.delaunay(&points).unwrap();

        // All 3 points should be on convex hull
        assert_eq!(tri.convex_hull.shape()[0], 3);
    }
}

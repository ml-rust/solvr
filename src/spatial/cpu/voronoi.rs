//! CPU implementation of Voronoi diagram.

use crate::spatial::impl_generic::{
    voronoi_find_region_impl, voronoi_from_delaunay_impl, voronoi_impl,
};
use crate::spatial::traits::delaunay::Delaunay;
use crate::spatial::traits::voronoi::{Voronoi, VoronoiAlgorithms};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl VoronoiAlgorithms<CpuRuntime> for CpuClient {
    fn voronoi(&self, points: &Tensor<CpuRuntime>) -> Result<Voronoi<CpuRuntime>> {
        voronoi_impl(self, points)
    }

    fn voronoi_from_delaunay(&self, tri: &Delaunay<CpuRuntime>) -> Result<Voronoi<CpuRuntime>> {
        voronoi_from_delaunay_impl(self, tri)
    }

    fn voronoi_find_region(
        &self,
        vor: &Voronoi<CpuRuntime>,
        query: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        voronoi_find_region_impl(self, vor, query)
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
    fn test_voronoi_triangle() {
        let (client, device) = setup();

        let points =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0], &[3, 2], &device);

        let vor = client.voronoi(&points).unwrap();

        // Triangle has 1 Delaunay triangle, so 1 Voronoi vertex (circumcenter)
        assert_eq!(vor.vertices.shape()[0], 1);
        assert_eq!(vor.vertices.shape()[1], 2);
    }

    #[test]
    fn test_voronoi_find_region() {
        let (client, device) = setup();

        let points =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0], &[3, 2], &device);

        let vor = client.voronoi(&points).unwrap();

        // Query at generator 0
        let query = Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0], &[1, 2], &device);

        let result = client.voronoi_find_region(&vor, &query).unwrap();
        let regions: Vec<i64> = result.to_vec();

        // Should be region 0
        assert_eq!(regions[0], 0);
    }

    #[test]
    fn test_voronoi_find_region_multiple() {
        let (client, device) = setup();

        let points =
            Tensor::<CpuRuntime>::from_slice(&[0.0, 0.0, 2.0, 0.0, 1.0, 2.0], &[3, 2], &device);

        let vor = client.voronoi(&points).unwrap();

        // Query points near each generator
        let query =
            Tensor::<CpuRuntime>::from_slice(&[0.1, 0.1, 1.9, 0.1, 1.0, 1.9], &[3, 2], &device);

        let result = client.voronoi_find_region(&vor, &query).unwrap();
        let regions: Vec<i64> = result.to_vec();

        assert_eq!(regions[0], 0);
        assert_eq!(regions[1], 1);
        assert_eq!(regions[2], 2);
    }
}

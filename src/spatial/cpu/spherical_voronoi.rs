//! CPU implementation of spherical Voronoi diagram.

use crate::spatial::impl_generic::{
    spherical_voronoi_impl, spherical_voronoi_region_areas_impl,
    spherical_voronoi_sort_regions_impl,
};
use crate::spatial::traits::spherical_voronoi::{SphericalVoronoi, SphericalVoronoiAlgorithms};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl SphericalVoronoiAlgorithms<CpuRuntime> for CpuClient {
    fn spherical_voronoi(
        &self,
        points: &Tensor<CpuRuntime>,
        radius: f64,
        center: Option<&Tensor<CpuRuntime>>,
    ) -> Result<SphericalVoronoi<CpuRuntime>> {
        spherical_voronoi_impl(self, points, radius, center)
    }

    fn spherical_voronoi_sort_regions(
        &self,
        sv: &SphericalVoronoi<CpuRuntime>,
    ) -> Result<SphericalVoronoi<CpuRuntime>> {
        spherical_voronoi_sort_regions_impl(self, sv)
    }

    fn spherical_voronoi_region_areas(
        &self,
        sv: &SphericalVoronoi<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        spherical_voronoi_region_areas_impl(self, sv)
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
    fn test_spherical_voronoi_octahedron() {
        let (client, device) = setup();

        // 6 points: octahedron vertices on unit sphere
        #[rustfmt::skip]
        let points = Tensor::<CpuRuntime>::from_slice(&[
             1.0,  0.0,  0.0,
            -1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0, -1.0,  0.0,
             0.0,  0.0,  1.0,
             0.0,  0.0, -1.0,
        ], &[6, 3], &device).unwrap();

        let sv = client.spherical_voronoi(&points, 1.0, None).unwrap();

        assert_eq!(sv.points.shape(), &[6, 3]);
        assert_eq!(sv.vertices.shape()[1], 3);
        // Octahedron has 8 faces → 8 Voronoi vertices
        assert_eq!(sv.vertices.shape()[0], 8);
    }

    #[test]
    fn test_spherical_voronoi_tetrahedron() {
        let (client, device) = setup();

        // 4 points: regular tetrahedron on unit sphere
        let a = 1.0 / 3.0_f64.sqrt();
        #[rustfmt::skip]
        let points = Tensor::<CpuRuntime>::from_slice(&[
             a,  a,  a,
             a, -a, -a,
            -a,  a, -a,
            -a, -a,  a,
        ], &[4, 3], &device).unwrap();

        let sv = client.spherical_voronoi(&points, 1.0, None).unwrap();

        assert_eq!(sv.points.shape(), &[4, 3]);
        // Tetrahedron has 4 faces → 4 Voronoi vertices
        assert_eq!(sv.vertices.shape()[0], 4);
    }

    #[test]
    fn test_spherical_voronoi_region_areas_sum() {
        let (client, device) = setup();

        // Octahedron vertices
        #[rustfmt::skip]
        let points = Tensor::<CpuRuntime>::from_slice(&[
             1.0,  0.0,  0.0,
            -1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0, -1.0,  0.0,
             0.0,  0.0,  1.0,
             0.0,  0.0, -1.0,
        ], &[6, 3], &device).unwrap();

        let sv = client.spherical_voronoi(&points, 1.0, None).unwrap();
        let sv = client.spherical_voronoi_sort_regions(&sv).unwrap();
        let areas = client.spherical_voronoi_region_areas(&sv).unwrap();

        let areas_data: Vec<f64> = areas.to_vec();
        let total: f64 = areas_data.iter().sum();

        // Total should be 4*pi*r^2 = 4*pi ≈ 12.566
        let expected = 4.0 * std::f64::consts::PI;
        assert!(
            (total - expected).abs() < 0.5,
            "Total area {:.4} should be close to 4π = {:.4}",
            total,
            expected
        );
    }

    #[test]
    fn test_spherical_voronoi_non_unit_sphere() {
        let (client, device) = setup();

        let r = 2.0;
        #[rustfmt::skip]
        let points = Tensor::<CpuRuntime>::from_slice(&[
             r,  0.0, 0.0,
            -r,  0.0, 0.0,
             0.0, r,  0.0,
             0.0,-r,  0.0,
             0.0, 0.0, r,
             0.0, 0.0,-r,
        ], &[6, 3], &device).unwrap();

        let sv = client.spherical_voronoi(&points, r, None).unwrap();
        assert_eq!(sv.radius, r);
        assert_eq!(sv.vertices.shape()[0], 8);
    }

    #[test]
    fn test_spherical_voronoi_too_few_points() {
        let (client, device) = setup();

        let points =
            Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0, 0.0, -1.0, 0.0, 0.0], &[2, 3], &device)
                .unwrap();
        assert!(client.spherical_voronoi(&points, 1.0, None).is_err());
    }
}

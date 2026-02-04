//! WebGPU implementation of convex hull algorithms.

use crate::spatial::impl_generic::{convex_hull_contains_impl, convex_hull_impl};
use crate::spatial::traits::convex_hull::{ConvexHull, ConvexHullAlgorithms};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::tensor::Tensor;

impl ConvexHullAlgorithms<WgpuRuntime> for WgpuClient {
    fn convex_hull(&self, points: &Tensor<WgpuRuntime>) -> Result<ConvexHull<WgpuRuntime>> {
        convex_hull_impl(self, points)
    }

    fn convex_hull_contains(
        &self,
        hull: &ConvexHull<WgpuRuntime>,
        points: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        convex_hull_contains_impl(self, hull, points)
    }
}

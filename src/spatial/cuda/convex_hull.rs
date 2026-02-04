//! CUDA implementation of convex hull algorithms.

use crate::spatial::impl_generic::{convex_hull_contains_impl, convex_hull_impl};
use crate::spatial::traits::convex_hull::{ConvexHull, ConvexHullAlgorithms};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::tensor::Tensor;

impl ConvexHullAlgorithms<CudaRuntime> for CudaClient {
    fn convex_hull(&self, points: &Tensor<CudaRuntime>) -> Result<ConvexHull<CudaRuntime>> {
        convex_hull_impl(self, points)
    }

    fn convex_hull_contains(
        &self,
        hull: &ConvexHull<CudaRuntime>,
        points: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        convex_hull_contains_impl(self, hull, points)
    }
}

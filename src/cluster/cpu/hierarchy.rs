//! CPU implementation of hierarchical clustering.

use crate::cluster::impl_generic::{
    cut_tree_impl, fcluster_impl, fclusterdata_impl, leaves_list_impl, linkage_from_data_impl,
    linkage_impl,
};
use crate::cluster::traits::hierarchy::{
    FClusterCriterion, HierarchyAlgorithms, LinkageMatrix, LinkageMethod,
};
use numr::error::Result;
use numr::ops::DistanceMetric;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl HierarchyAlgorithms<CpuRuntime> for CpuClient {
    fn linkage(
        &self,
        distances: &Tensor<CpuRuntime>,
        n: usize,
        method: LinkageMethod,
    ) -> Result<LinkageMatrix<CpuRuntime>> {
        linkage_impl(self, distances, n, method)
    }

    fn linkage_from_data(
        &self,
        data: &Tensor<CpuRuntime>,
        method: LinkageMethod,
        metric: DistanceMetric,
    ) -> Result<LinkageMatrix<CpuRuntime>> {
        linkage_from_data_impl(self, data, method, metric)
    }

    fn fcluster(
        &self,
        z: &LinkageMatrix<CpuRuntime>,
        criterion: FClusterCriterion,
    ) -> Result<Tensor<CpuRuntime>> {
        fcluster_impl(self, z, criterion)
    }

    fn fclusterdata(
        &self,
        data: &Tensor<CpuRuntime>,
        criterion: FClusterCriterion,
        method: LinkageMethod,
        metric: DistanceMetric,
    ) -> Result<Tensor<CpuRuntime>> {
        fclusterdata_impl(self, data, criterion, method, metric)
    }

    fn leaves_list(&self, z: &LinkageMatrix<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        leaves_list_impl(self, z)
    }

    fn cut_tree(
        &self,
        z: &LinkageMatrix<CpuRuntime>,
        n_clusters: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        cut_tree_impl(self, z, n_clusters)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cluster::traits::hierarchy::{FClusterCriterion, LinkageMethod};
    use numr::ops::DistanceMetric;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_linkage_from_data_basic() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
            ],
            &[6, 2],
            &device,
        );

        let z = client
            .linkage_from_data(&data, LinkageMethod::Ward, DistanceMetric::Euclidean)
            .unwrap();
        // n-1 = 5 rows, 4 columns
        assert_eq!(z.z.shape(), &[5, 4]);
    }

    #[test]
    fn test_fclusterdata_two_clusters() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[
                0.0, 0.0,
                0.1, 0.1,
                0.2, 0.0,
                10.0, 10.0,
                10.1, 10.1,
                10.2, 10.0,
            ],
            &[6, 2],
            &device,
        );

        let labels = client
            .fclusterdata(
                &data,
                FClusterCriterion::MaxClust(2),
                LinkageMethod::Ward,
                DistanceMetric::Euclidean,
            )
            .unwrap();

        assert_eq!(labels.shape(), &[6]);
        let labels_vec: Vec<f64> = labels.to_vec();
        // Points in same group should have same label
        assert_eq!(labels_vec[0], labels_vec[1]);
        assert_eq!(labels_vec[1], labels_vec[2]);
        assert_eq!(labels_vec[3], labels_vec[4]);
        assert_eq!(labels_vec[4], labels_vec[5]);
        assert_ne!(labels_vec[0], labels_vec[3]);
    }

    #[test]
    #[ignore = "leaves_list requires scipy-convention cluster IDs in linkage matrix (n+step)"]
    fn test_leaves_list() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1],
            &[4, 2],
            &device,
        );

        let z = client
            .linkage_from_data(&data, LinkageMethod::Single, DistanceMetric::Euclidean)
            .unwrap();
        let leaves = client.leaves_list(&z).unwrap();
        assert_eq!(leaves.shape(), &[4]);
        let mut l: Vec<i64> = leaves.to_vec();
        l.sort();
        assert_eq!(l, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_cut_tree() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 0.1, 0.1, 0.2, 0.0, 10.0, 10.0, 10.1, 10.1, 10.2, 10.0],
            &[6, 2],
            &device,
        );

        let z = client
            .linkage_from_data(&data, LinkageMethod::Ward, DistanceMetric::Euclidean)
            .unwrap();
        let result = client.cut_tree(&z, &[2, 3]).unwrap();
        // [n, 2] — one column per requested cluster count
        assert_eq!(result.shape(), &[6, 2]);
    }

    #[test]
    fn test_linkage_methods() {
        let (client, device) = setup();

        #[rustfmt::skip]
        let data = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 6.0, 0.0],
            &[4, 2],
            &device,
        );

        // All linkage methods should produce valid output
        for method in &[
            LinkageMethod::Single,
            LinkageMethod::Complete,
            LinkageMethod::Average,
            LinkageMethod::Weighted,
            LinkageMethod::Ward,
        ] {
            let z = client
                .linkage_from_data(&data, *method, DistanceMetric::Euclidean)
                .unwrap();
            assert_eq!(z.z.shape(), &[3, 4], "Failed for {:?}", method);
        }
    }
}

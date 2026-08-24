//! CPU implementation of sparse QR algorithms.

use crate::linalg::traits::sparse_qr::SparseQrAlgorithms;
use numr::algorithm::sparse_linalg::qr::{self, QrFactors, QrMetrics, QrOptions, QrSymbolic};
use numr::error::Result;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::sparse::{CscData, SparseStorage};
use numr::tensor::Tensor;

impl SparseQrAlgorithms<CpuRuntime> for CpuClient {
    fn sparse_qr(
        &self,
        a: &CscData<CpuRuntime>,
        options: &QrOptions,
    ) -> Result<QrFactors<CpuRuntime>> {
        qr::sparse_qr_simple_cpu(a, options)
    }

    fn sparse_qr_with_symbolic(
        &self,
        a: &CscData<CpuRuntime>,
        symbolic: &QrSymbolic,
        options: &QrOptions,
    ) -> Result<QrFactors<CpuRuntime>> {
        qr::sparse_qr_cpu(a, symbolic, options)
    }

    fn sparse_qr_with_metrics(
        &self,
        a: &CscData<CpuRuntime>,
        symbolic: &QrSymbolic,
        options: &QrOptions,
    ) -> Result<(QrFactors<CpuRuntime>, QrMetrics)> {
        qr::sparse_qr_cpu_with_metrics(a, symbolic, options)
    }

    fn sparse_qr_symbolic(
        &self,
        a: &CscData<CpuRuntime>,
        options: &QrOptions,
    ) -> Result<QrSymbolic> {
        let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
        let row_indices: Vec<i64> = a.row_indices().to_vec();
        let [m, n] = a.shape();
        qr::sparse_qr_symbolic(&col_ptrs, &row_indices, m, n, options)
    }

    fn sparse_qr_solve(
        &self,
        factors: &QrFactors<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        qr::sparse_qr_solve_cpu(factors, b)
    }

    fn sparse_least_squares(
        &self,
        factors: &QrFactors<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        qr::sparse_qr_least_squares_cpu(factors, b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_device() -> <CpuRuntime as numr::runtime::Runtime>::Device {
        <CpuRuntime as numr::runtime::Runtime>::Device::default()
    }

    fn client() -> CpuClient {
        CpuClient::new(cpu_device())
    }

    fn create_tridiagonal_4x4() -> CscData<CpuRuntime> {
        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0f64, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        CscData::from_slices(&col_ptrs, &row_indices, &values, [4, 4], &cpu_device()).unwrap()
    }

    #[test]
    fn test_sparse_qr_trait_solve() {
        let c = client();
        let a = create_tridiagonal_4x4();
        let options = QrOptions::default();

        let factors = c.sparse_qr(&a, &options).unwrap();
        assert_eq!(factors.rank, 4);

        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[4], &cpu_device())
            .unwrap();
        let x = c.sparse_qr_solve(&factors, &b).unwrap();
        let x_vals: Vec<f64> = x.to_vec();

        // Verify Ax ≈ b
        let a_dense: &[&[f64]] = &[
            &[4.0, 1.0, 0.0, 0.0],
            &[1.0, 4.0, 1.0, 0.0],
            &[0.0, 1.0, 4.0, 1.0],
            &[0.0, 0.0, 1.0, 4.0],
        ];
        let b_vals = [1.0, 2.0, 3.0, 4.0];
        for i in 0..4 {
            let ax_i: f64 = (0..4).map(|j| a_dense[i][j] * x_vals[j]).sum();
            assert!(
                (ax_i - b_vals[i]).abs() < 1e-10,
                "A*x[{i}] = {ax_i}, expected {}",
                b_vals[i]
            );
        }
    }

    #[test]
    fn test_sparse_qr_trait_least_squares() {
        let c = client();
        let col_ptrs = vec![0i64, 3, 6, 8];
        let row_indices = vec![0i64, 2, 4, 1, 3, 4, 0, 3];
        let values = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let a =
            CscData::from_slices(&col_ptrs, &row_indices, &values, [5, 3], &cpu_device()).unwrap();

        let options = QrOptions::no_ordering();
        let factors = c.sparse_qr(&a, &options).unwrap();

        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &cpu_device())
                .unwrap();
        let x = c.sparse_least_squares(&factors, &b).unwrap();
        let x_vals: Vec<f64> = x.to_vec();

        // Verify optimality: A^T * (Ax - b) ≈ 0
        let a_dense = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let b_v = [1.0, 2.0, 3.0, 4.0, 5.0];
        for j in 0..3 {
            let at_r: f64 = (0..5)
                .map(|i| {
                    let ax_i: f64 = (0..3).map(|k| a_dense[i][k] * x_vals[k]).sum();
                    a_dense[i][j] * (ax_i - b_v[i])
                })
                .sum();
            assert!(at_r.abs() < 1e-10, "A^T*residual[{j}] = {at_r}");
        }
    }

    #[test]
    fn test_sparse_qr_symbolic_reuse() {
        let c = client();
        let a = create_tridiagonal_4x4();
        let options = QrOptions::default();

        // Compute symbolic once
        let symbolic = c.sparse_qr_symbolic(&a, &options).unwrap();

        // Factorize with symbolic
        let (factors, metrics) = c.sparse_qr_with_metrics(&a, &symbolic, &options).unwrap();
        assert_eq!(factors.rank, 4);
        assert!(metrics.r_nnz > 0);

        // Same symbolic, different values would work (same pattern)
        let factors2 = c.sparse_qr_with_symbolic(&a, &symbolic, &options).unwrap();
        assert_eq!(factors2.rank, 4);
    }
}

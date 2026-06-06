//! CUDA implementation of sparse QR algorithms.

use crate::linalg::traits::sparse_qr::SparseQrAlgorithms;
use numr::algorithm::sparse_linalg::qr::{self, QrFactors, QrMetrics, QrOptions, QrSymbolic};
use numr::error::Result;
use numr::runtime::cuda::{CudaClient, CudaRuntime};
use numr::sparse::{CscData, SparseStorage};
use numr::tensor::Tensor;

impl SparseQrAlgorithms<CudaRuntime> for CudaClient {
    fn sparse_qr(
        &self,
        a: &CscData<CudaRuntime>,
        options: &QrOptions,
    ) -> Result<QrFactors<CudaRuntime>> {
        // `sparse_qr_simple_cuda` needs the CPU-resident CSC structure to run the
        // symbolic analysis host-side (avoids GPU→CPU transfers during traversal).
        let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
        let row_indices: Vec<i64> = a.row_indices().to_vec();
        qr::sparse_qr_simple_cuda(self, a, &col_ptrs, &row_indices, options)
    }

    fn sparse_qr_with_symbolic(
        &self,
        a: &CscData<CudaRuntime>,
        symbolic: &QrSymbolic,
        options: &QrOptions,
    ) -> Result<QrFactors<CudaRuntime>> {
        qr::sparse_qr_cuda(self, a, symbolic, options)
    }

    fn sparse_qr_with_metrics(
        &self,
        a: &CscData<CudaRuntime>,
        symbolic: &QrSymbolic,
        options: &QrOptions,
    ) -> Result<(QrFactors<CudaRuntime>, QrMetrics)> {
        // CUDA backend doesn't have a separate metrics variant;
        // factorize and compute metrics from the result.
        let factors = qr::sparse_qr_cuda(self, a, symbolic, options)?;
        let r_nnz = factors.r.nnz();
        let original_nnz = a.nnz();
        let metrics = QrMetrics {
            original_nnz,
            r_nnz,
            fill_ratio: if original_nnz > 0 {
                r_nnz as f64 / original_nnz as f64
            } else {
                0.0
            },
            numerical_rank: factors.rank,
        };
        Ok((factors, metrics))
    }

    fn sparse_qr_symbolic(
        &self,
        a: &CscData<CudaRuntime>,
        options: &QrOptions,
    ) -> Result<QrSymbolic> {
        let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
        let row_indices: Vec<i64> = a.row_indices().to_vec();
        let [m, n] = a.shape();
        qr::sparse_qr_symbolic(&col_ptrs, &row_indices, m, n, options)
    }

    fn sparse_qr_solve(
        &self,
        factors: &QrFactors<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        qr::sparse_qr_solve_cuda(self, factors, b)
    }

    fn sparse_least_squares(
        &self,
        factors: &QrFactors<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        // For CUDA, solve using the same solve path (works for overdetermined too)
        qr::sparse_qr_solve_cuda(self, factors, b)
    }
}

//! WebGPU implementation of sparse QR algorithms (F32 only).

use crate::linalg::traits::sparse_qr::SparseQrAlgorithms;
use numr::algorithm::sparse_linalg::qr::{self, QrFactors, QrMetrics, QrOptions, QrSymbolic};
use numr::error::Result;
use numr::runtime::wgpu::{WgpuClient, WgpuRuntime};
use numr::sparse::{CscData, SparseStorage};
use numr::tensor::Tensor;

impl SparseQrAlgorithms<WgpuRuntime> for WgpuClient {
    fn sparse_qr(
        &self,
        a: &CscData<WgpuRuntime>,
        options: &QrOptions,
    ) -> Result<QrFactors<WgpuRuntime>> {
        // `sparse_qr_simple_wgpu` needs the CPU-resident CSC structure to run the
        // symbolic analysis host-side (avoids GPU→CPU transfers during traversal).
        let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
        let row_indices: Vec<i64> = a.row_indices().to_vec();
        qr::sparse_qr_simple_wgpu(self, a, &col_ptrs, &row_indices, options)
    }

    fn sparse_qr_with_symbolic(
        &self,
        a: &CscData<WgpuRuntime>,
        symbolic: &QrSymbolic,
        options: &QrOptions,
    ) -> Result<QrFactors<WgpuRuntime>> {
        qr::sparse_qr_wgpu(self, a, symbolic, options)
    }

    fn sparse_qr_with_metrics(
        &self,
        a: &CscData<WgpuRuntime>,
        symbolic: &QrSymbolic,
        options: &QrOptions,
    ) -> Result<(QrFactors<WgpuRuntime>, QrMetrics)> {
        let factors = qr::sparse_qr_wgpu(self, a, symbolic, options)?;
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
        a: &CscData<WgpuRuntime>,
        options: &QrOptions,
    ) -> Result<QrSymbolic> {
        let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
        let row_indices: Vec<i64> = a.row_indices().to_vec();
        let [m, n] = a.shape();
        qr::sparse_qr_symbolic(&col_ptrs, &row_indices, m, n, options)
    }

    fn sparse_qr_solve(
        &self,
        factors: &QrFactors<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        qr::sparse_qr_solve_wgpu(self, factors, b)
    }

    fn sparse_least_squares(
        &self,
        factors: &QrFactors<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        qr::sparse_qr_solve_wgpu(self, factors, b)
    }
}

pub mod matrix_equations;
#[cfg(feature = "sparse")]
pub mod sparse_qr;

pub use matrix_equations::MatrixEquationAlgorithms;
#[cfg(feature = "sparse")]
pub use sparse_qr::SparseQrAlgorithms;

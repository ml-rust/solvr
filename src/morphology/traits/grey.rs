//! Grey-scale morphology algorithm traits.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Algorithmic contract for grey-scale morphological operations.
pub trait GreyMorphologyAlgorithms<R: Runtime<DType = DType>> {
    /// Grey-scale erosion (local minimum filter).
    fn grey_erosion(&self, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>;

    /// Grey-scale dilation (local maximum filter).
    fn grey_dilation(&self, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>;

    /// Grey-scale opening (erosion then dilation).
    fn grey_opening(&self, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>;

    /// Grey-scale closing (dilation then erosion).
    fn grey_closing(&self, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>;

    /// Morphological gradient (dilation - erosion).
    fn morphological_gradient(&self, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>;

    /// White tophat (input - opening). Extracts bright features.
    fn white_tophat(&self, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>;

    /// Black tophat (closing - input). Extracts dark features.
    fn black_tophat(&self, input: &Tensor<R>, size: &[usize]) -> Result<Tensor<R>>;
}

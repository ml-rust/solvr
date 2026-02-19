//! Binary morphology algorithm traits.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Structuring element for morphological operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StructuringElement {
    /// Cross/plus-shaped structuring element (connectivity=1 in N-D).
    #[default]
    Cross,
    /// Full rectangular structuring element (connectivity=N in N-D).
    Full,
}

/// Algorithmic contract for binary morphological operations.
///
/// Input tensors are treated as binary: nonzero = true, zero = false.
/// Results are binary tensors with values 0.0 and 1.0.
pub trait BinaryMorphologyAlgorithms<R: Runtime<DType = DType>> {
    /// Binary erosion.
    ///
    /// A pixel in the output is 1 only if ALL pixels in the structuring element
    /// neighborhood are 1 in the input.
    ///
    /// # Arguments
    ///
    /// * `input` - Binary input tensor (nonzero = true)
    /// * `structure` - Structuring element shape
    /// * `iterations` - Number of times to apply the operation
    fn binary_erosion(
        &self,
        input: &Tensor<R>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<R>>;

    /// Binary dilation.
    ///
    /// A pixel in the output is 1 if ANY pixel in the structuring element
    /// neighborhood is 1 in the input.
    fn binary_dilation(
        &self,
        input: &Tensor<R>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<R>>;

    /// Binary opening (erosion followed by dilation).
    ///
    /// Removes small objects while preserving shape of larger objects.
    fn binary_opening(
        &self,
        input: &Tensor<R>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<R>>;

    /// Binary closing (dilation followed by erosion).
    ///
    /// Fills small holes while preserving shape of objects.
    fn binary_closing(
        &self,
        input: &Tensor<R>,
        structure: StructuringElement,
        iterations: usize,
    ) -> Result<Tensor<R>>;

    /// Fill holes in binary objects.
    ///
    /// Fills regions of 0s that are completely surrounded by 1s.
    fn binary_fill_holes(&self, input: &Tensor<R>) -> Result<Tensor<R>>;
}

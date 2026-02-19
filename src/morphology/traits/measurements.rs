//! Connected component labeling and region measurement traits.
use crate::DType;

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::binary::StructuringElement;

/// Properties of a labeled region.
#[derive(Debug, Clone)]
pub struct RegionProperties {
    /// Label ID
    pub label: usize,
    /// Number of pixels in the region
    pub area: usize,
    /// Bounding box as `[min_row, min_col, max_row, max_col]` (2D) or
    /// `[min_0, min_1, ..., max_0, max_1, ...]` (N-D)
    pub bbox: Vec<usize>,
}

/// Algorithmic contract for connected component labeling and measurements.
pub trait MeasurementAlgorithms<R: Runtime<DType = DType>> {
    /// Label connected components in a binary array.
    ///
    /// Each connected component gets a unique integer label starting from 1.
    /// Background (zero) pixels remain 0.
    ///
    /// # Arguments
    ///
    /// * `input` - Binary input tensor (nonzero = foreground)
    /// * `structure` - Structuring element defining connectivity
    ///
    /// # Returns
    ///
    /// Tuple of (labeled tensor, number of labels found).
    fn label(&self, input: &Tensor<R>, structure: StructuringElement)
    -> Result<(Tensor<R>, usize)>;

    /// Find bounding boxes of labeled regions.
    ///
    /// # Arguments
    ///
    /// * `labels` - Labeled tensor (output of `label()`)
    /// * `num_labels` - Number of labels
    ///
    /// # Returns
    ///
    /// Vector of RegionProperties for each label.
    fn find_objects(&self, labels: &Tensor<R>, num_labels: usize) -> Result<Vec<RegionProperties>>;

    /// Compute sum of input values for each labeled region.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor with values
    /// * `labels` - Label tensor
    /// * `num_labels` - Number of labels
    ///
    /// # Returns
    ///
    /// Tensor of shape `[num_labels]` with sum per region.
    fn sum_labels(
        &self,
        input: &Tensor<R>,
        labels: &Tensor<R>,
        num_labels: usize,
    ) -> Result<Tensor<R>>;

    /// Compute mean of input values for each labeled region.
    fn mean_labels(
        &self,
        input: &Tensor<R>,
        labels: &Tensor<R>,
        num_labels: usize,
    ) -> Result<Tensor<R>>;

    /// Compute center of mass for each labeled region.
    ///
    /// # Returns
    ///
    /// Tensor of shape [num_labels, ndim] with center of mass coordinates.
    fn center_of_mass(
        &self,
        input: &Tensor<R>,
        labels: &Tensor<R>,
        num_labels: usize,
    ) -> Result<Tensor<R>>;
}

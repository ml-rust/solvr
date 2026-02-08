//! Generic connected component labeling and measurement implementations.

use crate::morphology::traits::binary::StructuringElement;
use crate::morphology::traits::measurements::RegionProperties;
use numr::error::{Error, Result};
use numr::ops::ScatterReduceOp;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps,
    TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::prelude::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Pad a single axis of a tensor with a constant value.
fn pad_single_axis<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    axis: usize,
    before: usize,
    after: usize,
    value: f64,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ShapeOps<R> + RuntimeClient<R>,
{
    let ndim = tensor.ndim();
    let mut padding = vec![0usize; ndim * 2];
    let pad_idx = (ndim - 1 - axis) * 2;
    padding[pad_idx] = before;
    padding[pad_idx + 1] = after;
    client.pad(tensor, &padding, value)
}

/// Connected component labeling via iterative min-propagation.
///
/// Assigns sequential IDs to foreground pixels, then iteratively propagates
/// minimum labels through neighbors using pad+narrow+minimum (fully on-device).
/// Final remapping to consecutive labels uses CPU (API boundary).
pub fn label_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    structure: StructuringElement,
) -> Result<(Tensor<R>, usize)>
where
    R: Runtime,
    C: ScalarOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ShapeOps<R>
        + ReduceOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "label requires at least 1D input".to_string(),
        });
    }

    let shape = input.shape().to_vec();
    let total: usize = shape.iter().product();
    let device = input.device();
    let dtype = input.dtype();

    // Build foreground mask on-device
    let zero_tensor = Tensor::from_slice(&vec![0.0; total], &shape, device);
    let fg_mask = client.ne(input, &zero_tensor)?;

    // Assign sequential 1-based IDs to all pixels: 1, 2, 3, ...
    let ids = client.arange(1.0, (total + 1) as f64, 1.0, DType::F64)?;
    let ids = ids.reshape(&shape)?;

    let inf_val = (total + 1) as f64;
    let inf_tensor = Tensor::from_slice(&vec![inf_val; total], &shape, device);

    // Initialize: foreground gets sequential IDs, background gets INF
    let mut labels = client.where_cond(&fg_mask, &ids, &inf_tensor)?;

    let full_connectivity = matches!(structure, StructuringElement::Full);

    // Iterative min-propagation: update in-place per axis for faster convergence.
    let max_iter = total;
    for _ in 0..max_iter {
        let prev = labels.clone();

        for (axis, &axis_len) in shape.iter().enumerate() {
            if axis_len <= 1 {
                continue;
            }

            let padded = pad_single_axis(client, &labels, axis, 1, 1, inf_val)?;
            let left = padded.narrow(axis as isize, 0, axis_len)?;
            let right = padded.narrow(axis as isize, 2, axis_len)?;

            labels = client.minimum(&labels, &left)?;
            labels = client.minimum(&labels, &right)?;
        }

        // Diagonal neighbors for full connectivity (2D+)
        if full_connectivity && ndim >= 2 {
            for a1 in 0..ndim {
                for a2 in (a1 + 1)..ndim {
                    if shape[a1] <= 1 || shape[a2] <= 1 {
                        continue;
                    }

                    let padded = pad_single_axis(client, &labels, a1, 1, 1, inf_val)?;
                    let padded = pad_single_axis(client, &padded, a2, 1, 1, inf_val)?;

                    let len1 = shape[a1];
                    let len2 = shape[a2];

                    for (s1, s2) in [(0, 0), (0, 2), (2, 0), (2, 2)] {
                        let view = padded.narrow(a1 as isize, s1, len1)?;
                        let view = view.narrow(a2 as isize, s2, len2)?;
                        labels = client.minimum(&labels, &view)?;
                    }
                }
            }
        }

        // Re-apply foreground mask
        labels = client.where_cond(&fg_mask, &labels, &inf_tensor)?;

        // Convergence check — single scalar transfer (acceptable)
        let diff = client.sub(&labels, &prev)?;
        let diff_abs = client.abs(&diff)?;
        let diff_sum = client.sum(&diff_abs, &[], false)?;
        let val: Vec<f64> = diff_sum.to_vec();
        if val[0] < 0.5 {
            break;
        }
    }

    // Remap unique labels to consecutive 1..N (API boundary — requires sorting/dedup)
    let label_data: Vec<f64> = labels.to_vec();
    let mut unique_labels: Vec<f64> = label_data
        .iter()
        .filter(|&&v| v < inf_val)
        .copied()
        .collect();
    // NaN labels are treated as greater (pushed to end)
    unique_labels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater));
    unique_labels.dedup();

    let num_labels = unique_labels.len();

    let mut result = vec![0.0f64; total];
    if !unique_labels.is_empty() {
        use std::collections::HashMap;
        let remap: HashMap<u64, f64> = unique_labels
            .iter()
            .enumerate()
            .map(|(i, &v)| (v.to_bits(), (i + 1) as f64))
            .collect();

        for (i, &v) in label_data.iter().enumerate() {
            if let Some(&new_label) = remap.get(&v.to_bits()) {
                result[i] = new_label;
            }
        }
    }

    let tensor = Tensor::from_slice(&result, &shape, device);
    let tensor = client.cast(&tensor, dtype)?;
    Ok((tensor, num_labels))
}

/// Find bounding boxes of labeled regions.
///
/// Returns CPU structs — API boundary transfer is acceptable.
pub fn find_objects_impl<R, C>(
    _client: &C,
    labels: &Tensor<R>,
    num_labels: usize,
) -> Result<Vec<RegionProperties>>
where
    R: Runtime,
    C: RuntimeClient<R>,
{
    let ndim = labels.ndim();
    let shape = labels.shape().to_vec();
    let data: Vec<f64> = labels.to_vec();

    let mut props: Vec<RegionProperties> = (1..=num_labels)
        .map(|label| RegionProperties {
            label,
            area: 0,
            bbox: {
                let mut b = vec![usize::MAX; ndim];
                b.extend(vec![0usize; ndim]);
                b
            },
        })
        .collect();

    let total: usize = shape.iter().product();
    for (flat_idx, &label_val_raw) in data.iter().enumerate().take(total) {
        let label_val = label_val_raw as usize;
        if label_val == 0 || label_val > num_labels {
            continue;
        }

        let prop = &mut props[label_val - 1];
        prop.area += 1;

        let mut remaining = flat_idx;
        for d in (0..ndim).rev() {
            let coord = remaining % shape[d];
            remaining /= shape[d];
            prop.bbox[d] = prop.bbox[d].min(coord);
            prop.bbox[ndim + d] = prop.bbox[ndim + d].max(coord);
        }
    }

    Ok(props)
}

/// Sum of input values per labeled region using scatter_reduce (on-device).
pub fn sum_labels_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    labels: &Tensor<R>,
    num_labels: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + IndexingOps<R> + TypeConversionOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let device = input.device();
    let dtype = input.dtype();
    let total: usize = input.shape().iter().product();

    // Flatten input and labels to 1D
    let flat_input = input.reshape(&[total])?;
    let flat_labels = labels.reshape(&[total])?;

    // Convert 1-based labels to 0-based I64 indices for scatter_reduce
    let indices = client.add_scalar(&flat_labels, -1.0)?;
    let indices = client.cast(&indices, DType::I64)?;

    // Destination: zeros of shape [num_labels]
    let dst = Tensor::from_slice(&vec![0.0; num_labels], &[num_labels], device);

    let result =
        client.scatter_reduce(&dst, 0, &indices, &flat_input, ScatterReduceOp::Sum, true)?;

    client.cast(&result, dtype)
}

/// Mean of input values per labeled region using scatter_reduce (on-device).
pub fn mean_labels_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    labels: &Tensor<R>,
    num_labels: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R> + IndexingOps<R> + TypeConversionOps<R> + UtilityOps<R> + RuntimeClient<R>,
{
    let device = input.device();
    let dtype = input.dtype();
    let total: usize = input.shape().iter().product();

    let flat_input = input.reshape(&[total])?;
    let flat_labels = labels.reshape(&[total])?;
    let indices = client.add_scalar(&flat_labels, -1.0)?;
    let indices = client.cast(&indices, DType::I64)?;

    let dst = Tensor::from_slice(&vec![0.0; num_labels], &[num_labels], device);

    let result =
        client.scatter_reduce(&dst, 0, &indices, &flat_input, ScatterReduceOp::Mean, true)?;

    client.cast(&result, dtype)
}

/// Center of mass per labeled region using scatter_reduce (on-device).
///
/// For each dimension, computes weighted coordinate sum per label, then divides
/// by total weight per label.
pub fn center_of_mass_impl<R, C>(
    client: &C,
    input: &Tensor<R>,
    labels: &Tensor<R>,
    num_labels: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ScalarOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + TypeConversionOps<R>
        + ConditionalOps<R>
        + CompareOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + RuntimeClient<R>,
{
    let ndim = input.ndim();
    let shape = input.shape().to_vec();
    let device = input.device();
    let dtype = input.dtype();
    let total: usize = shape.iter().product();

    let flat_input = input.reshape(&[total])?;
    let flat_labels = labels.reshape(&[total])?;
    let indices = client.add_scalar(&flat_labels, -1.0)?;
    let indices = client.cast(&indices, DType::I64)?;

    // Compute total weight per label
    let dst_zeros = Tensor::from_slice(&vec![0.0; num_labels], &[num_labels], device);
    let total_weights = client.scatter_reduce(
        &dst_zeros,
        0,
        &indices,
        &flat_input,
        ScatterReduceOp::Sum,
        true,
    )?;

    // Build coordinate tensor for each dimension on-device
    // For a shape [H, W], flat_idx i has coord[d] = (i / stride_d) % shape[d]
    let mut results = Vec::with_capacity(ndim);

    for d in 0..ndim {
        // Compute coordinate values for dimension d across all flat indices
        let stride: usize = shape[d + 1..].iter().product();
        // coords[i] = (i / stride) % shape[d]
        let flat_indices = client.arange(0.0, total as f64, 1.0, DType::F64)?;
        let divided = client.mul_scalar(&flat_indices, 1.0 / stride as f64)?;
        // Floor division: floor(i / stride)
        let floored = client.floor(&divided)?;
        // Modulo: floor(i / stride) % shape[d]
        let shape_d = shape[d] as f64;
        let scaled = client.mul_scalar(&floored, 1.0 / shape_d)?;
        let floored2 = client.floor(&scaled)?;
        let subtract = client.mul_scalar(&floored2, shape_d)?;
        let coords = client.sub(&floored, &subtract)?;

        // Weighted coords = input * coord
        let weighted = client.mul(&flat_input, &coords)?;

        // Scatter-sum weighted coords per label
        let weighted_sum = client.scatter_reduce(
            &dst_zeros,
            0,
            &indices,
            &weighted,
            ScatterReduceOp::Sum,
            true,
        )?;

        // center = weighted_sum / total_weight (0 where weight is 0)
        let zero_mask = client.eq(
            &total_weights,
            &Tensor::from_slice(&vec![0.0; num_labels], &[num_labels], device),
        )?;
        let center = client.div(&weighted_sum, &total_weights)?;
        let zero_tensor = Tensor::from_slice(&vec![0.0; num_labels], &[num_labels], device);
        let center = client.where_cond(&zero_mask, &zero_tensor, &center)?;

        results.push(center);
    }

    // Stack results into [num_labels, ndim]
    // Reshape each [num_labels] -> [num_labels, 1], then concatenate along dim 1
    let mut stacked = results[0].reshape(&[num_labels, 1])?;
    for r in &results[1..] {
        let col = r.reshape(&[num_labels, 1])?;
        stacked = client.cat(&[&stacked, &col], 1)?;
    }

    client.cast(&stacked, dtype)
}

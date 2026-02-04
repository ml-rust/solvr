//! CPU implementation of local extrema detection algorithms.
//!
//! This algorithm is CPU-only because it requires element-wise comparisons
//! with variable-sized neighborhoods. While GPUs can parallelize comparisons,
//! the gather/scatter patterns for variable order make this inefficient.

use crate::signal::traits::extrema::{ExtremaAlgorithms, ExtremaResult, ExtremumMode};
use numr::error::{Error, Result};
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::tensor::Tensor;

impl ExtremaAlgorithms<CpuRuntime> for CpuClient {
    fn argrelmin(
        &self,
        x: &Tensor<CpuRuntime>,
        order: usize,
        mode: ExtremumMode,
    ) -> Result<ExtremaResult<CpuRuntime>> {
        argrel_cpu(x, order, mode, Comparison::Less)
    }

    fn argrelmax(
        &self,
        x: &Tensor<CpuRuntime>,
        order: usize,
        mode: ExtremumMode,
    ) -> Result<ExtremaResult<CpuRuntime>> {
        argrel_cpu(x, order, mode, Comparison::Greater)
    }
}

#[derive(Clone, Copy)]
enum Comparison {
    Less,
    Greater,
}

/// CPU implementation of local extrema detection.
fn argrel_cpu(
    x: &Tensor<CpuRuntime>,
    order: usize,
    mode: ExtremumMode,
    comparison: Comparison,
) -> Result<ExtremaResult<CpuRuntime>> {
    if x.ndim() != 1 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input must be 1D".to_string(),
        });
    }

    let n = x.shape()[0];
    let device = x.device();

    if n == 0 {
        return Err(Error::InvalidArgument {
            arg: "x",
            reason: "Input signal cannot be empty".to_string(),
        });
    }

    if order == 0 {
        return Err(Error::InvalidArgument {
            arg: "order",
            reason: "Order must be at least 1".to_string(),
        });
    }

    // CPU-specific: extract data for comparison
    let data: Vec<f64> = x.to_vec();

    let mut indices = Vec::new();
    let mut values = Vec::new();

    // Helper to get value at index with boundary handling
    let get_val = |idx: isize| -> f64 {
        match mode {
            ExtremumMode::Wrap => {
                let wrapped = ((idx % n as isize) + n as isize) as usize % n;
                data[wrapped]
            }
            ExtremumMode::Clip => {
                let clamped = idx.clamp(0, n as isize - 1) as usize;
                data[clamped]
            }
        }
    };

    // Check each point
    for (i, &val) in data.iter().enumerate() {
        let mut is_extremum = true;

        // Compare with neighbors within order distance
        for offset in 1..=order {
            let left_idx = i as isize - offset as isize;
            let right_idx = i as isize + offset as isize;

            // For Clip mode, skip comparisons that would be out of bounds
            let check_left = match mode {
                ExtremumMode::Wrap => true,
                ExtremumMode::Clip => left_idx >= 0,
            };

            let check_right = match mode {
                ExtremumMode::Wrap => true,
                ExtremumMode::Clip => right_idx < n as isize,
            };

            if check_left {
                let left_val = get_val(left_idx);
                match comparison {
                    Comparison::Less => {
                        if val >= left_val {
                            is_extremum = false;
                            break;
                        }
                    }
                    Comparison::Greater => {
                        if val <= left_val {
                            is_extremum = false;
                            break;
                        }
                    }
                }
            }

            if is_extremum && check_right {
                let right_val = get_val(right_idx);
                match comparison {
                    Comparison::Less => {
                        if val >= right_val {
                            is_extremum = false;
                            break;
                        }
                    }
                    Comparison::Greater => {
                        if val <= right_val {
                            is_extremum = false;
                            break;
                        }
                    }
                }
            }

            if !is_extremum {
                break;
            }
        }

        if is_extremum {
            indices.push(i);
            values.push(val);
        }
    }

    let values_tensor = Tensor::from_slice(&values, &[values.len()], device);

    Ok(ExtremaResult {
        indices,
        values: values_tensor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::CpuDevice;

    fn setup() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (client, device)
    }

    #[test]
    fn test_argrelmin_simple() {
        let (client, device) = setup();

        // Signal with clear minima at indices 2 and 6
        let signal = vec![1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.argrelmin(&x, 1, ExtremumMode::Clip).unwrap();

        assert_eq!(result.indices, vec![2, 6]);
        let values: Vec<f64> = result.values.to_vec();
        assert!((values[0] - 0.0).abs() < 1e-10);
        assert!((values[1] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_argrelmax_simple() {
        let (client, device) = setup();

        // Signal with clear maxima at indices 0, 4, 8
        let signal = vec![1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.argrelmax(&x, 1, ExtremumMode::Clip).unwrap();

        // With Clip mode, edges count as maxima if they're larger than their neighbors
        assert!(result.indices.contains(&4));
        // Edge behavior: index 0 is compared only with index 1 (0.5), so 1.0 > 0.5 makes it a max
        assert!(result.indices.contains(&0));
        // Same for index 8
        assert!(result.indices.contains(&8));
    }

    #[test]
    fn test_argrelmin_higher_order() {
        let (client, device) = setup();

        // With order=2, a minimum must be less than 2 neighbors on each side
        let signal = vec![5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.argrelmin(&x, 2, ExtremumMode::Clip).unwrap();

        // Index 4 (value 1.0) should be a local minimum with order=2
        assert_eq!(result.indices, vec![4]);
    }

    #[test]
    fn test_argrelmax_wrap_mode() {
        let (client, device) = setup();

        // Periodic signal
        let signal = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.argrelmax(&x, 1, ExtremumMode::Wrap).unwrap();

        // Maxima at indices 1 and 3
        assert_eq!(result.indices, vec![1, 3]);
    }

    #[test]
    fn test_argrelextrema() {
        let (client, device) = setup();

        let signal = vec![0.0, 1.0, 0.0, -1.0, 0.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let (minima, maxima) = client.argrelextrema(&x, 1, ExtremumMode::Clip).unwrap();

        // Maximum at index 1
        assert!(maxima.indices.contains(&1));
        // Minimum at index 3
        assert!(minima.indices.contains(&3));
    }

    #[test]
    fn test_argrelmin_no_extrema() {
        let (client, device) = setup();

        // Monotonically increasing signal - no local minima (except possibly edges)
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.argrelmin(&x, 1, ExtremumMode::Clip).unwrap();

        // With Clip mode, index 0 might be a minimum since it's only compared with index 1
        // 1.0 < 2.0 on the right, but we don't check left (clipped), so it IS a minimum
        assert_eq!(result.indices, vec![0]);
    }

    #[test]
    fn test_argrelmin_plateau() {
        let (client, device) = setup();

        // Signal with a plateau (equal values) - these should NOT be minima
        // because we require strictly less than neighbors
        let signal = vec![1.0, 0.0, 0.0, 0.0, 1.0];
        let x = Tensor::from_slice(&signal, &[signal.len()], &device);

        let result = client.argrelmin(&x, 1, ExtremumMode::Clip).unwrap();

        // No strict minima because 0.0 is not strictly less than adjacent 0.0
        assert!(result.indices.is_empty());
    }
}

//! Shared boundary condition extraction helpers.
use crate::DType;

use numr::runtime::Runtime;

use crate::pde::error::{PdeError, PdeResult};
use crate::pde::impl_generic::stencil::SideBc;
use crate::pde::types::{BoundaryCondition, BoundarySide, BoundarySpec};

/// Extract left/right Dirichlet scalar values from 1D boundary specs.
///
/// Returns `(left, right)` values. Only Dirichlet BCs are supported;
/// Neumann and Periodic produce an error with the given `solver_name` context.
pub fn extract_dirichlet_1d_bcs<R: Runtime<DType = DType>>(
    boundary: &[BoundarySpec<R>],
    solver_name: &str,
) -> PdeResult<(f64, f64)> {
    let mut left = 0.0;
    let mut right = 0.0;

    for spec in boundary {
        match &spec.condition {
            BoundaryCondition::Dirichlet(vals) => {
                let v: Vec<f64> = vals.to_vec();
                let val = if v.is_empty() { 0.0 } else { v[0] };
                match spec.side {
                    BoundarySide::Left => left = val,
                    BoundarySide::Right => right = val,
                    BoundarySide::All => {
                        left = val;
                        right = val;
                    }
                    _ => {}
                }
            }
            _ => {
                return Err(PdeError::InvalidBoundary {
                    context: format!("Only Dirichlet BCs supported for {}", solver_name),
                });
            }
        }
    }

    Ok((left, right))
}

/// Extract a single Dirichlet scalar from the first boundary spec.
///
/// Used by solvers that apply a uniform value to all boundary nodes.
pub fn extract_dirichlet_scalar<R: Runtime<DType = DType>>(
    boundary: &[BoundarySpec<R>],
    solver_name: &str,
) -> PdeResult<f64> {
    if let Some(spec) = boundary.first() {
        match &spec.condition {
            BoundaryCondition::Dirichlet(vals) => {
                let v: Vec<f64> = vals.to_vec();
                Ok(if v.is_empty() { 0.0 } else { v[0] })
            }
            _ => Err(PdeError::InvalidBoundary {
                context: format!("Only Dirichlet BCs supported for {}", solver_name),
            }),
        }
    } else {
        Ok(0.0)
    }
}

/// Per-side boundary condition kinds `[left, right, bottom, top]`.
///
/// Builds the [`SideBc`] array consumed by the general (mixed) Poisson
/// assembler. Sides not named in `boundary` default to homogeneous Dirichlet.
/// Validates that Periodic conditions are paired (left&right, or bottom&top),
/// which the wrapping stencil requires.
pub fn side_conditions_2d<R: Runtime<DType = DType>>(
    boundary: &[BoundarySpec<R>],
) -> PdeResult<[SideBc; 4]> {
    // [left, right, bottom, top]; unspecified sides default to Dirichlet.
    let mut sides = [SideBc::Dirichlet; 4];

    for spec in boundary {
        let bc = match &spec.condition {
            BoundaryCondition::Dirichlet(_) => SideBc::Dirichlet,
            BoundaryCondition::Neumann(vals) => {
                let v: Vec<f64> = vals.to_vec();
                SideBc::Neumann(if v.is_empty() { 0.0 } else { v[0] })
            }
            BoundaryCondition::Periodic => SideBc::Periodic,
        };
        match spec.side {
            BoundarySide::Left => sides[0] = bc,
            BoundarySide::Right => sides[1] = bc,
            BoundarySide::Bottom => sides[2] = bc,
            BoundarySide::Top => sides[3] = bc,
            BoundarySide::All => sides = [bc; 4],
            _ => {}
        }
    }

    let is_periodic = |s: SideBc| s == SideBc::Periodic;
    if is_periodic(sides[0]) != is_periodic(sides[1]) {
        return Err(PdeError::InvalidBoundary {
            context: "Periodic BC must be applied to both Left and Right sides (paired)"
                .to_string(),
        });
    }
    if is_periodic(sides[2]) != is_periodic(sides[3]) {
        return Err(PdeError::InvalidBoundary {
            context: "Periodic BC must be applied to both Bottom and Top sides (paired)"
                .to_string(),
        });
    }

    Ok(sides)
}

/// Extract 2D Dirichlet boundary values as a flat array `[nx*ny]` (row-major).
///
/// Fills per-side Dirichlet values; Neumann/Periodic sides carry no nodal value
/// and are skipped (so this is safe to call for mixed boundary specs — the
/// assembler only reads entries for nodes that lie on a Dirichlet side).
pub fn extract_boundary_values_2d<R: Runtime<DType = DType>>(
    boundary: &[BoundarySpec<R>],
    nx: usize,
    ny: usize,
) -> PdeResult<Vec<f64>> {
    let n = nx * ny;
    let mut values = vec![0.0; n];

    for spec in boundary {
        match &spec.condition {
            BoundaryCondition::Dirichlet(vals) => {
                let v: Vec<f64> = vals.to_vec();
                match spec.side {
                    BoundarySide::Left => {
                        let copy_len = ny.min(v.len());
                        values[..copy_len].copy_from_slice(&v[..copy_len]);
                    }
                    BoundarySide::Right => {
                        let copy_len = ny.min(v.len());
                        for (idx, &val) in v[..copy_len].iter().enumerate() {
                            values[(nx - 1) * ny + idx] = val;
                        }
                    }
                    BoundarySide::Bottom => {
                        let copy_len = nx.min(v.len());
                        for (i, &val) in v[..copy_len].iter().enumerate() {
                            values[i * ny] = val;
                        }
                    }
                    BoundarySide::Top => {
                        let copy_len = nx.min(v.len());
                        for (i, &val) in v[..copy_len].iter().enumerate() {
                            values[i * ny + (ny - 1)] = val;
                        }
                    }
                    BoundarySide::All => {
                        let val = if v.is_empty() { 0.0 } else { v[0] };
                        for i in 0..nx {
                            for j in 0..ny {
                                if i == 0 || i == nx - 1 || j == 0 || j == ny - 1 {
                                    values[i * ny + j] = val;
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            BoundaryCondition::Neumann(_) | BoundaryCondition::Periodic => {
                // Neumann/Periodic sides carry no fixed nodal value; skip them.
                // The mixed assembler uses `side_conditions_2d` to learn each
                // side's BC type and only reads these values for Dirichlet nodes.
            }
        }
    }

    Ok(values)
}

// ============================================================================
// Unit tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    fn device() -> CpuDevice {
        CpuDevice::new()
    }

    fn scalar_tensor(val: f64) -> Tensor<CpuRuntime> {
        let dev = device();
        Tensor::<CpuRuntime>::from_slice(&[val], &[1], &dev)
    }

    // ------------------------------------------------------------------
    // side_conditions_2d
    // ------------------------------------------------------------------

    #[test]
    fn test_side_conditions_default_dirichlet() {
        // Unspecified sides default to Dirichlet.
        let bcs: Vec<BoundarySpec<CpuRuntime>> = vec![];
        let sides = side_conditions_2d(&bcs).unwrap();
        assert_eq!(sides, [SideBc::Dirichlet; 4]);
    }

    #[test]
    fn test_side_conditions_per_side_mixed() {
        let bcs = vec![
            BoundarySpec {
                side: BoundarySide::Left,
                condition: BoundaryCondition::Dirichlet(scalar_tensor(0.0)),
            },
            BoundarySpec {
                side: BoundarySide::Right,
                condition: BoundaryCondition::Neumann(scalar_tensor(2.5)),
            },
            BoundarySpec {
                side: BoundarySide::Bottom,
                condition: BoundaryCondition::Dirichlet(scalar_tensor(1.0)),
            },
            BoundarySpec {
                side: BoundarySide::Top,
                condition: BoundaryCondition::Neumann(scalar_tensor(-1.0)),
            },
        ];
        let sides = side_conditions_2d(&bcs).unwrap();
        assert_eq!(sides[0], SideBc::Dirichlet);
        assert_eq!(sides[1], SideBc::Neumann(2.5));
        assert_eq!(sides[2], SideBc::Dirichlet);
        assert_eq!(sides[3], SideBc::Neumann(-1.0));
    }

    #[test]
    fn test_side_conditions_periodic_paired_ok() {
        let bcs: Vec<BoundarySpec<CpuRuntime>> = vec![
            BoundarySpec {
                side: BoundarySide::Left,
                condition: BoundaryCondition::Periodic,
            },
            BoundarySpec {
                side: BoundarySide::Right,
                condition: BoundaryCondition::Periodic,
            },
        ];
        let sides = side_conditions_2d(&bcs).unwrap();
        assert_eq!(sides[0], SideBc::Periodic);
        assert_eq!(sides[1], SideBc::Periodic);
        // y-sides default to Dirichlet (paired trivially).
        assert_eq!(sides[2], SideBc::Dirichlet);
        assert_eq!(sides[3], SideBc::Dirichlet);
    }

    #[test]
    fn test_side_conditions_periodic_unpaired_errors() {
        // Periodic on Left but not Right must be rejected.
        let bcs: Vec<BoundarySpec<CpuRuntime>> = vec![BoundarySpec {
            side: BoundarySide::Left,
            condition: BoundaryCondition::Periodic,
        }];
        assert!(side_conditions_2d(&bcs).is_err());
    }

    // ------------------------------------------------------------------
    // extract_boundary_values_2d (lenient: skips Neumann/Periodic)
    // ------------------------------------------------------------------

    #[test]
    fn test_extract_values_skips_non_dirichlet() {
        // Neumann/Periodic specs carry no nodal value → returns Ok (all zeros).
        let bcs = vec![BoundarySpec {
            side: BoundarySide::All,
            condition: BoundaryCondition::Neumann(scalar_tensor(3.0)),
        }];
        let values = extract_boundary_values_2d(&bcs, 5, 5).unwrap();
        assert!(values.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_extract_values_dirichlet_side() {
        // Left Dirichlet = 7.0 on a 4x3 grid fills the left column (j = 0..ny).
        let bcs = vec![BoundarySpec {
            side: BoundarySide::Left,
            condition: BoundaryCondition::Dirichlet(Tensor::<CpuRuntime>::from_slice(
                &[7.0, 7.0, 7.0],
                &[3],
                &device(),
            )),
        }];
        let (nx, ny) = (4, 3);
        let values = extract_boundary_values_2d(&bcs, nx, ny).unwrap();
        for (j, &v) in values.iter().take(ny).enumerate() {
            assert_eq!(v, 7.0, "left column node j={j}");
        }
    }
}

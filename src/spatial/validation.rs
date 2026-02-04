//! Validation helpers for spatial algorithms.

use numr::dtype::DType;
use numr::error::{Error, Result};

/// Validate point set dtype (must be F32 or F64).
pub fn validate_points_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Validate that points are 2D [n, d].
pub fn validate_points_2d(shape: &[usize], op: &'static str) -> Result<()> {
    if shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: format!("{op} requires 2D point set [n, d], got {}-D", shape.len()),
        });
    }
    Ok(())
}

/// Validate that two point sets have matching dimensionality.
pub fn validate_matching_dims(
    x_shape: &[usize],
    y_shape: &[usize],
    op: &'static str,
) -> Result<()> {
    if x_shape.len() < 2 || y_shape.len() < 2 {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: format!("{op} requires 2D point sets"),
        });
    }
    if x_shape[1] != y_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "points",
            reason: format!(
                "{op} requires matching dimensions: x has {} dims, y has {} dims",
                x_shape[1], y_shape[1]
            ),
        });
    }
    Ok(())
}

/// Validate k parameter for k-nearest neighbors.
pub fn validate_k(k: usize, n_points: usize, op: &'static str) -> Result<()> {
    if k == 0 {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("{op} requires k > 0"),
        });
    }
    if k > n_points {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!("{op}: k={k} exceeds number of points {n_points}"),
        });
    }
    Ok(())
}

/// Validate radius parameter for radius search.
pub fn validate_radius(radius: f64, op: &'static str) -> Result<()> {
    if radius < 0.0 {
        return Err(Error::InvalidArgument {
            arg: "radius",
            reason: format!("{op} requires radius >= 0, got {radius}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_points_dtype() {
        assert!(validate_points_dtype(DType::F32, "test").is_ok());
        assert!(validate_points_dtype(DType::F64, "test").is_ok());
        assert!(validate_points_dtype(DType::I32, "test").is_err());
    }

    #[test]
    fn test_validate_points_2d() {
        assert!(validate_points_2d(&[10, 3], "test").is_ok());
        assert!(validate_points_2d(&[10], "test").is_err());
        assert!(validate_points_2d(&[10, 3, 2], "test").is_err());
    }

    #[test]
    fn test_validate_k() {
        assert!(validate_k(5, 100, "test").is_ok());
        assert!(validate_k(0, 100, "test").is_err());
        assert!(validate_k(101, 100, "test").is_err());
    }
}

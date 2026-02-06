//! Validation helpers for clustering algorithms.

use numr::dtype::DType;
use numr::error::{Error, Result};

/// Validate point set dtype (must be F32 or F64).
pub fn validate_cluster_dtype(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::F64 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Validate that data is 2D [n, d].
pub fn validate_data_2d(shape: &[usize], op: &'static str) -> Result<()> {
    if shape.len() != 2 {
        return Err(Error::InvalidArgument {
            arg: "data",
            reason: format!("{op} requires 2D data [n, d], got {}-D", shape.len()),
        });
    }
    if shape[0] == 0 {
        return Err(Error::InvalidArgument {
            arg: "data",
            reason: format!("{op} requires at least 1 data point"),
        });
    }
    Ok(())
}

/// Validate n_clusters parameter.
pub fn validate_n_clusters(n_clusters: usize, n_points: usize, op: &'static str) -> Result<()> {
    if n_clusters == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_clusters",
            reason: format!("{op} requires n_clusters > 0"),
        });
    }
    if n_clusters > n_points {
        return Err(Error::InvalidArgument {
            arg: "n_clusters",
            reason: format!("{op}: n_clusters={n_clusters} exceeds number of points {n_points}"),
        });
    }
    Ok(())
}

/// Validate eps parameter (must be positive).
pub fn validate_eps(eps: f64, op: &'static str) -> Result<()> {
    if eps <= 0.0 || !eps.is_finite() {
        return Err(Error::InvalidArgument {
            arg: "eps",
            reason: format!("{op} requires finite eps > 0, got {eps}"),
        });
    }
    Ok(())
}

/// Validate min_samples parameter.
pub fn validate_min_samples(min_samples: usize, op: &'static str) -> Result<()> {
    if min_samples == 0 {
        return Err(Error::InvalidArgument {
            arg: "min_samples",
            reason: format!("{op} requires min_samples > 0"),
        });
    }
    Ok(())
}

/// Validate labels tensor is 1D I64.
pub fn validate_labels(shape: &[usize], dtype: DType, op: &'static str) -> Result<()> {
    if shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "labels",
            reason: format!("{op} requires 1D labels, got {}-D", shape.len()),
        });
    }
    if dtype != DType::I64 {
        return Err(Error::InvalidArgument {
            arg: "labels",
            reason: format!("{op} requires I64 labels, got {dtype:?}"),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_cluster_dtype() {
        assert!(validate_cluster_dtype(DType::F32, "test").is_ok());
        assert!(validate_cluster_dtype(DType::F64, "test").is_ok());
        assert!(validate_cluster_dtype(DType::I32, "test").is_err());
    }

    #[test]
    fn test_validate_data_2d() {
        assert!(validate_data_2d(&[10, 3], "test").is_ok());
        assert!(validate_data_2d(&[10], "test").is_err());
        assert!(validate_data_2d(&[0, 3], "test").is_err());
    }

    #[test]
    fn test_validate_n_clusters() {
        assert!(validate_n_clusters(3, 100, "test").is_ok());
        assert!(validate_n_clusters(0, 100, "test").is_err());
        assert!(validate_n_clusters(101, 100, "test").is_err());
    }

    #[test]
    fn test_validate_eps() {
        assert!(validate_eps(0.5, "test").is_ok());
        assert!(validate_eps(0.0, "test").is_err());
        assert!(validate_eps(-1.0, "test").is_err());
        assert!(validate_eps(f64::INFINITY, "test").is_err());
    }
}

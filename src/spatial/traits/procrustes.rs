//! Procrustes analysis trait.
//!
//! Procrustes analysis finds the optimal transformation (rotation, translation,
//! and optionally scaling) that best aligns two sets of corresponding points.
//! Uses the Kabsch algorithm (SVD-based) for finding the optimal rotation.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::rotation::Rotation;

/// Result of Procrustes analysis.
#[derive(Debug, Clone)]
pub struct ProcrustesResult<R: Runtime> {
    /// Optimal rotation to align source to target.
    pub rotation: Rotation<R>,

    /// Optimal translation vector [d].
    /// Applied after rotation: target ≈ scale * R @ source + translation
    pub translation: Tensor<R>,

    /// Optimal scaling factor (1.0 if scaling was disabled).
    pub scale: f64,

    /// Transformed source points [n, d].
    /// Equals scale * R @ source + translation
    pub transformed: Tensor<R>,

    /// Residual sum of squares after transformation.
    /// Lower is better.
    pub disparity: f64,
}

/// Algorithmic contract for Procrustes analysis.
///
/// All backends implementing Procrustes algorithms MUST implement this trait.
pub trait ProcrustesAlgorithms<R: Runtime> {
    /// Compute the optimal rotation, translation, and optional scaling
    /// to align source points to target points.
    ///
    /// # Arguments
    ///
    /// * `source` - Source point set with shape (n, d)
    /// * `target` - Target point set with shape (n, d)
    /// * `scaling` - If true, also compute optimal scaling
    /// * `reflection` - If true, allow improper rotations (reflections)
    ///
    /// # Returns
    ///
    /// ProcrustesResult containing the optimal transformation.
    ///
    /// # Algorithm
    ///
    /// Uses the Kabsch algorithm:
    /// 1. Center both point sets by subtracting their centroids
    /// 2. Compute cross-covariance matrix H = source_centered.T @ target_centered
    /// 3. SVD: H = U @ S @ Vt
    /// 4. Optimal rotation: R = V @ U.T
    /// 5. Handle reflection: if det(R) < 0 and reflection=false, flip sign
    /// 6. If scaling: scale = trace(S) / ||source_centered||²
    /// 7. Translation: t = target_centroid - scale * R @ source_centroid
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // Align two corresponding point sets
    /// let source = Tensor::from_slice(&[0.0, 0.0, 1.0, 0.0, 0.5, 1.0], &[3, 2], &device);
    /// let target = Tensor::from_slice(&[1.0, 1.0, 2.0, 1.0, 1.5, 2.0], &[3, 2], &device);
    ///
    /// let result = client.procrustes(&source, &target, true, false)?;
    /// // result.transformed is close to target
    /// ```
    fn procrustes(
        &self,
        source: &Tensor<R>,
        target: &Tensor<R>,
        scaling: bool,
        reflection: bool,
    ) -> Result<ProcrustesResult<R>>;

    /// Orthogonal Procrustes: find the orthogonal matrix R that minimizes
    /// ||A @ R - B||_F (Frobenius norm).
    ///
    /// # Arguments
    ///
    /// * `a` - Matrix A with shape (m, n)
    /// * `b` - Matrix B with shape (m, n)
    ///
    /// # Returns
    ///
    /// Orthogonal matrix R with shape (n, n) and residual norm.
    ///
    /// # Algorithm
    ///
    /// SVD of A.T @ B = U @ S @ Vt, then R = V @ U.T
    fn orthogonal_procrustes(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<(Tensor<R>, f64)>;
}

#[cfg(test)]
mod tests {
    // Tests will be in the implementation files
}

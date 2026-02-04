//! Rotation representation and operations trait.
//!
//! Provides quaternion-based rotations with conversions to/from rotation matrices
//! and Euler angles. Quaternions are used internally for numerical stability and
//! efficient composition.

use numr::error::Result;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Euler angle convention specifying the order of rotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EulerOrder {
    /// Intrinsic rotations: X then Y then Z (most common in aerospace)
    #[default]
    XYZ,
    /// Intrinsic rotations: X then Z then Y
    XZY,
    /// Intrinsic rotations: Y then X then Z
    YXZ,
    /// Intrinsic rotations: Y then Z then X
    YZX,
    /// Intrinsic rotations: Z then X then Y
    ZXY,
    /// Intrinsic rotations: Z then Y then X (yaw-pitch-roll in aerospace)
    ZYX,
}

/// Rotation represented internally as unit quaternions.
///
/// Quaternions are stored as [w, x, y, z] where w is the scalar part.
#[derive(Debug, Clone)]
pub struct Rotation<R: Runtime> {
    /// Unit quaternions [n, 4] where each row is [w, x, y, z].
    /// For a single rotation, shape is [4].
    /// For batch rotations, shape is [n, 4].
    pub quaternions: Tensor<R>,

    /// Whether this represents a batch of rotations.
    pub is_batch: bool,
}

/// Algorithmic contract for rotation operations.
///
/// All backends implementing rotation algorithms MUST implement this trait.
pub trait RotationAlgorithms<R: Runtime> {
    /// Create a rotation from a quaternion [w, x, y, z].
    ///
    /// # Arguments
    ///
    /// * `quaternion` - Quaternion(s) with shape [4] or [n, 4]
    ///
    /// # Returns
    ///
    /// Rotation object (quaternion will be normalized).
    fn rotation_from_quat(&self, quaternion: &Tensor<R>) -> Result<Rotation<R>>;

    /// Create a rotation from a 3x3 rotation matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Rotation matrix with shape [3, 3] or [n, 3, 3]
    ///
    /// # Returns
    ///
    /// Rotation object converted from the matrix.
    fn rotation_from_matrix(&self, matrix: &Tensor<R>) -> Result<Rotation<R>>;

    /// Create a rotation from Euler angles.
    ///
    /// # Arguments
    ///
    /// * `angles` - Euler angles [α, β, γ] in radians with shape [3] or [n, 3]
    /// * `order` - Order of rotations (e.g., XYZ, ZYX)
    ///
    /// # Returns
    ///
    /// Rotation object.
    fn rotation_from_euler(&self, angles: &Tensor<R>, order: EulerOrder) -> Result<Rotation<R>>;

    /// Create a rotation from axis-angle representation.
    ///
    /// # Arguments
    ///
    /// * `axis` - Unit rotation axis with shape [3] or [n, 3]
    /// * `angle` - Rotation angle in radians with shape [] or [n]
    ///
    /// # Returns
    ///
    /// Rotation object.
    fn rotation_from_axis_angle(&self, axis: &Tensor<R>, angle: &Tensor<R>) -> Result<Rotation<R>>;

    /// Create a rotation from a rotation vector (Rodrigues vector).
    ///
    /// The rotation vector is axis * angle (axis scaled by angle).
    ///
    /// # Arguments
    ///
    /// * `rotvec` - Rotation vector with shape [3] or [n, 3]
    fn rotation_from_rotvec(&self, rotvec: &Tensor<R>) -> Result<Rotation<R>>;

    /// Convert rotation to quaternion representation.
    ///
    /// Returns quaternion(s) as [w, x, y, z] with shape [4] or [n, 4].
    fn rotation_as_quat(&self, rot: &Rotation<R>) -> Result<Tensor<R>>;

    /// Convert rotation to 3x3 rotation matrix.
    ///
    /// Returns matrix with shape [3, 3] or [n, 3, 3].
    fn rotation_as_matrix(&self, rot: &Rotation<R>) -> Result<Tensor<R>>;

    /// Convert rotation to Euler angles.
    ///
    /// # Arguments
    ///
    /// * `rot` - The rotation
    /// * `order` - Euler angle convention
    ///
    /// # Returns
    ///
    /// Euler angles [α, β, γ] in radians with shape [3] or [n, 3].
    fn rotation_as_euler(&self, rot: &Rotation<R>, order: EulerOrder) -> Result<Tensor<R>>;

    /// Convert rotation to rotation vector (Rodrigues representation).
    ///
    /// Returns rotation vector with shape [3] or [n, 3].
    fn rotation_as_rotvec(&self, rot: &Rotation<R>) -> Result<Tensor<R>>;

    /// Apply rotation to vectors.
    ///
    /// # Arguments
    ///
    /// * `rot` - The rotation
    /// * `vectors` - Vectors to rotate with shape [3], [m, 3], or compatible batch shape
    ///
    /// # Returns
    ///
    /// Rotated vectors with same shape as input.
    fn rotation_apply(&self, rot: &Rotation<R>, vectors: &Tensor<R>) -> Result<Tensor<R>>;

    /// Compose two rotations (multiplication).
    ///
    /// r_composed represents applying r2 first, then r1: r_composed(v) = r1(r2(v))
    ///
    /// # Arguments
    ///
    /// * `r1` - First (outer) rotation
    /// * `r2` - Second (inner) rotation
    ///
    /// # Returns
    ///
    /// Composed rotation.
    fn rotation_compose(&self, r1: &Rotation<R>, r2: &Rotation<R>) -> Result<Rotation<R>>;

    /// Compute the inverse rotation.
    ///
    /// r_inv(r(v)) = v
    fn rotation_inverse(&self, rot: &Rotation<R>) -> Result<Rotation<R>>;

    /// Spherical linear interpolation (slerp) between rotations.
    ///
    /// # Arguments
    ///
    /// * `r1` - Start rotation
    /// * `r2` - End rotation
    /// * `t` - Interpolation parameter in [0, 1]. t=0 gives r1, t=1 gives r2.
    ///
    /// # Returns
    ///
    /// Interpolated rotation.
    fn rotation_slerp(&self, r1: &Rotation<R>, r2: &Rotation<R>, t: f64) -> Result<Rotation<R>>;

    /// Create identity rotation(s).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of identity rotations to create (None for single rotation)
    fn rotation_identity(&self, n: Option<usize>) -> Result<Rotation<R>>;

    /// Create random rotation(s) uniformly sampled from SO(3).
    ///
    /// # Arguments
    ///
    /// * `n` - Number of random rotations (None for single rotation)
    fn rotation_random(&self, n: Option<usize>) -> Result<Rotation<R>>;

    /// Compute the rotation angle (magnitude of rotation).
    ///
    /// Returns angle in radians with shape [] or [n].
    fn rotation_magnitude(&self, rot: &Rotation<R>) -> Result<Tensor<R>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euler_order() {
        assert_eq!(EulerOrder::default(), EulerOrder::XYZ);
    }
}

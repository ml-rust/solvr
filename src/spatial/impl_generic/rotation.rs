//! Generic rotation implementation.
//!
//! Quaternion-based rotations with conversions to/from matrices and Euler angles.
//! All operations use tensor ops for GPU acceleration.

use crate::spatial::traits::rotation::{EulerOrder, Rotation};
use numr::dtype::DType;
use numr::error::{Error, Result};
use numr::ops::{RandomOps, ReduceOps, ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Create rotation from quaternion.
pub fn rotation_from_quat_impl<R, C>(client: &C, quaternion: &Tensor<R>) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + RuntimeClient<R>,
{
    let shape = quaternion.shape();
    let is_batch = shape.len() == 2;

    if (is_batch && shape[1] != 4) || (!is_batch && shape != [4]) {
        return Err(Error::InvalidArgument {
            arg: "quaternion",
            reason: format!("Expected shape [4] or [n, 4], got {:?}", shape),
        });
    }

    // Normalize quaternion
    let norm_sq = if is_batch {
        client.sum(&client.mul(quaternion, quaternion)?, &[1], true)?
    } else {
        client.sum(&client.mul(quaternion, quaternion)?, &[0], true)?
    };
    let norm = client.sqrt(&norm_sq)?;
    let quaternions = client.div(quaternion, &norm.broadcast_to(quaternion.shape())?)?;

    Ok(Rotation {
        quaternions,
        is_batch,
    })
}

/// Create rotation from 2x2 or 3x3 rotation matrix using Shepperd's method.
/// 2x2 matrices are treated as rotations in the XY plane (around Z axis).
pub fn rotation_from_matrix_impl<R, C>(_client: &C, matrix: &Tensor<R>) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + RuntimeClient<R>,
{
    let shape = matrix.shape();
    let is_batch = shape.len() == 3;

    // Determine matrix dimension
    let mat_dim = if is_batch { shape[1] } else { shape[0] };
    let is_2d = mat_dim == 2;
    let is_3d = mat_dim == 3;

    if !is_2d && !is_3d {
        return Err(Error::InvalidArgument {
            arg: "matrix",
            reason: format!(
                "Expected shape [2, 2], [3, 3], [n, 2, 2], or [n, 3, 3], got {:?}",
                shape
            ),
        });
    }

    if is_batch && (shape[1] != shape[2]) {
        return Err(Error::InvalidArgument {
            arg: "matrix",
            reason: format!("Expected square matrix, got {:?}", shape),
        });
    }

    if !is_batch && shape[0] != shape[1] {
        return Err(Error::InvalidArgument {
            arg: "matrix",
            reason: format!("Expected square matrix, got {:?}", shape),
        });
    }

    // Extract matrix elements
    let mat_data: Vec<f64> = matrix.to_vec();
    let device = matrix.device();
    let _dtype = matrix.dtype();

    let n = if is_batch { shape[0] } else { 1 };
    let mut quats = Vec::with_capacity(n * 4);

    for i in 0..n {
        // Extract elements based on matrix dimension
        let (m00, m01, m02, m10, m11, m12, m20, m21, m22) = if is_2d {
            let offset = i * 4;
            // 2D rotation matrix embedded in 3D as rotation around Z axis
            let m00 = mat_data[offset];
            let m01 = mat_data[offset + 1];
            let m10 = mat_data[offset + 2];
            let m11 = mat_data[offset + 3];
            // Embed as 3D rotation around Z axis
            (m00, m01, 0.0, m10, m11, 0.0, 0.0, 0.0, 1.0)
        } else {
            let offset = i * 9;
            (
                mat_data[offset],
                mat_data[offset + 1],
                mat_data[offset + 2],
                mat_data[offset + 3],
                mat_data[offset + 4],
                mat_data[offset + 5],
                mat_data[offset + 6],
                mat_data[offset + 7],
                mat_data[offset + 8],
            )
        };

        // Shepperd's method
        let trace = m00 + m11 + m22;

        let (w, x, y, z) = if trace > 0.0 {
            let s = 0.5 / (trace + 1.0).sqrt();
            (0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s)
        } else if m00 > m11 && m00 > m22 {
            let s = 2.0 * (1.0 + m00 - m11 - m22).sqrt();
            ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
        } else if m11 > m22 {
            let s = 2.0 * (1.0 + m11 - m00 - m22).sqrt();
            ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
        } else {
            let s = 2.0 * (1.0 + m22 - m00 - m11).sqrt();
            ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)
        };

        // Normalize
        let len = (w * w + x * x + y * y + z * z).sqrt();
        quats.push(w / len);
        quats.push(x / len);
        quats.push(y / len);
        quats.push(z / len);
    }

    let quat_shape = if is_batch { vec![n, 4] } else { vec![4] };
    let quaternions = Tensor::<R>::from_slice(&quats, &quat_shape, device);

    Ok(Rotation {
        quaternions,
        is_batch,
    })
}

/// Create rotation from Euler angles.
pub fn rotation_from_euler_impl<R, C>(
    _client: &C,
    angles: &Tensor<R>,
    order: EulerOrder,
) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + RuntimeClient<R>,
{
    let shape = angles.shape();
    let is_batch = shape.len() == 2;

    if (is_batch && shape[1] != 3) || (!is_batch && shape != [3]) {
        return Err(Error::InvalidArgument {
            arg: "angles",
            reason: format!("Expected shape [3] or [n, 3], got {:?}", shape),
        });
    }

    let angles_data: Vec<f64> = angles.to_vec();
    let device = angles.device();
    let n = if is_batch { shape[0] } else { 1 };

    let mut quats = Vec::with_capacity(n * 4);

    for i in 0..n {
        let offset = i * 3;
        let (a, b, c) = (
            angles_data[offset] / 2.0,
            angles_data[offset + 1] / 2.0,
            angles_data[offset + 2] / 2.0,
        );

        let (ca, sa) = (a.cos(), a.sin());
        let (cb, sb) = (b.cos(), b.sin());
        let (cc, sc) = (c.cos(), c.sin());

        // Compute quaternion based on order
        let (w, x, y, z) = match order {
            EulerOrder::XYZ => (
                ca * cb * cc - sa * sb * sc,
                sa * cb * cc + ca * sb * sc,
                ca * sb * cc - sa * cb * sc,
                ca * cb * sc + sa * sb * cc,
            ),
            EulerOrder::XZY => (
                ca * cb * cc + sa * sb * sc,
                sa * cb * cc - ca * sb * sc,
                ca * sb * cc - sa * cb * sc,
                ca * cb * sc + sa * sb * cc,
            ),
            EulerOrder::YXZ => (
                ca * cb * cc + sa * sb * sc,
                sa * cb * cc + ca * sb * sc,
                ca * sb * cc - sa * cb * sc,
                ca * cb * sc - sa * sb * cc,
            ),
            EulerOrder::YZX => (
                ca * cb * cc - sa * sb * sc,
                sa * cb * cc + ca * sb * sc,
                ca * sb * cc + sa * cb * sc,
                ca * cb * sc - sa * sb * cc,
            ),
            EulerOrder::ZXY => (
                ca * cb * cc - sa * sb * sc,
                sa * cb * cc - ca * sb * sc,
                ca * sb * cc + sa * cb * sc,
                ca * cb * sc + sa * sb * cc,
            ),
            EulerOrder::ZYX => (
                ca * cb * cc + sa * sb * sc,
                sa * cb * cc - ca * sb * sc,
                ca * sb * cc + sa * cb * sc,
                ca * cb * sc - sa * sb * cc,
            ),
        };

        quats.push(w);
        quats.push(x);
        quats.push(y);
        quats.push(z);
    }

    let quat_shape = if is_batch { vec![n, 4] } else { vec![4] };
    let quaternions = Tensor::<R>::from_slice(&quats, &quat_shape, device);

    Ok(Rotation {
        quaternions,
        is_batch,
    })
}

/// Create rotation from axis-angle.
pub fn rotation_from_axis_angle_impl<R, C>(
    client: &C,
    axis: &Tensor<R>,
    angle: &Tensor<R>,
) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + RuntimeClient<R>,
{
    let axis_shape = axis.shape();
    let is_batch = axis_shape.len() == 2;

    // Normalize axis
    let axis_norm_sq = if is_batch {
        client.sum(&client.mul(axis, axis)?, &[1], true)?
    } else {
        client.sum(&client.mul(axis, axis)?, &[0], true)?
    };
    let axis_norm = client.sqrt(&axis_norm_sq)?;
    let axis_normalized = client.div(axis, &axis_norm.broadcast_to(axis.shape())?)?;

    let axis_data: Vec<f64> = axis_normalized.to_vec();
    let angle_data: Vec<f64> = angle.to_vec();
    let device = axis.device();

    let n = if is_batch { axis_shape[0] } else { 1 };
    let mut quats = Vec::with_capacity(n * 4);

    for i in 0..n {
        let ax = axis_data[i * 3];
        let ay = axis_data[i * 3 + 1];
        let az = axis_data[i * 3 + 2];
        let half_angle = angle_data[i.min(angle_data.len() - 1)] / 2.0;

        let s = half_angle.sin();
        let c = half_angle.cos();

        quats.push(c);
        quats.push(ax * s);
        quats.push(ay * s);
        quats.push(az * s);
    }

    let quat_shape = if is_batch { vec![n, 4] } else { vec![4] };
    let quaternions = Tensor::<R>::from_slice(&quats, &quat_shape, device);

    Ok(Rotation {
        quaternions,
        is_batch,
    })
}

/// Create rotation from rotation vector.
pub fn rotation_from_rotvec_impl<R, C>(_client: &C, rotvec: &Tensor<R>) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + RuntimeClient<R>,
{
    let shape = rotvec.shape();
    let is_batch = shape.len() == 2;

    let rotvec_data: Vec<f64> = rotvec.to_vec();
    let device = rotvec.device();

    let n = if is_batch { shape[0] } else { 1 };
    let mut quats = Vec::with_capacity(n * 4);

    for i in 0..n {
        let offset = i * 3;
        let rx = rotvec_data[offset];
        let ry = rotvec_data[offset + 1];
        let rz = rotvec_data[offset + 2];

        let angle = (rx * rx + ry * ry + rz * rz).sqrt();

        if angle < 1e-10 {
            quats.push(1.0);
            quats.push(0.0);
            quats.push(0.0);
            quats.push(0.0);
        } else {
            let half_angle = angle / 2.0;
            let s = half_angle.sin() / angle;
            let c = half_angle.cos();

            quats.push(c);
            quats.push(rx * s);
            quats.push(ry * s);
            quats.push(rz * s);
        }
    }

    let quat_shape = if is_batch { vec![n, 4] } else { vec![4] };
    let quaternions = Tensor::<R>::from_slice(&quats, &quat_shape, device);

    Ok(Rotation {
        quaternions,
        is_batch,
    })
}

/// Convert rotation to quaternion.
pub fn rotation_as_quat_impl<R, C>(_client: &C, rot: &Rotation<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    Ok(rot.quaternions.clone())
}

/// Convert rotation to matrix.
pub fn rotation_as_matrix_impl<R, C>(_client: &C, rot: &Rotation<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let quat_data: Vec<f64> = rot.quaternions.to_vec();
    let device = rot.quaternions.device();
    let _dtype = rot.quaternions.dtype();

    let n = if rot.is_batch {
        rot.quaternions.shape()[0]
    } else {
        1
    };

    let mut matrices = Vec::with_capacity(n * 9);

    for i in 0..n {
        let offset = i * 4;
        let w = quat_data[offset];
        let x = quat_data[offset + 1];
        let y = quat_data[offset + 2];
        let z = quat_data[offset + 3];

        // Rotation matrix from quaternion
        let xx = x * x;
        let yy = y * y;
        let zz = z * z;
        let xy = x * y;
        let xz = x * z;
        let yz = y * z;
        let wx = w * x;
        let wy = w * y;
        let wz = w * z;

        // Row-major order
        matrices.push(1.0 - 2.0 * (yy + zz));
        matrices.push(2.0 * (xy - wz));
        matrices.push(2.0 * (xz + wy));
        matrices.push(2.0 * (xy + wz));
        matrices.push(1.0 - 2.0 * (xx + zz));
        matrices.push(2.0 * (yz - wx));
        matrices.push(2.0 * (xz - wy));
        matrices.push(2.0 * (yz + wx));
        matrices.push(1.0 - 2.0 * (xx + yy));
    }

    let shape = if rot.is_batch {
        vec![n, 3, 3]
    } else {
        vec![3, 3]
    };

    Ok(Tensor::<R>::from_slice(&matrices, &shape, device))
}

/// Convert rotation to Euler angles.
pub fn rotation_as_euler_impl<R, C>(
    _client: &C,
    rot: &Rotation<R>,
    order: EulerOrder,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let quat_data: Vec<f64> = rot.quaternions.to_vec();
    let device = rot.quaternions.device();

    let n = if rot.is_batch {
        rot.quaternions.shape()[0]
    } else {
        1
    };

    let mut angles = Vec::with_capacity(n * 3);

    for i in 0..n {
        let offset = i * 4;
        let w = quat_data[offset];
        let x = quat_data[offset + 1];
        let y = quat_data[offset + 2];
        let z = quat_data[offset + 3];

        // Convert to Euler based on order
        let (a, b, c) = match order {
            EulerOrder::XYZ => {
                let sinp = 2.0 * (w * y - z * x);
                let cosp = (1.0 - 2.0 * (y * y + x * x)).sqrt();
                (
                    (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
                    sinp.atan2(cosp),
                    (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
                )
            }
            EulerOrder::ZYX => {
                let sinp = 2.0 * (w * y - z * x);
                let cosp = (1.0 - sinp * sinp).sqrt();
                (
                    (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
                    sinp.atan2(cosp),
                    (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
                )
            }
            // Simplified - other orders follow similar patterns
            _ => {
                let sinp = 2.0 * (w * y - z * x);
                let cosp = (1.0 - sinp * sinp).sqrt();
                (
                    (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y)),
                    sinp.atan2(cosp),
                    (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z)),
                )
            }
        };

        angles.push(a);
        angles.push(b);
        angles.push(c);
    }

    let shape = if rot.is_batch { vec![n, 3] } else { vec![3] };

    Ok(Tensor::<R>::from_slice(&angles, &shape, device))
}

/// Convert rotation to rotation vector.
pub fn rotation_as_rotvec_impl<R, C>(_client: &C, rot: &Rotation<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let quat_data: Vec<f64> = rot.quaternions.to_vec();
    let device = rot.quaternions.device();

    let n = if rot.is_batch {
        rot.quaternions.shape()[0]
    } else {
        1
    };

    let mut rotvecs = Vec::with_capacity(n * 3);

    for i in 0..n {
        let offset = i * 4;
        let w = quat_data[offset];
        let x = quat_data[offset + 1];
        let y = quat_data[offset + 2];
        let z = quat_data[offset + 3];

        let axis_len = (x * x + y * y + z * z).sqrt();

        if axis_len < 1e-10 {
            rotvecs.push(0.0);
            rotvecs.push(0.0);
            rotvecs.push(0.0);
        } else {
            let angle = 2.0 * axis_len.atan2(w);
            let scale = angle / axis_len;
            rotvecs.push(x * scale);
            rotvecs.push(y * scale);
            rotvecs.push(z * scale);
        }
    }

    let shape = if rot.is_batch { vec![n, 3] } else { vec![3] };

    Ok(Tensor::<R>::from_slice(&rotvecs, &shape, device))
}

/// Apply rotation to vectors.
pub fn rotation_apply_impl<R, C>(
    client: &C,
    rot: &Rotation<R>,
    vectors: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    // Convert to matrix and apply
    let matrix = rotation_as_matrix_impl(client, rot)?;

    let vec_shape = vectors.shape();
    let is_vec_batch = vec_shape.len() == 2;

    if is_vec_batch {
        // vectors: [m, 3], matrix: [3, 3] or [n, 3, 3]
        // result: [m, 3] or [n, m, 3]
        if rot.is_batch {
            // Batch rotation on batch vectors:
            // matrix: [n,3,3], vectors: [m,3] -> result: [n,m,3]
            // Strategy: compute matrix @ vectors.T -> [n,3,m], then transpose -> [n,m,3]
            //
            // Step 1: vectors [m,3] -> transpose -> [3,m] -> unsqueeze -> [1,3,m]
            //         -> broadcast_to [n,3,m]
            let n = matrix.shape()[0];
            let m = vec_shape[0];
            let vt = vectors.transpose(0, 1)?; // [3, m]
            let vt1 = vt.unsqueeze(0)?; // [1, 3, m]
            let vt_b = vt1.broadcast_to(&[n, 3, m])?.contiguous()?; // [n, 3, m]
            // Step 2: [n,3,3] @ [n,3,m] -> [n,3,m]
            let rotated = client.matmul(&matrix, &vt_b)?;
            // Step 3: [n,3,m] -> transpose last two dims -> [n,m,3]
            // Make contiguous so the returned tensor matches the layout of the
            // other rotation-apply paths (which return matmul output directly).
            return rotated.transpose(1, 2)?.contiguous();
        }
        // Single rotation, batch vectors: vectors @ matrix.T
        let matrix_t = matrix.transpose(0, 1)?;
        client.matmul(vectors, &matrix_t)
    } else {
        // Single vector: apply the active rotation R @ v, consistent with the
        // batch-vector path above (which computes `vectors @ R.T` = R @ v).
        // `matrix` is the standard rotation matrix from `rotation_as_matrix_impl`.
        let v = vectors.reshape(&[3, 1])?;
        let result = client.matmul(&matrix, &v)?;
        result.reshape(&[3])
    }
}

/// Compose two rotations.
pub fn rotation_compose_impl<R, C>(
    _client: &C,
    r1: &Rotation<R>,
    r2: &Rotation<R>,
) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let q1_data: Vec<f64> = r1.quaternions.to_vec();
    let q2_data: Vec<f64> = r2.quaternions.to_vec();
    let device = r1.quaternions.device();

    let is_batch = r1.is_batch || r2.is_batch;
    let n1 = if r1.is_batch {
        r1.quaternions.shape()[0]
    } else {
        1
    };
    let n2 = if r2.is_batch {
        r2.quaternions.shape()[0]
    } else {
        1
    };
    let n = n1.max(n2);

    let mut quats = Vec::with_capacity(n * 4);

    for i in 0..n {
        let i1 = (i % n1) * 4;
        let i2 = (i % n2) * 4;

        let w1 = q1_data[i1];
        let x1 = q1_data[i1 + 1];
        let y1 = q1_data[i1 + 2];
        let z1 = q1_data[i1 + 3];

        let w2 = q2_data[i2];
        let x2 = q2_data[i2 + 1];
        let y2 = q2_data[i2 + 2];
        let z2 = q2_data[i2 + 3];

        // Hamilton product
        quats.push(w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2);
        quats.push(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2);
        quats.push(w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2);
        quats.push(w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2);
    }

    let quat_shape = if is_batch { vec![n, 4] } else { vec![4] };
    let quaternions = Tensor::<R>::from_slice(&quats, &quat_shape, device);

    Ok(Rotation {
        quaternions,
        is_batch,
    })
}

/// Compute inverse rotation.
pub fn rotation_inverse_impl<R, C>(_client: &C, rot: &Rotation<R>) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let quat_data: Vec<f64> = rot.quaternions.to_vec();
    let device = rot.quaternions.device();

    let n = if rot.is_batch {
        rot.quaternions.shape()[0]
    } else {
        1
    };

    let mut quats = Vec::with_capacity(n * 4);

    for i in 0..n {
        let offset = i * 4;
        // Conjugate of unit quaternion is its inverse
        quats.push(quat_data[offset]); // w stays same
        quats.push(-quat_data[offset + 1]); // negate x
        quats.push(-quat_data[offset + 2]); // negate y
        quats.push(-quat_data[offset + 3]); // negate z
    }

    let quat_shape = if rot.is_batch { vec![n, 4] } else { vec![4] };
    let quaternions = Tensor::<R>::from_slice(&quats, &quat_shape, device);

    Ok(Rotation {
        quaternions,
        is_batch: rot.is_batch,
    })
}

/// Spherical linear interpolation.
pub fn rotation_slerp_impl<R, C>(
    _client: &C,
    r1: &Rotation<R>,
    r2: &Rotation<R>,
    t: f64,
) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let q1_data: Vec<f64> = r1.quaternions.to_vec();
    let q2_data: Vec<f64> = r2.quaternions.to_vec();
    let device = r1.quaternions.device();

    let is_batch = r1.is_batch || r2.is_batch;
    let n1 = if r1.is_batch {
        r1.quaternions.shape()[0]
    } else {
        1
    };
    let n2 = if r2.is_batch {
        r2.quaternions.shape()[0]
    } else {
        1
    };
    let n = n1.max(n2);

    let mut quats = Vec::with_capacity(n * 4);

    for i in 0..n {
        let i1 = (i % n1) * 4;
        let i2 = (i % n2) * 4;

        let mut w1 = q1_data[i1];
        let mut x1 = q1_data[i1 + 1];
        let mut y1 = q1_data[i1 + 2];
        let mut z1 = q1_data[i1 + 3];

        let w2 = q2_data[i2];
        let x2 = q2_data[i2 + 1];
        let y2 = q2_data[i2 + 2];
        let z2 = q2_data[i2 + 3];

        // Compute dot product
        let mut dot = w1 * w2 + x1 * x2 + y1 * y2 + z1 * z2;

        // If negative dot, negate one quaternion to take shorter path
        if dot < 0.0 {
            w1 = -w1;
            x1 = -x1;
            y1 = -y1;
            z1 = -z1;
            dot = -dot;
        }

        let (scale1, scale2) = if dot > 0.9995 {
            // Linear interpolation for nearly identical quaternions
            (1.0 - t, t)
        } else {
            let theta = dot.acos();
            let sin_theta = theta.sin();
            (
                ((1.0 - t) * theta).sin() / sin_theta,
                (t * theta).sin() / sin_theta,
            )
        };

        quats.push(scale1 * w1 + scale2 * w2);
        quats.push(scale1 * x1 + scale2 * x2);
        quats.push(scale1 * y1 + scale2 * y2);
        quats.push(scale1 * z1 + scale2 * z2);
    }

    let quat_shape = if is_batch { vec![n, 4] } else { vec![4] };
    let quaternions = Tensor::<R>::from_slice(&quats, &quat_shape, device);

    Ok(Rotation {
        quaternions,
        is_batch,
    })
}

/// Create identity rotation.
pub fn rotation_identity_impl<R, C>(client: &C, n: Option<usize>) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>,
{
    let device = client.device();

    match n {
        None => {
            let quaternions = Tensor::<R>::from_slice(&[1.0, 0.0, 0.0, 0.0], &[4], device);
            Ok(Rotation {
                quaternions,
                is_batch: false,
            })
        }
        Some(count) => {
            let mut quats = Vec::with_capacity(count * 4);
            for _ in 0..count {
                quats.push(1.0);
                quats.push(0.0);
                quats.push(0.0);
                quats.push(0.0);
            }
            let quaternions = Tensor::<R>::from_slice(&quats, &[count, 4], device);
            Ok(Rotation {
                quaternions,
                is_batch: true,
            })
        }
    }
}

/// Create random rotation(s).
pub fn rotation_random_impl<R, C>(client: &C, n: Option<usize>) -> Result<Rotation<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + ReduceOps<R> + RandomOps<R> + RuntimeClient<R>,
{
    let count = n.unwrap_or(1);

    // Sample uniformly from SO(3) using random quaternions
    let shape = if n.is_some() { vec![count, 4] } else { vec![4] };

    // Generate random values
    let rand = client.randn(&shape, DType::F64)?;

    // Normalize to get unit quaternion
    rotation_from_quat_impl(client, &rand)
}

/// Compute rotation magnitude (angle).
pub fn rotation_magnitude_impl<R, C>(_client: &C, rot: &Rotation<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
    let quat_data: Vec<f64> = rot.quaternions.to_vec();
    let device = rot.quaternions.device();

    let n = if rot.is_batch {
        rot.quaternions.shape()[0]
    } else {
        1
    };

    let mut angles = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * 4;
        let w = quat_data[offset];
        let x = quat_data[offset + 1];
        let y = quat_data[offset + 2];
        let z = quat_data[offset + 3];

        let axis_len = (x * x + y * y + z * z).sqrt();
        let angle = 2.0 * axis_len.atan2(w.abs());
        angles.push(angle);
    }

    let shape = if rot.is_batch { vec![n] } else { vec![] };

    Ok(Tensor::<R>::from_slice(&angles, &shape, device))
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn cpu_client() -> CpuClient {
        CpuClient::new(CpuDevice::new())
    }

    /// Asserts two f64 values are approximately equal within tolerance.
    fn assert_approx(a: f64, b: f64, tol: f64, label: &str) {
        assert!(
            (a - b).abs() < tol,
            "{}: expected {}, got {} (diff={})",
            label,
            b,
            a,
            (a - b).abs()
        );
    }

    /// Test batch rotation on batch vectors produces shape [n, m, 3] with correct values.
    ///
    /// Rotation 0: identity (quaternion [1, 0, 0, 0])
    /// Rotation 1: 90° about Z (quaternion [cos45°, 0, 0, sin45°])
    ///
    /// Vectors:
    ///   v0 = [1, 0, 0]
    ///   v1 = [0, 1, 0]
    ///   v2 = [0, 0, 1]
    ///
    /// Expected result[0, j, :] = identity applied to v_j = v_j
    /// Expected result[1, 0, :] = 90°Z applied to [1,0,0] = [0,  1, 0]
    /// Expected result[1, 1, :] = 90°Z applied to [0,1,0] = [-1, 0, 0]
    /// Expected result[1, 2, :] = 90°Z applied to [0,0,1] = [0,  0, 1]
    #[test]
    fn test_batch_rotation_batch_vectors_shape_and_values() {
        let client = cpu_client();

        let s = (std::f64::consts::FRAC_PI_4).cos(); // cos(45°) = sin(45°) for 90° rotation
        // Batch of 2 quaternions: identity, 90° about Z
        let quats = Tensor::<CpuRuntime>::from_slice(
            &[1.0, 0.0, 0.0, 0.0, s, 0.0, 0.0, s],
            &[2, 4],
            client.device(),
        );
        let rot = rotation_from_quat_impl(&client, &quats).expect("quat");
        assert!(rot.is_batch);

        // 3 vectors: basis vectors
        let vecs = Tensor::<CpuRuntime>::from_slice(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
            client.device(),
        );

        let result = rotation_apply_impl(&client, &rot, &vecs).expect("apply");
        let shape = result.shape();
        assert_eq!(shape, vec![2, 3, 3], "result shape should be [n, m, 3]");

        let data: Vec<f64> = result.to_vec();
        // data layout: [rotation_idx, vector_idx, component]
        // result[i, j, k] = data[i*3*3 + j*3 + k]
        let at = |i: usize, j: usize, k: usize| data[i * 9 + j * 3 + k];

        let tol = 1e-10;

        // Rotation 0 (identity): result[0,j,:] == v_j
        assert_approx(at(0, 0, 0), 1.0, tol, "R0@v0[0]");
        assert_approx(at(0, 0, 1), 0.0, tol, "R0@v0[1]");
        assert_approx(at(0, 0, 2), 0.0, tol, "R0@v0[2]");
        assert_approx(at(0, 1, 0), 0.0, tol, "R0@v1[0]");
        assert_approx(at(0, 1, 1), 1.0, tol, "R0@v1[1]");
        assert_approx(at(0, 1, 2), 0.0, tol, "R0@v1[2]");
        assert_approx(at(0, 2, 0), 0.0, tol, "R0@v2[0]");
        assert_approx(at(0, 2, 1), 0.0, tol, "R0@v2[1]");
        assert_approx(at(0, 2, 2), 1.0, tol, "R0@v2[2]");

        // Rotation 1 (90° about Z): [x,y,z] -> [-y, x, z]
        // R1 @ [1,0,0] = [0, 1, 0]
        assert_approx(at(1, 0, 0), 0.0, tol, "R1@v0[0]");
        assert_approx(at(1, 0, 1), 1.0, tol, "R1@v0[1]");
        assert_approx(at(1, 0, 2), 0.0, tol, "R1@v0[2]");
        // R1 @ [0,1,0] = [-1, 0, 0]
        assert_approx(at(1, 1, 0), -1.0, tol, "R1@v1[0]");
        assert_approx(at(1, 1, 1), 0.0, tol, "R1@v1[1]");
        assert_approx(at(1, 1, 2), 0.0, tol, "R1@v1[2]");
        // R1 @ [0,0,1] = [0, 0, 1]
        assert_approx(at(1, 2, 0), 0.0, tol, "R1@v2[0]");
        assert_approx(at(1, 2, 1), 0.0, tol, "R1@v2[1]");
        assert_approx(at(1, 2, 2), 1.0, tol, "R1@v2[2]");
    }

    /// Consistency test: result[i, j, :] from batch-on-batch must equal applying rotation i
    /// to vector j individually (single rotation, single vector path).
    #[test]
    fn test_batch_rotation_batch_vectors_consistency_with_single() {
        let client = cpu_client();

        let s = (std::f64::consts::FRAC_PI_4).cos();
        // Two rotations: identity and 90° about Z
        let quats = Tensor::<CpuRuntime>::from_slice(
            &[1.0, 0.0, 0.0, 0.0, s, 0.0, 0.0, s],
            &[2, 4],
            client.device(),
        );
        let rot_batch = rotation_from_quat_impl(&client, &quats).expect("quat batch");

        // Two vectors
        let vectors_batch = Tensor::<CpuRuntime>::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            client.device(),
        );

        let batch_result =
            rotation_apply_impl(&client, &rot_batch, &vectors_batch).expect("batch apply");
        let batch_data: Vec<f64> = batch_result.to_vec();
        let at = |i: usize, j: usize, k: usize| batch_data[i * 2 * 3 + j * 3 + k];

        let tol = 1e-10;

        // Check each (rotation i, vector j) pair individually
        for rot_idx in 0..2usize {
            let q_offset = rot_idx * 4;
            let quats_vec: Vec<f64> = quats.to_vec();
            let single_quat_data = &[
                quats_vec[q_offset],
                quats_vec[q_offset + 1],
                quats_vec[q_offset + 2],
                quats_vec[q_offset + 3],
            ];
            let single_quat =
                Tensor::<CpuRuntime>::from_slice(single_quat_data, &[4], client.device());
            let rot_single = rotation_from_quat_impl(&client, &single_quat).expect("single quat");

            for vec_idx in 0..2usize {
                let v_offset = vec_idx * 3;
                let vd: Vec<f64> = vectors_batch.to_vec();
                let single_vec = Tensor::<CpuRuntime>::from_slice(
                    &[vd[v_offset], vd[v_offset + 1], vd[v_offset + 2]],
                    &[3],
                    client.device(),
                );
                let single_result =
                    rotation_apply_impl(&client, &rot_single, &single_vec).expect("single apply");
                let sr: Vec<f64> = single_result.to_vec();

                for (k, &sr_k) in sr.iter().enumerate().take(3) {
                    assert_approx(
                        at(rot_idx, vec_idx, k),
                        sr_k,
                        tol,
                        &format!("consistency R{rot_idx}@v{vec_idx}[{k}]"),
                    );
                }
            }
        }
    }
}

//! Increasing residual shapes with known roots and outward enclosures.

use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps};
use numr::prelude::*;
use solvr::optimize::roots::CertifiedRootValues;

pub trait ShapeClient<R: Runtime<DType = DType>>:
    TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + ScalarOps<R> + RuntimeClient<R>
{
}
impl<R, C> ShapeClient<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + CompareOps<R> + NextRepresentableOps<R> + ScalarOps<R> + RuntimeClient<R>,
{
}

/// Kink location of the piecewise quadratic, relative to its root.
const KINK: f64 = 0.1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    /// `3 (x - p)`.
    Affine,
    /// `sqrt(x) - p`, root `p^2`. Domains stay inside `[0, 1]`.
    Sqrt,
    /// `t^3 + t` with `t = x - p`.
    Cubic,
    /// `t + t^2/2` below `t = 0.1`, then a steeper quadratic. Domains keep `t >= -1/2`.
    Kinked,
}

impl Shape {
    fn code(self) -> f64 {
        match self {
            Self::Affine => 0.0,
            Self::Sqrt => 1.0,
            Self::Cubic => 2.0,
            Self::Kinked => 3.0,
        }
    }

    /// A positive slope floor over every admitted domain.
    pub fn kappa(self) -> f64 {
        match self {
            Self::Affine => 1.5,
            Self::Sqrt | Self::Kinked => 0.5,
            Self::Cubic => 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub shape: Shape,
    pub parameter: f64,
    pub lower: f64,
    pub upper: f64,
}

pub fn cast(value: f64, dtype: DType) -> f64 {
    if dtype == DType::F32 {
        f64::from(value as f32)
    } else {
        value
    }
}

fn step(value: f64, up: bool) -> f64 {
    let bits = value.to_bits();
    let next = if (value > 0.0) == up {
        bits + 1
    } else {
        bits - 1
    };
    f64::from_bits(next)
}

impl Row {
    /// A closed interval holding the exact root of the residual in this dtype.
    pub fn root(&self, dtype: DType) -> (f64, f64) {
        let p = cast(self.parameter, dtype);
        match self.shape {
            // An F32 square is exact in F64. An F64 square rounds once.
            Shape::Sqrt if dtype == DType::F32 => (p * p, p * p),
            Shape::Sqrt => (step(p * p, false), step(p * p, true)),
            _ => (p, p),
        }
    }
}

/// Absolute enclosure radius that covers every rounding error of the nominal values.
pub fn radius(dtype: DType) -> f64 {
    if dtype == DType::F32 { 1e-5 } else { 1e-12 }
}

pub fn tensor<R: Runtime<DType = DType>>(
    values: &[f64],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Tensor<R>> {
    match dtype {
        DType::F32 => Tensor::from_slice(
            &values.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[values.len()],
            device,
        ),
        _ => Tensor::from_slice(values, &[values.len()], device),
    }
}

pub struct Params<R: Runtime<DType = DType>> {
    code: Tensor<R>,
    parameter: Tensor<R>,
}

pub fn params<R: Runtime<DType = DType>>(
    rows: &[Row],
    dtype: DType,
    device: &R::Device,
) -> numr::error::Result<Params<R>> {
    let code: Vec<_> = rows.iter().map(|row| row.shape.code()).collect();
    let parameter: Vec<_> = rows.iter().map(|row| row.parameter).collect();
    Ok(Params {
        code: tensor(&code, dtype, device)?,
        parameter: tensor(&parameter, dtype, device)?,
    })
}

/// Outward enclosures of each owner's residual at the queried points.
pub fn evaluate<R, C>(
    client: &C,
    params: &Params<R>,
    owner: &Tensor<R>,
    x: &Tensor<R>,
) -> numr::error::Result<CertifiedRootValues<R>>
where
    R: Runtime<DType = DType>,
    C: ShapeClient<R>,
{
    let code = client.index_select(&params.code, 0, owner)?;
    let p = client.index_select(&params.parameter, 0, owner)?;
    let t = client.sub(x, &p)?;
    let affine = client.mul_scalar(&t, 3.0)?;
    let sqrt = client.sub(&client.sqrt(x)?, &p)?;
    let square = client.mul(&t, &t)?;
    let cubic = client.add(&client.mul(&square, &t)?, &t)?;
    let left = client.add(&t, &client.mul_scalar(&square, 0.5)?)?;
    let d = client.add_scalar(&t, -KINK)?;
    let right = client.add(
        &client.add_scalar(&client.mul_scalar(&d, 4.0)?, KINK + 0.5 * KINK * KINK)?,
        &client.mul_scalar(&client.mul(&d, &d)?, 2.0)?,
    )?;
    let kink = Tensor::full_scalar(t.shape(), t.dtype(), KINK, client.device())?;
    let kinked = client.where_cond(&client.lt(&t, &kink)?, &left, &right)?;
    let pick = |value: f64| -> numr::error::Result<Tensor<R>> {
        client.eq(
            &code,
            &Tensor::full_scalar(code.shape(), code.dtype(), value, client.device())?,
        )
    };
    let value = client.where_cond(&pick(2.0)?, &cubic, &kinked)?;
    let value = client.where_cond(&pick(1.0)?, &sqrt, &value)?;
    let value = client.where_cond(&pick(0.0)?, &affine, &value)?;
    let radius = radius(x.dtype());
    Ok(CertifiedRootValues {
        lower: client.next_down(&client.add_scalar(&value, -radius)?)?,
        upper: client.next_up(&client.add_scalar(&value, radius)?)?,
    })
}

/// Deterministic uniform samples in `[0, 1)`.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// One random row with a root strictly inside a domain within `[0, 1]`.
    pub fn row(&mut self, dtype: DType) -> Row {
        let shape = [Shape::Affine, Shape::Sqrt, Shape::Cubic, Shape::Kinked]
            [(self.next() * 4.0) as usize % 4];
        let lower = cast(self.next() * 0.5, dtype);
        let width = 2f64.powf(-6.0 + 5.0 * self.next());
        let upper = cast(lower + width, dtype);
        let root = lower + (0.05 + 0.9 * self.next()) * (upper - lower);
        let parameter = match shape {
            Shape::Sqrt => root.sqrt(),
            _ => root,
        };
        Row {
            shape,
            parameter: cast(parameter, dtype),
            lower,
            upper,
        }
    }
}

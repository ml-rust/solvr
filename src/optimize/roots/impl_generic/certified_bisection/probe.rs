//! Slack-gated secant probes inside certified root brackets.
//!
//! Each root row packs its probe state in one row of eight columns. Samples only steer
//! where the next probe lands. Certified signs, exact zeros, and the slope correction
//! remain the only operations that move a bracket endpoint.
//!
//! A row interpolates only when its bracket width is at most half of bisection's width
//! schedule. Any interior probe then keeps the width within the next schedule entry, so
//! the worst-case round count to a width criterion equals bisection's.
//!
//! A row without that slack probes its midpoint. When its samples give a usable secant,
//! the same callback also evaluates the secant point. Both corrected brackets enclose the
//! root, so the row keeps their intersection, which is never wider than the midpoint's.
//!
//! A row that has seen an uncertain sign is in its noise band. It always probes its
//! midpoint, and pairs it with the float of fewest significant bits inside the bracket.
//! An exact root can only be certified by evaluating that exact float. Midpoint probes
//! reach such floats from dyadic brackets, but secant steps move brackets off that grid.

use super::native::{self, RootClient};
use super::simplest::simplest;
use crate::DType;
use crate::optimize::OptimizeResult;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Columns of one packed probe row.
pub(crate) const COLUMNS: usize = 8;
/// Bisection's width bound after the executed rounds, rounded up.
const SCHEDULE: usize = 0;
/// ITP truncation factor for exponent two, fixed per row from the admitted width.
const TRUNCATION: usize = 1;
const X_PREVIOUS: usize = 2;
const F_PREVIOUS: usize = 3;
const X_LAST: usize = 4;
const F_LAST: usize = 5;
/// Count of usable certified samples: zero, one, or two.
const SAMPLES: usize = 6;
/// One after any probe of the row had an uncertain sign.
const NOISE: usize = 7;

/// ITP truncation numerator for exponent two.
const TRUNCATION_SCALE: f64 = 0.2;

/// Probe rows and the probe kind for each pending root.
pub(crate) struct Probe<R: Runtime<DType = DType>> {
    pub rows: Tensor<R>,
    /// One for a midpoint probe, zero for an interpolated probe.
    pub midpoint: Tensor<R>,
    /// One where a midpoint row also evaluates a second point in the same callback.
    pub paired: Tensor<R>,
    /// Second point of each paired row: its simplest float or its secant point.
    /// Other rows hold their midpoint.
    pub second: Tensor<R>,
    /// One where a row outside its noise band has a simplest float to try.
    /// A midpoint stall then keeps the row active for one paired round.
    pub retry: Tensor<R>,
}

pub(crate) struct Selection<R: Runtime<DType = DType>> {
    pub point: Tensor<R>,
    pub probe: Option<Probe<R>>,
}

impl<R: Runtime<DType = DType>> Probe<R> {
    /// Keep the probe rows of the evaluated roots.
    pub(crate) fn compact<C>(self, client: &C, kept: &Tensor<R>) -> OptimizeResult<Self>
    where
        C: RootClient<R>,
    {
        // Sorted indices that keep every row are the identity.
        if kept.numel() == self.midpoint.numel() {
            return Ok(self);
        }
        Ok(Self {
            rows: client.index_select(&self.rows, 0, kept)?,
            midpoint: client.index_select(&self.midpoint, 0, kept)?,
            paired: client.index_select(&self.paired, 0, kept)?,
            second: client.index_select(&self.second, 0, kept)?,
            retry: client.index_select(&self.retry, 0, kept)?,
        })
    }
}

fn columns<R: Runtime<DType = DType>>(rows: &Tensor<R>) -> OptimizeResult<Vec<Tensor<R>>> {
    let count = rows.shape()[0];
    let packed = rows.transpose(0, 1)?.contiguous()?;
    (0..COLUMNS)
        .map(|column| {
            Ok(packed
                .narrow(0, column, 1)?
                .contiguous()?
                .reshape(&[count])?)
        })
        .collect()
}

fn pack<R, C>(client: &C, columns: [&Tensor<R>; COLUMNS]) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    Ok(client.stack(&columns, 1)?)
}

fn finite<R, C>(client: &C, value: &Tensor<R>, zero: &Tensor<R>) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    Ok(client.eq(&native::nonfinite(client, value)?, zero)?)
}

/// Seed probe rows from the admitted brackets and their endpoint enclosures.
///
/// Both samples are stored only when both endpoint signs are certified.
pub(crate) fn seed<R, C>(
    client: &C,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    left: [&Tensor<R>; 2],
    right: [&Tensor<R>; 2],
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let zero = Tensor::zeros(lower.shape(), lower.dtype(), client.device())?;
    let one = Tensor::ones(lower.shape(), lower.dtype(), client.device())?;
    let schedule = client.next_up(&client.sub(upper, lower)?)?;
    // Degenerate and overflowing widths take a zero factor, which disables truncation.
    let usable = client.mul(
        &client.lt(lower, upper)?,
        &finite(client, &schedule, &zero)?,
    )?;
    let scale = Tensor::full_scalar(
        lower.shape(),
        lower.dtype(),
        TRUNCATION_SCALE,
        client.device(),
    )?;
    let factor = client.div(&scale, &client.where_cond(&usable, &schedule, &one)?)?;
    let usable = client.mul(&usable, &finite(client, &factor, &zero)?)?;
    let truncation = client.where_cond(&usable, &factor, &zero)?;
    let certified = client.mul(&client.lt(left[1], &zero)?, &client.gt(right[0], &zero)?)?;
    let x_previous = client.where_cond(&certified, lower, &zero)?;
    let f_previous = client.where_cond(&certified, &client.midpoint(left[0], left[1])?, &zero)?;
    let x_last = client.where_cond(&certified, upper, &zero)?;
    let f_last = client.where_cond(&certified, &client.midpoint(right[0], right[1])?, &zero)?;
    let samples = client.mul_scalar(&certified, 2.0)?;
    pack(
        client,
        [
            &schedule,
            &truncation,
            &x_previous,
            &f_previous,
            &x_last,
            &f_last,
            &samples,
            &zero,
        ],
    )
}

/// Clear the noise flag of the given rows.
///
/// A residual precision change makes the old uncertain signs stale.
pub(crate) fn forget<R, C>(
    client: &C,
    all: &Tensor<R>,
    roots: &Tensor<R>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let rows = client.index_select(all, 0, roots)?;
    let column = columns(&rows)?;
    let zero = Tensor::zeros(roots.shape(), rows.dtype(), client.device())?;
    let mut fields: Vec<&Tensor<R>> = column.iter().collect();
    fields[NOISE] = &zero;
    let rows = client.stack(&fields, 1)?;
    Ok(client.index_put(all, 0, roots, &rows)?)
}

/// Select one probe per pending root.
///
/// Rows without probe state take the bisection midpoint and no further operation.
pub(crate) fn select<R, C>(
    client: &C,
    rows: Option<Tensor<R>>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> OptimizeResult<Selection<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let midpoint = client.midpoint(lower, upper)?;
    let Some(rows) = rows else {
        return Ok(Selection {
            point: midpoint,
            probe: None,
        });
    };
    let column = columns(&rows)?;
    let zero = Tensor::zeros(lower.shape(), lower.dtype(), client.device())?;
    let one = Tensor::ones(lower.shape(), lower.dtype(), client.device())?;
    let two = Tensor::full_scalar(lower.shape(), lower.dtype(), 2.0, client.device())?;
    let schedule = &column[SCHEDULE];
    let noise = &column[NOISE];
    let quiet = client.sub(&one, noise)?;
    let width = client.next_up(&client.sub(upper, lower)?)?;
    let (simple, candidate) = simplest(client, [lower, upper], &width, &midpoint)?;
    // The gate requires a full round of slack against a finite schedule, outside the noise band.
    let gate = client.mul(
        &client.eq(&column[SAMPLES], &two)?,
        &client.le(&client.mul_scalar(&width, 2.0)?, schedule)?,
    )?;
    let gate = client.mul(
        &client.mul(&gate, &finite(client, schedule, &zero)?)?,
        &quiet,
    )?;
    let dx = client.sub(&column[X_LAST], &column[X_PREVIOUS])?;
    let df = client.sub(&column[F_LAST], &column[F_PREVIOUS])?;
    let moved = client.ne(&dx, &zero)?;
    // Invalid lanes take harmless operands before each division.
    let slope = client.div(&df, &client.where_cond(&moved, &dx, &one)?)?;
    // A steerable row has two samples and a finite positive secant slope.
    let steer = client.mul(
        &client.mul(&client.eq(&column[SAMPLES], &two)?, &moved)?,
        &client.mul(&client.gt(&slope, &zero)?, &finite(client, &slope, &zero)?)?,
    )?;
    let step = client.div(&column[F_LAST], &client.where_cond(&steer, &slope, &one)?)?;
    let raw = client.sub(&column[X_LAST], &step)?;
    let steer = client.mul(&steer, &finite(client, &raw, &zero)?)?;
    let raw = client.where_cond(&steer, &raw, &midpoint)?;
    let usable = client.mul(&gate, &steer)?;
    let secant = client.where_cond(&usable, &raw, &midpoint)?;
    let width = client.where_cond(&usable, &width, &zero)?;
    // The factor and width are finite here, and their product stays below one width.
    let delta = client.mul(&client.mul(&column[TRUNCATION], &width)?, &width)?;
    let toward = client.where_cond(
        &client.lt(&secant, &midpoint)?,
        &client.add(&secant, &delta)?,
        &client.sub(&secant, &delta)?,
    )?;
    let distance = client.abs(&client.sub(&midpoint, &secant)?)?;
    let truncated = client.where_cond(&client.ge(&delta, &distance)?, &midpoint, &toward)?;
    let inner_lower = client.next_up(lower)?;
    let inner_upper = client.next_down(upper)?;
    // An empty interior leaves adjacent endpoints to the midpoint floor rule.
    let interior = client.le(&inner_lower, &inner_upper)?;
    let usable = client.mul(&usable, &interior)?;
    let clamped = client.minimum(&client.maximum(&truncated, &inner_lower)?, &inner_upper)?;
    let point = client.where_cond(&usable, &clamped, &midpoint)?;
    // A row without slack probes its midpoint, so the schedule holds for any outcome.
    // Its untruncated secant point joins the same callback as a second sample.
    let closed = client.sub(&one, &gate)?;
    let secant = client.minimum(&client.maximum(&raw, &inner_lower)?, &inner_upper)?;
    let steered = client.mul(&client.mul(&closed, &steer)?, &interior)?;
    let steered = client.mul(&steered, &client.ne(&secant, &midpoint)?)?;
    // A row in its noise band pairs its midpoint with the simplest float when one exists.
    let dyadic = client.mul(noise, &candidate)?;
    let steered = client.mul(&steered, &client.sub(&one, &dyadic)?)?;
    let paired = client.maximum(&dyadic, &steered)?;
    let second = client.where_cond(
        &dyadic,
        &simple,
        &client.where_cond(&steered, &secant, &midpoint)?,
    )?;
    Ok(Selection {
        point,
        probe: Some(Probe {
            rows,
            midpoint: client.sub(&one, &usable)?,
            paired,
            second,
            retry: client.mul(&quiet, &candidate)?,
        }),
    })
}

/// The second point of each paired row and its residual enclosure.
pub(crate) struct Second<'a, R: Runtime<DType = DType>> {
    /// One for a paired row. Other lanes hold harmless values.
    pub paired: &'a Tensor<R>,
    pub point: &'a Tensor<R>,
    pub residual: [&'a Tensor<R>; 2],
}

/// Record one evaluated probe and advance the width schedule.
///
/// A certified sign stores the probe and its nominal residual. Any other outcome clears
/// both samples, so the next probes are midpoints until two certified samples exist.
/// A paired row then records its second point the same way, after its midpoint.
/// An uncertain sign at either point also sets the noise flag, which never clears here.
pub(crate) fn record<R, C>(
    client: &C,
    rows: &Tensor<R>,
    point: &Tensor<R>,
    residual: [&Tensor<R>; 2],
    bracket: [&Tensor<R>; 2],
    second: Option<Second<'_, R>>,
) -> OptimizeResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RootClient<R>,
{
    let column = columns(rows)?;
    let zero = Tensor::zeros(point.shape(), point.dtype(), client.device())?;
    let one = Tensor::ones(point.shape(), point.dtype(), client.device())?;
    let two = Tensor::full_scalar(point.shape(), point.dtype(), 2.0, client.device())?;
    let certified = client.maximum(
        &client.gt(residual[0], &zero)?,
        &client.lt(residual[1], &zero)?,
    )?;
    let nominal = client.midpoint(residual[0], residual[1])?;
    let mut x_previous = client.where_cond(&certified, &column[X_LAST], &column[X_PREVIOUS])?;
    let mut f_previous = client.where_cond(&certified, &column[F_LAST], &column[F_PREVIOUS])?;
    let mut x_last = client.where_cond(&certified, point, &column[X_LAST])?;
    let mut f_last = client.where_cond(&certified, &nominal, &column[F_LAST])?;
    let counted = client.minimum(&client.add(&column[SAMPLES], &one)?, &two)?;
    let mut samples = client.where_cond(&certified, &counted, &zero)?;
    // An enclosure that is neither signed nor exactly zero marks the noise band.
    let unresolved = |residual: [&Tensor<R>; 2], signed: &Tensor<R>| -> OptimizeResult<Tensor<R>> {
        let exact = client.mul(
            &client.eq(residual[0], &zero)?,
            &client.eq(residual[1], &zero)?,
        )?;
        Ok(client.mul(&client.sub(&one, signed)?, &client.sub(&one, &exact)?)?)
    };
    let mut noise = client.maximum(&column[NOISE], &unresolved(residual, &certified)?)?;
    if let Some(second) = second {
        let signed = client.maximum(
            &client.gt(second.residual[0], &zero)?,
            &client.lt(second.residual[1], &zero)?,
        )?;
        let certified = client.mul(second.paired, &signed)?;
        let uncertain = client.mul(second.paired, &client.sub(&one, &signed)?)?;
        noise = client.maximum(
            &noise,
            &client.mul(second.paired, &unresolved(second.residual, &signed)?)?,
        )?;
        let nominal = client.midpoint(second.residual[0], second.residual[1])?;
        x_previous = client.where_cond(&certified, &x_last, &x_previous)?;
        f_previous = client.where_cond(&certified, &f_last, &f_previous)?;
        x_last = client.where_cond(&certified, second.point, &x_last)?;
        f_last = client.where_cond(&certified, &nominal, &f_last)?;
        let counted = client.minimum(&client.add(&samples, &one)?, &two)?;
        samples = client.where_cond(&certified, &counted, &samples)?;
        samples = client.where_cond(&uncertain, &zero, &samples)?;
    }
    let width = client.next_up(&client.sub(bracket[1], bracket[0])?)?;
    let schedule = &column[SCHEDULE];
    let halved = client.maximum(&client.mul_scalar(schedule, 0.5)?, &width)?;
    // An overflowing admitted width restarts the schedule at the first finite width.
    let schedule = client.where_cond(&finite(client, schedule, &zero)?, &halved, &width)?;
    pack(
        client,
        [
            &schedule,
            &column[TRUNCATION],
            &x_previous,
            &f_previous,
            &x_last,
            &f_last,
            &samples,
            &noise,
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimize::roots::traits::{
        CertifiedBisectionAlgorithms, CertifiedBisectionOptions,
        CertifiedMappedBisectionAlgorithms, CertifiedRootAdvanceAllowance,
        CertifiedRootContinuation, CertifiedRootDomainQuery, CertifiedRootMappedValues,
        CertifiedRootProbe, CertifiedRootQuery, CertifiedRootTolerance, CertifiedRootValues,
    };
    use numr::prelude::*;

    type T = Tensor<CpuRuntime>;
    type Result<V = ()> = std::result::Result<V, Box<dyn std::error::Error>>;

    fn client() -> CpuClient {
        CpuClient::new(CpuDevice::new())
    }

    fn tensor(client: &CpuClient, values: &[f64], shape: &[usize], dtype: DType) -> Result<T> {
        Ok(match dtype {
            DType::F32 => T::from_slice(
                &values.iter().map(|&x| x as f32).collect::<Vec<_>>(),
                shape,
                client.device(),
            )?,
            _ => T::from_slice(values, shape, client.device())?,
        })
    }

    fn read(value: &T) -> Result<Vec<f64>> {
        let value = value.contiguous()?;
        Ok(match value.dtype() {
            DType::F32 => value
                .try_to_vec::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect(),
            _ => value.try_to_vec::<f64>()?,
        })
    }

    /// Probe rows as host columns.
    fn host(rows: &T) -> Result<Vec<Vec<f64>>> {
        columns(rows)?.iter().map(read).collect()
    }

    fn bits(rows: &T) -> Result<Vec<u64>> {
        Ok(read(rows)?.into_iter().map(f64::to_bits).collect())
    }

    #[test]
    fn select_without_probe_state_takes_the_midpoint() -> Result {
        let client = client();
        let lower = tensor(&client, &[0.0, 1.0], &[2], DType::F64)?;
        let upper = tensor(&client, &[1.0, 3.0], &[2], DType::F64)?;
        let selection = select(&client, None, &lower, &upper)?;
        assert!(selection.probe.is_none());
        assert_eq!(read(&selection.point)?, [0.5, 2.0]);
        Ok(())
    }

    #[test]
    fn select_interpolates_only_with_two_samples_slack_and_positive_slope() -> Result {
        let client = client();
        let adjacent = 1.0f64.next_up();
        // Affine samples of x - 0.375 at 0.25 and 0.75 give the exact secant root.
        let lanes: [(f64, f64, [f64; COLUMNS]); 9] = [
            (0.0, 1.0, [4.0, 0.0, 0.25, -0.125, 0.75, 0.375, 2.0, 0.0]),
            (0.0, 1.0, [4.0, 0.0, 0.25, -0.125, 0.75, 0.375, 1.0, 0.0]),
            (0.0, 1.0, [1.5, 0.0, 0.25, -0.125, 0.75, 0.375, 2.0, 0.0]),
            (0.0, 1.0, [4.0, 0.0, 0.25, 0.375, 0.75, -0.125, 2.0, 0.0]),
            (0.0, 1.0, [4.0, 0.0625, 0.25, -0.125, 0.75, 0.375, 2.0, 0.0]),
            (0.0, 1.0, [4.0, 0.0, 0.25, 0.5, 0.75, 0.625, 2.0, 0.0]),
            (
                1.0,
                adjacent,
                [4.0, 0.0, 0.25, -0.125, 0.75, 0.375, 2.0, 0.0],
            ),
            // A noise row with slack and a secant still probes its midpoint, paired with 0.5.
            (0.3, 0.9, [4.0, 0.0, 0.25, -0.125, 0.75, 0.375, 2.0, 1.0]),
            // A quiet row without a secant probes its midpoint and may retry 0.5 after a stall.
            (0.3, 0.9, [4.0, 0.0, 0.25, -0.125, 0.75, 0.375, 1.0, 0.0]),
        ];
        let count = lanes.len();
        let lower: Vec<_> = lanes.iter().map(|lane| lane.0).collect();
        let upper: Vec<_> = lanes.iter().map(|lane| lane.1).collect();
        let packed: Vec<_> = lanes.iter().flat_map(|lane| lane.2).collect();
        let lower = tensor(&client, &lower, &[count], DType::F64)?;
        let upper = tensor(&client, &upper, &[count], DType::F64)?;
        let rows = tensor(&client, &packed, &[count, COLUMNS], DType::F64)?;
        let selection = select(&client, Some(rows), &lower, &upper)?;
        let point = read(&selection.point)?;
        let probe = selection.probe.ok_or("probe")?;
        let midpoint = read(&probe.midpoint)?;
        // The steerable row without slack pairs its midpoint with the raw secant.
        // The noise row pairs its midpoint with the simplest float.
        assert_eq!(
            read(&probe.paired)?,
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(read(&probe.second)?[2], 0.375);
        assert_eq!(read(&probe.second)?[7], 0.5);
        assert_eq!(point[0], 0.375);
        assert_eq!(&point[1..4], [0.5, 0.5, 0.5]);
        // Truncation moves the secant root toward the midpoint by one sixteenth.
        assert!((0.4375..0.4375 + 1e-15).contains(&point[4]), "{}", point[4]);
        assert_eq!(point[5], f64::from_bits(1));
        assert!(point[6] == 1.0 || point[6] == adjacent);
        assert_eq!(point[7], 0.6);
        assert_eq!(midpoint, [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
        assert_eq!(
            read(&probe.retry)?,
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
        Ok(())
    }

    #[test]
    fn record_shifts_certified_samples_and_halves_the_schedule() -> Result {
        let client = client();
        let packed = [
            [1.0, 0.2, 0.1, -0.5, 0.9, 0.5, 2.0, 0.0],
            [1.0, 0.2, 0.1, -0.5, 0.9, 0.5, 2.0, 0.0],
            [f64::INFINITY, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let rows = tensor(&client, packed.as_flattened(), &[3, COLUMNS], DType::F64)?;
        let point = tensor(&client, &[0.5, 0.5, 0.5], &[3], DType::F64)?;
        let residual_lower = tensor(&client, &[0.25, -0.25, 0.25], &[3], DType::F64)?;
        let residual_upper = tensor(&client, &[0.75, 0.25, 0.75], &[3], DType::F64)?;
        let lower = tensor(&client, &[0.1, 0.1, 0.1], &[3], DType::F64)?;
        let upper = tensor(&client, &[0.2, 0.9, 0.2], &[3], DType::F64)?;
        let recorded = record(
            &client,
            &rows,
            &point,
            [&residual_lower, &residual_upper],
            [&lower, &upper],
            None,
        )?;
        let column = host(&recorded)?;
        let narrow = (0.2f64 - 0.1).next_up();
        assert_eq!(column[SCHEDULE], [0.5, (0.9f64 - 0.1).next_up(), narrow]);
        assert_eq!(column[TRUNCATION], [0.2, 0.2, 0.0]);
        assert_eq!(column[X_PREVIOUS], [0.9, 0.1, 0.0]);
        assert_eq!(column[F_PREVIOUS], [0.5, -0.5, 0.0]);
        assert_eq!(column[X_LAST], [0.5, 0.9, 0.5]);
        assert_eq!(column[F_LAST], [0.5, 0.5, 0.5]);
        assert_eq!(column[SAMPLES], [2.0, 0.0, 1.0]);
        // Only the uncertain midpoint marks the noise band.
        assert_eq!(column[NOISE], [0.0, 1.0, 0.0]);
        Ok(())
    }

    #[test]
    fn record_stores_a_paired_secant_point_after_its_midpoint() -> Result {
        let client = client();
        let row = [1.0, 0.2, 0.1, -0.5, 0.9, 0.5, 2.0, 0.0];
        let packed = [row, row, row];
        let rows = tensor(&client, packed.as_flattened(), &[3, COLUMNS], DType::F64)?;
        let point = tensor(&client, &[0.5, 0.5, 0.5], &[3], DType::F64)?;
        let residual_lower = tensor(&client, &[0.25, 0.25, 0.25], &[3], DType::F64)?;
        let residual_upper = tensor(&client, &[0.75, 0.75, 0.75], &[3], DType::F64)?;
        let lower = tensor(&client, &[0.1, 0.1, 0.1], &[3], DType::F64)?;
        let upper = tensor(&client, &[0.2, 0.2, 0.2], &[3], DType::F64)?;
        let paired = tensor(&client, &[1.0, 1.0, 0.0], &[3], DType::F64)?;
        let second = tensor(&client, &[0.3, 0.3, 0.3], &[3], DType::F64)?;
        let second_lower = tensor(&client, &[-0.25, -0.25, -0.25], &[3], DType::F64)?;
        let second_upper = tensor(&client, &[-0.125, 0.25, -0.125], &[3], DType::F64)?;
        let recorded = record(
            &client,
            &rows,
            &point,
            [&residual_lower, &residual_upper],
            [&lower, &upper],
            Some(Second {
                paired: &paired,
                point: &second,
                residual: [&second_lower, &second_upper],
            }),
        )?;
        let column = host(&recorded)?;
        // A certified pair keeps the midpoint as the older sample and the secant point as the newer.
        // An uncertain secant point clears both samples. An unpaired lane ignores its secant lane.
        assert_eq!(column[X_PREVIOUS], [0.5, 0.9, 0.9]);
        assert_eq!(column[F_PREVIOUS], [0.5, 0.5, 0.5]);
        assert_eq!(column[X_LAST], [0.3, 0.5, 0.5]);
        assert_eq!(column[F_LAST], [-0.1875, 0.5, 0.5]);
        assert_eq!(column[SAMPLES], [2.0, 0.0, 2.0]);
        assert_eq!(column[NOISE], [0.0, 1.0, 0.0]);
        assert_eq!(column[SCHEDULE], [0.5, 0.5, 0.5]);
        Ok(())
    }

    #[test]
    fn seed_stores_samples_only_for_two_certified_endpoints() -> Result {
        let client = client();
        let lower = tensor(&client, &[0.0, 0.0, 0.5], &[3], DType::F64)?;
        let upper = tensor(&client, &[1.0, 1.0, 0.5], &[3], DType::F64)?;
        let left_lower = tensor(&client, &[-0.5, -0.5, 0.0], &[3], DType::F64)?;
        let left_upper = tensor(&client, &[-0.25, 0.0, 0.0], &[3], DType::F64)?;
        let right_lower = tensor(&client, &[0.25, 0.25, 0.0], &[3], DType::F64)?;
        let right_upper = tensor(&client, &[0.75, 0.75, 0.0], &[3], DType::F64)?;
        let rows = seed(
            &client,
            &lower,
            &upper,
            [&left_lower, &left_upper],
            [&right_lower, &right_upper],
        )?;
        let column = host(&rows)?;
        let one_up = 1.0f64.next_up();
        assert_eq!(column[SCHEDULE], [one_up, one_up, f64::from_bits(1)]);
        assert_eq!(column[TRUNCATION], [0.2 / one_up, 0.2 / one_up, 0.0]);
        assert_eq!(column[X_PREVIOUS], [0.0, 0.0, 0.0]);
        assert_eq!(column[F_PREVIOUS], [-0.375, 0.0, 0.0]);
        assert_eq!(column[X_LAST], [1.0, 0.0, 0.0]);
        assert_eq!(column[F_LAST], [0.5, 0.0, 0.0]);
        assert_eq!(column[SAMPLES], [2.0, 0.0, 0.0]);
        Ok(())
    }

    /// Per-owner roots of `3 (x - r)` for even owners and `t^3 + t` for odd owners.
    struct Residual {
        roots: T,
        odd: T,
        band: f64,
    }

    impl Residual {
        fn new(client: &CpuClient, roots: &[f64], band: f64, dtype: DType) -> Result<Self> {
            let odd: Vec<_> = (0..roots.len()).map(|i| (i % 2) as f64).collect();
            Ok(Self {
                roots: tensor(client, roots, &[roots.len()], dtype)?,
                odd: tensor(client, &odd, &[roots.len()], dtype)?,
                band,
            })
        }

        /// Outward enclosures. Points within the band take an uncertain `[-1, 1]`.
        fn evaluate(
            &self,
            client: &CpuClient,
            query: &CertifiedRootQuery<CpuRuntime>,
        ) -> numr::error::Result<CertifiedRootValues<CpuRuntime>> {
            let x = &query.point;
            let t = client.sub(x, &client.index_select(&self.roots, 0, &query.owner)?)?;
            let odd = client.index_select(&self.odd, 0, &query.owner)?;
            let cubic = client.add(&client.mul(&client.mul(&t, &t)?, &t)?, &t)?;
            let value = client.where_cond(&odd, &cubic, &client.mul_scalar(&t, 3.0)?)?;
            let radius = if x.dtype() == DType::F32 { 1e-5 } else { 1e-12 };
            let full = |v: f64| T::full_scalar(x.shape(), x.dtype(), v, client.device());
            let near = client.lt(&client.abs(&t)?, &full(self.band)?)?;
            Ok(CertifiedRootValues {
                lower: client.where_cond(
                    &near,
                    &full(-1.0)?,
                    &client.next_down(&client.add_scalar(&value, -radius)?)?,
                )?,
                upper: client.where_cond(
                    &near,
                    &full(1.0)?,
                    &client.next_up(&client.add_scalar(&value, radius)?)?,
                )?,
            })
        }
    }

    fn identity(
        query: &CertifiedRootDomainQuery<CpuRuntime>,
    ) -> numr::error::Result<CertifiedRootMappedValues<CpuRuntime>> {
        Ok(CertifiedRootMappedValues {
            lower: query.lower.clone(),
            upper: query.upper.clone(),
        })
    }

    fn options(probe: CertifiedRootProbe) -> CertifiedBisectionOptions {
        // A width above every domain stops the first call after endpoint admission.
        CertifiedBisectionOptions {
            atol: 8.0,
            rtol: 0.0,
            max_iterations: 400,
            max_points: 1 << 20,
            probe,
        }
    }

    fn begin(
        client: &CpuClient,
        residual: &Residual,
        domain: [&[f64]; 2],
        dtype: DType,
        probe: CertifiedRootProbe,
    ) -> Result<CertifiedRootContinuation<CpuRuntime>> {
        let count = domain[0].len();
        let kappa: Vec<_> = (0..count).map(|i| [0.25, 1.0][i % 2]).collect();
        let progress = client.bisect_certified_mapped(
            |query| residual.evaluate(client, query),
            identity,
            &tensor(client, domain[0], &[count], dtype)?,
            &tensor(client, domain[1], &[count], dtype)?,
            &tensor(client, &kappa, &[count], dtype)?,
            &options(probe),
        )?;
        assert_eq!(progress.result().iterations(), 0);
        Ok(progress.into_parts().1)
    }

    fn step(
        client: &CpuClient,
        residual: &Residual,
        continuation: CertifiedRootContinuation<CpuRuntime>,
        atol: f64,
        rounds: usize,
    ) -> Result<CertifiedRootContinuation<CpuRuntime>> {
        Ok(client
            .resume_certified_mapped(
                |query| residual.evaluate(client, query),
                identity,
                continuation,
                &CertifiedRootTolerance { atol, rtol: 0.0 },
                &CertifiedRootAdvanceAllowance {
                    max_iterations: rounds,
                    max_points: 1 << 20,
                },
            )?
            .into_parts()
            .1)
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    fn random_domain(random: &mut Lcg, count: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut domain = (Vec::new(), Vec::new(), Vec::new());
        for _ in 0..count {
            let lower = random.next() - 0.5;
            let width = 2f64.powf(-4.0 + 5.0 * random.next());
            domain.0.push(lower);
            domain.1.push(lower + width);
            domain.2.push(lower + (0.05 + 0.9 * random.next()) * width);
        }
        domain
    }

    #[test]
    fn slack_schedule_bounds_the_width_after_every_round() -> Result {
        let client = client();
        for dtype in [DType::F32, DType::F64] {
            let mut random = Lcg(0x5c4e_d01e + dtype.size_in_bytes() as u64);
            // Each tolerance stays above the residual floor width of its dtype.
            let atol = if dtype == DType::F32 { 1e-3 } else { 1e-9 };
            for _ in 0..4 {
                let (lower, upper, roots) = random_domain(&mut random, 32);
                let residual = Residual::new(&client, &roots, 0.0, dtype)?;
                let mut continuation = begin(
                    &client,
                    &residual,
                    [&lower, &upper],
                    dtype,
                    CertifiedRootProbe::SlackSecant,
                )?;
                let mut interpolated = 0;
                for _ in 0..200 {
                    let state = &continuation.state;
                    let rows = state.probe_state.clone();
                    let before = host(rows.as_ref().ok_or("probe state")?)?;
                    let old_lower = read(&state.lower)?;
                    let selection = select(&client, rows, &state.lower, &state.upper)?;
                    let midpoint = read(&selection.probe.ok_or("probe")?.midpoint)?;
                    continuation = step(&client, &residual, continuation, atol, 1)?;
                    let state = &continuation.state;
                    let column = host(state.probe_state.as_ref().ok_or("probe state")?)?;
                    let lower = read(&state.lower)?;
                    for (row, upper) in read(&state.upper)?.into_iter().enumerate() {
                        let schedule = column[SCHEDULE][row];
                        assert!(upper - lower[row] <= schedule, "{dtype:?} row {row}");
                        assert!(schedule <= before[SCHEDULE][row], "{dtype:?} row {row}");
                        // A moved endpoint shows the row was evaluated this round.
                        if lower[row] != old_lower[row] && midpoint[row] == 0.0 {
                            interpolated += 1;
                        }
                    }
                    let status = state.status.try_to_vec::<i64>()?;
                    if status.iter().all(|&code| code == 0) {
                        break;
                    }
                }
                let status = continuation.state.status.try_to_vec::<i64>()?;
                assert!(status.iter().all(|&code| code == 0), "{dtype:?} {status:?}");
                assert!(interpolated > 0, "{dtype:?} never interpolated");
            }
        }
        Ok(())
    }

    #[test]
    fn interpolated_stall_keeps_rows_active_and_midpoint_stall_floors() -> Result {
        let client = client();
        for dtype in [DType::F32, DType::F64] {
            let band = if dtype == DType::F32 { 1e-3 } else { 1e-7 };
            let mut random = Lcg(0xf100 + dtype.size_in_bytes() as u64);
            let (lower, upper, roots) = random_domain(&mut random, 32);
            let residual = Residual::new(&client, &roots, band, dtype)?;
            let mut continuation = begin(
                &client,
                &residual,
                [&lower, &upper],
                dtype,
                CertifiedRootProbe::SlackSecant,
            )?;
            let (mut interpolated_stalls, mut midpoint_stalls) = (0, 0);
            for _ in 0..400 {
                let state = &continuation.state;
                // With a zero tolerance every unfloored row takes the next round.
                let active: Vec<_> = read(&continuation.floor)?
                    .into_iter()
                    .map(|floor| 1.0 - floor)
                    .collect();
                if active.iter().all(|&a| a == 0.0) {
                    break;
                }
                let before = (read(&state.lower)?, read(&state.upper)?);
                let selection = select(
                    &client,
                    state.probe_state.clone(),
                    &state.lower,
                    &state.upper,
                )?;
                let probe = selection.probe.ok_or("probe")?;
                let midpoint = read(&probe.midpoint)?;
                let retry = read(&probe.retry)?;
                continuation = step(&client, &residual, continuation, 0.0, 1)?;
                let state = &continuation.state;
                let after = (read(&state.lower)?, read(&state.upper)?);
                let owner = state.owner.try_to_vec::<i64>()?;
                let status = state.status.try_to_vec::<i64>()?;
                let floor = read(&continuation.floor)?;
                let still = read(&state.active)?;
                for row in 0..active.len() {
                    let unchanged = before.0[row] == after.0[row] && before.1[row] == after.1[row];
                    if active[row] != 1.0 || !unchanged || before.0[row] == before.1[row] {
                        continue;
                    }
                    let code = status[owner[row] as usize];
                    if midpoint[row] == 0.0 {
                        interpolated_stalls += 1;
                        assert_eq!((code, floor[row], still[row]), (4, 0.0, 1.0));
                    } else if retry[row] == 1.0 {
                        // A first midpoint stall tries the simplest float next round.
                        assert_eq!((code, floor[row], still[row]), (4, 0.0, 1.0));
                    } else {
                        midpoint_stalls += 1;
                        assert_eq!((code, floor[row], still[row]), (3, 1.0, 0.0));
                    }
                }
            }
            let status = continuation.state.status.try_to_vec::<i64>()?;
            assert!(status.iter().all(|&code| code == 3), "{dtype:?} {status:?}");
            assert!(interpolated_stalls > 0, "{dtype:?} no interpolated stall");
            assert!(midpoint_stalls > 0, "{dtype:?} no midpoint stall");
            for (index, (lo, hi)) in read(&continuation.state.lower)?
                .into_iter()
                .zip(read(&continuation.state.upper)?)
                .enumerate()
            {
                let root = if dtype == DType::F32 {
                    f64::from(roots[index] as f32)
                } else {
                    roots[index]
                };
                assert!(lo < root && root < hi, "{dtype:?} row {index}");
            }
        }
        Ok(())
    }

    #[test]
    fn continuations_keep_probe_rows_as_specified() -> Result {
        let client = client();
        for dtype in [DType::F32, DType::F64] {
            let mut random = Lcg(0xc047 + dtype.size_in_bytes() as u64);
            let (lower, upper, roots) = random_domain(&mut random, 8);
            let residual = Residual::new(&client, &roots, 0.0, dtype)?;
            let continuation = begin(
                &client,
                &residual,
                [&lower, &upper],
                dtype,
                CertifiedRootProbe::SlackSecant,
            )?;
            let fine = if dtype == DType::F32 { 1e-3 } else { 1e-9 };
            // One round leaves every row active, so the selected resume below moves each demanded row.
            let continuation = step(&client, &residual, continuation, fine, 1)?;
            let probe_rows = |c: &CertifiedRootContinuation<CpuRuntime>| -> Result<Vec<u64>> {
                bits(c.state.probe_state.as_ref().ok_or("probe state")?)
            };
            let original = probe_rows(&continuation)?;

            // A resume without rounds keeps every probe bit.
            let continuation = step(&client, &residual, continuation, fine, 0)?;
            assert_eq!(probe_rows(&continuation)?, original);

            // A refresh keeps the probe rows of demanded roots and clears sampled history.
            let all = T::from_slice(&(0..8i64).collect::<Vec<_>>(), &[8], client.device())?;
            let refreshed = client
                .refresh_certified_mapped_selected(
                    |query| residual.evaluate(&client, query),
                    identity,
                    continuation,
                    &all,
                    &CertifiedRootTolerance {
                        atol: fine,
                        rtol: 0.0,
                    },
                    &CertifiedRootAdvanceAllowance {
                        max_iterations: 0,
                        max_points: 0,
                    },
                )?
                .into_parts()
                .1;
            // The refresh clears only the noise flag, which records the old precision.
            let kept = probe_rows(&refreshed)?;
            let mut noisy = 0;
            for (index, (&kept, &old)) in kept.iter().zip(&original).enumerate() {
                if index % COLUMNS == NOISE {
                    assert_eq!(kept, 0, "{dtype:?} lane {index}");
                    noisy += usize::from(old != 0);
                } else {
                    assert_eq!(kept, old, "{dtype:?} lane {index}");
                }
            }
            // Secant points on the affine rows land inside the enclosure radius of the root.
            assert!(noisy > 0, "{dtype:?}: no row reached its noise band");
            let original = kept;
            assert!(read(&refreshed.state.sampled)?.iter().all(|&s| s == 0.0));

            // A selected resume to a finer width advances only demanded probe rows.
            let finer = fine * 1e-3;
            let pending: Vec<_> = read(&refreshed.state.upper)?
                .into_iter()
                .zip(read(&refreshed.state.lower)?)
                .map(|(upper, lower)| upper - lower > finer)
                .collect();
            let demanded = T::from_slice(&[1i64, 4, 6], &[3], client.device())?;
            let resumed = client
                .resume_certified_mapped_selected(
                    |query| residual.evaluate(&client, query),
                    identity,
                    refreshed,
                    &demanded,
                    &CertifiedRootTolerance {
                        atol: finer,
                        rtol: 0.0,
                    },
                    &CertifiedRootAdvanceAllowance {
                        max_iterations: 2,
                        max_points: 1 << 20,
                    },
                )?
                .into_parts()
                .1;
            let after = probe_rows(&resumed)?;
            let owner = resumed.state.owner.try_to_vec::<i64>()?;
            let mut advanced = 0;
            for (row, &owner) in owner.iter().enumerate() {
                let span = row * COLUMNS..(row + 1) * COLUMNS;
                if [1, 4, 6].contains(&owner) && pending[row] {
                    assert_ne!(after[span.clone()], original[span], "owner {owner}");
                    advanced += 1;
                } else {
                    assert_eq!(after[span.clone()], original[span], "owner {owner}");
                }
            }
            assert!(advanced > 0, "{dtype:?}: no demanded row was pending");
        }
        Ok(())
    }

    #[test]
    fn midpoint_sessions_keep_no_probe_state() -> Result {
        let client = client();
        for dtype in [DType::F32, DType::F64] {
            let residual = Residual::new(&client, &[0.3, 0.6], 0.0, dtype)?;
            let continuation = begin(
                &client,
                &residual,
                [&[0.0, 0.0], &[1.0, 1.0]],
                dtype,
                CertifiedRootProbe::Midpoint,
            )?;
            assert!(continuation.state.probe_state.is_none());
            let continuation = step(&client, &residual, continuation, 1e-3, 400)?;
            assert!(continuation.state.probe_state.is_none());
            let lower = tensor(&client, &[0.0], &[1], dtype)?;
            let upper = tensor(&client, &[1.0], &[1], dtype)?;
            let result = client.bisect_certified(
                |query| residual.evaluate(&client, query),
                &lower,
                &upper,
                &upper,
                &CertifiedBisectionOptions {
                    atol: 1e-3,
                    ..Default::default()
                },
            )?;
            assert_eq!(result.status().try_to_vec::<i64>()?, [0]);
        }
        Ok(())
    }
}

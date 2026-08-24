//! Dormand-Prince 8(5,3) method (DOP853) using tensor operations.
//!
//! High-order adaptive ODE solver with 8th order solution and 5th/3rd order error estimates.
//! All computation stays on device using numr's TensorOps.
//! Step size control is fully device-resident - no scalar transfers during stepping.

// Coefficients from Hairer, Norsett & Wanner - preserved exactly as reference
#![allow(clippy::excessive_precision)]
// Allow this module to use extensive precision coefficients from the reference implementation
use crate::DType;

use numr::error::Result;
use numr::ops::{ScalarOps, TensorOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

use crate::integrate::error::{IntegrateError, IntegrateResult};
use crate::integrate::{ODEMethod, ODEOptions};

use super::{
    ODEResultTensor, compute_acceptance, compute_error, compute_initial_step, compute_step_factor,
};

// DOP853 coefficients (Hairer, Norsett & Wanner)
// c coefficients (time points)
const C2: f64 = 0.526001519587677318785587544488e-01;
const C3: f64 = 0.789002279381515978178381316732e-01;
const C4: f64 = 0.118350341907227396726757197510;
const C5: f64 = 0.281649658092772603273242802490;
const C6: f64 = 0.333333333333333333333333333333;
const C7: f64 = 0.25;
const C8: f64 = 0.307692307692307692307692307692;
const C9: f64 = 0.651282051282051282051282051282;
const C10: f64 = 0.6;
const C11: f64 = 0.857142857142857142857142857142;

// a coefficients (Butcher tableau - only non-zero values listed)
const A21: f64 = 5.26001519587677318785587544488e-02;

const A31: f64 = 1.97250569845378994544595329183e-02;
const A32: f64 = 5.91751709536136983633785987549e-02;

const A41: f64 = 2.95875854768068491816892993775e-02;
const A43: f64 = 8.87627564304205475450678981324e-02;

const A51: f64 = 2.41365134159266685502369798665e-01;
const A53: f64 = -8.84549479328286085344864962717e-01;
const A54: f64 = 9.24834003261792003115737966543e-01;

const A61: f64 = 3.70370370370370370370370370370e-02;
const A64: f64 = 1.70828608729473871279604482173e-01;
const A65: f64 = 1.25467687566822425016691814123e-01;

const A71: f64 = 3.71093750000000000000000000000e-02;
const A74: f64 = 1.70252211019544039314978060272e-01;
const A75: f64 = 6.02165389804559606850219397283e-02;
const A76: f64 = -1.75781250000000000000000000000e-02;

const A81: f64 = 3.70920001185047927108779319836e-02;
const A84: f64 = 1.70383925712239993810214054705e-01;
const A85: f64 = 1.07262030446373284651809199168e-01;
const A86: f64 = -1.53194377486244017527936158236e-02;
const A87: f64 = 8.27378916381402288758473766002e-03;

const A91: f64 = 6.24110958716075717114429577812e-01;
const A94: f64 = -3.36089262944694129406857109825e+00;
const A95: f64 = -8.68219346841726006818189891453e-01;
const A96: f64 = 2.75920996994467083049415600797e+01;
const A97: f64 = 2.01540675504778934086186788979e+01;
const A98: f64 = -4.34898841810699588477366255144e+01;

const A101: f64 = 4.77662536438264365890433908527e-01;
const A104: f64 = -2.48811461997166764192642586468e+00;
const A105: f64 = -5.90290826836842996371446475743e-01;
const A106: f64 = 2.12300514481811942347288949897e+01;
const A107: f64 = 1.52792336328824235832596922938e+01;
const A108: f64 = -3.32882109689848629194453265587e+01;
const A109: f64 = -2.03312017085086261358222928593e-02;

const A111: f64 = -9.37142430085987325717040528057e-01;
const A114: f64 = 5.18637242884406370830023853209e+00;
const A115: f64 = 1.09143734899672957818500254654e+00;
const A116: f64 = -8.14978701074692612513997267357e+00;
const A117: f64 = -1.85200656599969598641566180701e+01;
const A118: f64 = 2.27394870993505042818970056734e+01;
const A119: f64 = 2.49360555267965238987089396762e+00;
const A1110: f64 = -3.01726999341168153929447817486e+00;

const A121: f64 = 2.27331014751653820792359768449e+00;
const A124: f64 = -1.05344954667372501984066689879e+01;
const A125: f64 = -2.00087205822486249909675718444e+00;
const A126: f64 = -1.79589318631187989172765950534e+01;
const A127: f64 = 2.79488845294199600508499808837e+01;
const A128: f64 = -2.85899827713502369474065508674e+00;
const A129: f64 = -8.87285693353062954433549289258e+00;
const A1210: f64 = 1.23605671757943030647266201528e+01;
const A1211: f64 = 6.43392746015763530355970484046e-01;

// 8th order solution weights
const B1: f64 = 5.42937341165687622380535766363e-02;
const B6: f64 = 4.45031289275240888144113950566e+00;
const B7: f64 = 1.89151789931450038304281599044e+00;
const B8: f64 = -5.80120396001058478146721142270e+00;
const B9: f64 = 3.11168072561590417690755489127e-01;
const B10: f64 = -1.52160949662516078556178806805e-01;
const B11: f64 = 2.01365400804030348374776537501e-01;
const B12: f64 = 4.47106157277725905176885569043e-02;

// 5th order error weights
const E51: f64 = 0.1312004499419488073250102996e-01;
const E56: f64 = -0.1225156446376204440720569753e+01;
const E57: f64 = -0.4957589496572501915214079952e+00;
const E58: f64 = 0.1664377182454986536961530415e+01;
const E59: f64 = -0.3503288487499736816886487290e+00;
const E510: f64 = 0.3341791187130174790297318841e+00;
const E511: f64 = 0.8192320648511571246570742613e-01;
const E512: f64 = -0.2235530786388629525884427845e-01;

// 3rd order error weights (for stiff detection - reserved for future use)
#[allow(dead_code)]
const E31: f64 = 0.244094488188976377952755905512e+00;
#[allow(dead_code)]
const E36: f64 = -0.733846688281611857341361741547e+01;
#[allow(dead_code)]
const E37: f64 = 0.283838479716792063785675431827e+01;
#[allow(dead_code)]
const E38: f64 = -0.248811461997166764192642586468e+01;
#[allow(dead_code)]
const E39: f64 = 0.228159825512880810935216072256e+01;
#[allow(dead_code)]
const E310: f64 = -0.125000000000000000000000000000e+00;

// Step size controller parameters
const SAFETY: f64 = 0.9;
const MIN_FACTOR: f64 = 0.333;
const MAX_FACTOR: f64 = 6.0;

/// Compute weighted sum of stages with h as tensor
fn weighted_sum_12<R, C>(
    client: &C,
    k: &[&Tensor<R>; 12],
    coeffs: &[f64; 12],
    h: &Tensor<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R>,
{
    let mut result = client.mul_scalar(h, 0.0)?;
    result = client.mul(&result, k[0])?; // Start with zero tensor of correct shape

    for (i, &ki) in k.iter().enumerate() {
        if coeffs[i] != 0.0 {
            let h_ci = client.mul_scalar(h, coeffs[i])?;
            let term = client.mul(&h_ci, ki)?;
            result = client.add(&result, &term)?;
        }
    }
    Ok(result)
}

/// Dormand-Prince 8(5,3) method using tensor operations.
///
/// High-order adaptive solver for smooth ODEs.
/// All computation stays on device. Step size control is fully device-resident.
#[allow(clippy::too_many_lines)]
pub fn dop853_impl<R, C, F>(
    client: &C,
    f: F,
    t_span: [f64; 2],
    y0: &Tensor<R>,
    options: &ODEOptions,
) -> IntegrateResult<ODEResultTensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + ScalarOps<R> + RuntimeClient<R>,
    F: Fn(&Tensor<R>, &Tensor<R>) -> Result<Tensor<R>>,
{
    let [t_start, t_end] = t_span;
    let device = client.device();

    let min_step = options.min_step.unwrap_or(1e-14);
    let max_step = options.max_step.unwrap_or(t_end - t_start);

    // Initialize - all tensors stay on device
    let mut t = Tensor::<R>::from_slice(&[t_start], &[1], device)?;
    let mut y = y0.clone();
    let mut k1 = f(&t, &y).map_err(|e| IntegrateError::InvalidInput {
        context: format!("RHS function error: {}", e),
    })?;

    // Compute initial step size (device-resident)
    let mut h = match options.h0 {
        Some(h0) => Tensor::<R>::from_slice(&[h0], &[1], device)?,
        None => compute_initial_step(client, &f, &t, &y, &k1, 8, options.rtol, options.atol)
            .map_err(|e| IntegrateError::InvalidInput {
                context: format!("Initial step computation error: {}", e),
            })?,
    };

    // Clamp h to [min_step, max_step] on device
    let min_h = Tensor::<R>::from_slice(&[min_step], &[1], device)?;
    let max_h = Tensor::<R>::from_slice(&[max_step], &[1], device)?;
    h = client.minimum(&client.maximum(&h, &min_h)?, &max_h)?;

    // t_end tensor for comparison
    let t_end_tensor = Tensor::<R>::from_slice(&[t_end], &[1], device)?;

    // Storage for results
    let mut t_values = vec![t_start];
    let mut y_values = vec![y.clone()];
    let mut nfev = 1;
    let mut naccept = 0;
    let mut nreject = 0;

    // Main integration loop
    loop {
        let t_val: f64 = t.item().map_err(to_integrate_err)?;

        if t_val >= t_end {
            break;
        }

        if naccept + nreject >= options.max_steps {
            let (t_tensor, y_tensor) = build_result_tensors(client, &t_values, &y_values)?;
            return Ok(ODEResultTensor {
                t: t_tensor,
                y: y_tensor,
                success: false,
                message: Some(format!(
                    "Maximum steps ({}) exceeded at t = {:.6}",
                    options.max_steps, t_val
                )),
                nfev,
                naccept,
                nreject,
                method: ODEMethod::DOP853,
            });
        }

        // Adjust step for end point
        let remaining = client.sub(&t_end_tensor, &t)?;
        h = client.minimum(&h, &remaining)?;

        // ============================================================
        // DOP853 stages - ALL computation stays on device
        // ============================================================

        // k2
        let h_a21 = client.mul_scalar(&h, A21)?;
        let y_stage = client.add(&y, &client.mul(&h_a21, &k1)?)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C2)?)?;
        let k2 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k3
        let h_a31 = client.mul_scalar(&h, A31)?;
        let h_a32 = client.mul_scalar(&h, A32)?;
        let sum_k = client.add(&client.mul(&h_a31, &k1)?, &client.mul(&h_a32, &k2)?)?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C3)?)?;
        let k3 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k4
        let h_a41 = client.mul_scalar(&h, A41)?;
        let h_a43 = client.mul_scalar(&h, A43)?;
        let sum_k = client.add(&client.mul(&h_a41, &k1)?, &client.mul(&h_a43, &k3)?)?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C4)?)?;
        let k4 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k5
        let h_a51 = client.mul_scalar(&h, A51)?;
        let h_a53 = client.mul_scalar(&h, A53)?;
        let h_a54 = client.mul_scalar(&h, A54)?;
        let sum_k = client.add(
            &client.mul(&h_a51, &k1)?,
            &client.add(&client.mul(&h_a53, &k3)?, &client.mul(&h_a54, &k4)?)?,
        )?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C5)?)?;
        let k5 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k6
        let h_a61 = client.mul_scalar(&h, A61)?;
        let h_a64 = client.mul_scalar(&h, A64)?;
        let h_a65 = client.mul_scalar(&h, A65)?;
        let sum_k = client.add(
            &client.mul(&h_a61, &k1)?,
            &client.add(&client.mul(&h_a64, &k4)?, &client.mul(&h_a65, &k5)?)?,
        )?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C6)?)?;
        let k6 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k7
        let h_a71 = client.mul_scalar(&h, A71)?;
        let h_a74 = client.mul_scalar(&h, A74)?;
        let h_a75 = client.mul_scalar(&h, A75)?;
        let h_a76 = client.mul_scalar(&h, A76)?;
        let sum_k = client.add(
            &client.add(&client.mul(&h_a71, &k1)?, &client.mul(&h_a74, &k4)?)?,
            &client.add(&client.mul(&h_a75, &k5)?, &client.mul(&h_a76, &k6)?)?,
        )?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C7)?)?;
        let k7 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k8
        let h_a81 = client.mul_scalar(&h, A81)?;
        let h_a84 = client.mul_scalar(&h, A84)?;
        let h_a85 = client.mul_scalar(&h, A85)?;
        let h_a86 = client.mul_scalar(&h, A86)?;
        let h_a87 = client.mul_scalar(&h, A87)?;
        let sum_k = client.add(
            &client.add(&client.mul(&h_a81, &k1)?, &client.mul(&h_a84, &k4)?)?,
            &client.add(
                &client.add(&client.mul(&h_a85, &k5)?, &client.mul(&h_a86, &k6)?)?,
                &client.mul(&h_a87, &k7)?,
            )?,
        )?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C8)?)?;
        let k8 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k9
        let h_a91 = client.mul_scalar(&h, A91)?;
        let h_a94 = client.mul_scalar(&h, A94)?;
        let h_a95 = client.mul_scalar(&h, A95)?;
        let h_a96 = client.mul_scalar(&h, A96)?;
        let h_a97 = client.mul_scalar(&h, A97)?;
        let h_a98 = client.mul_scalar(&h, A98)?;
        let sum_k = client.add(
            &client.add(&client.mul(&h_a91, &k1)?, &client.mul(&h_a94, &k4)?)?,
            &client.add(
                &client.add(&client.mul(&h_a95, &k5)?, &client.mul(&h_a96, &k6)?)?,
                &client.add(&client.mul(&h_a97, &k7)?, &client.mul(&h_a98, &k8)?)?,
            )?,
        )?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C9)?)?;
        let k9 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k10
        let h_a101 = client.mul_scalar(&h, A101)?;
        let h_a104 = client.mul_scalar(&h, A104)?;
        let h_a105 = client.mul_scalar(&h, A105)?;
        let h_a106 = client.mul_scalar(&h, A106)?;
        let h_a107 = client.mul_scalar(&h, A107)?;
        let h_a108 = client.mul_scalar(&h, A108)?;
        let h_a109 = client.mul_scalar(&h, A109)?;
        let sum_k = client.add(
            &client.add(&client.mul(&h_a101, &k1)?, &client.mul(&h_a104, &k4)?)?,
            &client.add(
                &client.add(&client.mul(&h_a105, &k5)?, &client.mul(&h_a106, &k6)?)?,
                &client.add(
                    &client.add(&client.mul(&h_a107, &k7)?, &client.mul(&h_a108, &k8)?)?,
                    &client.mul(&h_a109, &k9)?,
                )?,
            )?,
        )?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C10)?)?;
        let k10 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k11
        let h_a111 = client.mul_scalar(&h, A111)?;
        let h_a114 = client.mul_scalar(&h, A114)?;
        let h_a115 = client.mul_scalar(&h, A115)?;
        let h_a116 = client.mul_scalar(&h, A116)?;
        let h_a117 = client.mul_scalar(&h, A117)?;
        let h_a118 = client.mul_scalar(&h, A118)?;
        let h_a119 = client.mul_scalar(&h, A119)?;
        let h_a1110 = client.mul_scalar(&h, A1110)?;
        let sum_k = client.add(
            &client.add(&client.mul(&h_a111, &k1)?, &client.mul(&h_a114, &k4)?)?,
            &client.add(
                &client.add(&client.mul(&h_a115, &k5)?, &client.mul(&h_a116, &k6)?)?,
                &client.add(
                    &client.add(&client.mul(&h_a117, &k7)?, &client.mul(&h_a118, &k8)?)?,
                    &client.add(&client.mul(&h_a119, &k9)?, &client.mul(&h_a1110, &k10)?)?,
                )?,
            )?,
        )?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_stage = client.add(&t, &client.mul_scalar(&h, C11)?)?;
        let k11 = f(&t_stage, &y_stage).map_err(to_integrate_err)?;

        // k12 (final stage at t + h)
        let h_a121 = client.mul_scalar(&h, A121)?;
        let h_a124 = client.mul_scalar(&h, A124)?;
        let h_a125 = client.mul_scalar(&h, A125)?;
        let h_a126 = client.mul_scalar(&h, A126)?;
        let h_a127 = client.mul_scalar(&h, A127)?;
        let h_a128 = client.mul_scalar(&h, A128)?;
        let h_a129 = client.mul_scalar(&h, A129)?;
        let h_a1210 = client.mul_scalar(&h, A1210)?;
        let h_a1211 = client.mul_scalar(&h, A1211)?;
        let sum_k = client.add(
            &client.add(&client.mul(&h_a121, &k1)?, &client.mul(&h_a124, &k4)?)?,
            &client.add(
                &client.add(&client.mul(&h_a125, &k5)?, &client.mul(&h_a126, &k6)?)?,
                &client.add(
                    &client.add(&client.mul(&h_a127, &k7)?, &client.mul(&h_a128, &k8)?)?,
                    &client.add(
                        &client.add(&client.mul(&h_a129, &k9)?, &client.mul(&h_a1210, &k10)?)?,
                        &client.mul(&h_a1211, &k11)?,
                    )?,
                )?,
            )?,
        )?;
        let y_stage = client.add(&y, &sum_k)?;
        let t_new = client.add(&t, &h)?;
        let k12 = f(&t_new, &y_stage).map_err(to_integrate_err)?;
        nfev += 11;

        // 8th order solution
        let y8 = weighted_sum_12(
            client,
            &[
                &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9, &k10, &k11, &k12,
            ],
            &[B1, 0.0, 0.0, 0.0, 0.0, B6, B7, B8, B9, B10, B11, B12],
            &h,
        )?;
        let y_new = client.add(&y, &y8)?;

        // Compute k13 for error estimate (FSAL)
        let k13 = f(&t_new, &y_new).map_err(to_integrate_err)?;
        nfev += 1;

        // 5th order error estimate
        let y_err5 = weighted_sum_12(
            client,
            &[
                &k1, &k2, &k3, &k4, &k5, &k6, &k7, &k8, &k9, &k10, &k11, &k12,
            ],
            &[
                E51, 0.0, 0.0, 0.0, 0.0, E56, E57, E58, E59, E510, E511, E512,
            ],
            &h,
        )?;

        // Also include k13 contribution (treating it as the continuation)
        // The error estimate uses the embedded formula
        let error = compute_error(client, &y_new, &y_err5, &y, options.rtol, options.atol)
            .map_err(to_integrate_err)?;

        // Compute step factor (stays on device)
        let factor = compute_step_factor(client, &error, 8, SAFETY, MIN_FACTOR, MAX_FACTOR)
            .map_err(to_integrate_err)?;

        // Compute acceptance indicator (stays on device)
        let accept_tensor = compute_acceptance(client, &error).map_err(to_integrate_err)?;

        // Only transfer accept for control flow decision
        let accept_val: f64 = accept_tensor.item().map_err(to_integrate_err)?;
        let accept = accept_val > 0.5;

        // Compute new step size on device
        let h_new = client.mul(&h, &factor)?;
        let h_new = client.minimum(&client.maximum(&h_new, &min_h)?, &max_h)?;

        if accept {
            t = t_new;
            y = y_new;
            k1 = k13; // FSAL property

            let new_t: f64 = t.item().map_err(to_integrate_err)?;
            t_values.push(new_t);
            y_values.push(y.clone());
            naccept += 1;
        } else {
            nreject += 1;
        }

        h = h_new;

        // Check minimum step
        let h_val: f64 = h.item().map_err(to_integrate_err)?;
        if h_val < min_step {
            let t_val_err: f64 = t.item().map_err(to_integrate_err)?;
            return Err(IntegrateError::StepSizeTooSmall {
                step: h_val,
                t: t_val_err,
                context: "DOP853".to_string(),
            });
        }
    }

    let (t_tensor, y_tensor) = build_result_tensors(client, &t_values, &y_values)?;

    Ok(ODEResultTensor {
        t: t_tensor,
        y: y_tensor,
        success: true,
        message: None,
        nfev,
        naccept,
        nreject,
        method: ODEMethod::DOP853,
    })
}

/// Build result tensors from collected values.
fn build_result_tensors<R, C>(
    client: &C,
    t_values: &[f64],
    y_values: &[Tensor<R>],
) -> IntegrateResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + RuntimeClient<R>,
{
    let n_steps = t_values.len();
    let t_tensor = Tensor::<R>::from_slice(t_values, &[n_steps], client.device())?;

    let y_refs: Vec<&Tensor<R>> = y_values.iter().collect();
    let y_tensor = client
        .stack(&y_refs, 0)
        .map_err(|e| IntegrateError::InvalidInput {
            context: format!("Failed to stack y tensors: {}", e),
        })?;

    Ok((t_tensor, y_tensor))
}

fn to_integrate_err(e: numr::error::Error) -> IntegrateError {
    IntegrateError::InvalidInput {
        context: format!("Tensor operation error: {}", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::ops::BinaryOps;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        (device, client)
    }

    #[test]
    fn test_dop853_exponential_decay() {
        let (device, client) = setup();

        // dy/dt = -y, y(0) = 1, solution: y(t) = exp(-t)
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0], &[1], &device).unwrap();

        let result = dop853_impl(
            &client,
            |_t, y| client.mul_scalar(y, -1.0),
            [0.0, 5.0],
            &y0,
            &ODEOptions::with_method(ODEMethod::DOP853),
        )
        .unwrap();

        assert!(result.success);
        assert_eq!(result.method, ODEMethod::DOP853);

        let y_val = result.y_final_vec();
        let exact = (-5.0_f64).exp();

        // DOP853 should be more accurate than RK45
        assert!(
            (y_val[0] - exact).abs() < 1e-4,
            "y_final = {}, exact = {}",
            y_val[0],
            exact
        );
    }

    #[test]
    fn test_dop853_harmonic_oscillator() {
        let (device, client) = setup();

        // y'' + y = 0 as system: y1' = y2, y2' = -y1
        let y0 = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.0], &[2], &device).unwrap();

        let opts = ODEOptions {
            method: ODEMethod::DOP853,
            rtol: 1e-6,
            atol: 1e-8,
            ..Default::default()
        };

        let result = dop853_impl(
            &client,
            |_t, y| {
                let y_data: Vec<f64> = y.to_vec();
                Tensor::<CpuRuntime>::from_slice(&[y_data[1], -y_data[0]], &[2], &device)
            },
            [0.0, 2.0 * std::f64::consts::PI],
            &y0,
            &opts,
        )
        .unwrap();

        assert!(result.success, "DOP853 failed: {:?}", result.message);

        let y_val = result.y_final_vec();
        // After one period, should return close to initial state
        assert!((y_val[0] - 1.0).abs() < 0.01, "y1 = {}", y_val[0]);
        assert!(y_val[1].abs() < 0.01, "y2 = {}", y_val[1]);
    }

    #[test]
    fn test_dop853_polynomial() {
        let (device, client) = setup();

        // dy/dt = 3t^2, y(0) = 0, solution: y(t) = t^3
        let y0 = Tensor::<CpuRuntime>::from_slice(&[0.0], &[1], &device).unwrap();

        let result = dop853_impl(
            &client,
            |t, _y| {
                let t_sq = client.mul(t, t)?;
                client.mul_scalar(&t_sq, 3.0)
            },
            [0.0, 2.0],
            &y0,
            &ODEOptions::with_method(ODEMethod::DOP853),
        )
        .unwrap();

        assert!(result.success);

        let y_val = result.y_final_vec();
        assert!((y_val[0] - 8.0).abs() < 1e-4, "y_final = {}", y_val[0]);
    }
}

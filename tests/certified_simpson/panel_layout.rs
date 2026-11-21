use super::admission::ones;
use super::callback::Client;
use super::fixture::{self, Result};
use numr::prelude::*;
use solvr::integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions};
use std::cell::RefCell;

/// Panel `j` of each lane spans the lower bound of value node `2j` to the upper bound of node `2j+2`.
pub fn run<R, C>(client: &C) -> Result
where
    R: Runtime<DType = DType>,
    C: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    for dtype in [DType::F32, DType::F64] {
        let left = fixture::tensor::<R>(&[0.0, -0.75, 2.0, 0.25], &[4], dtype, client.device())?;
        let right = fixture::tensor::<R>(&[1.0, 3.5, 2.0, 0.3], &[4], dtype, client.device())?;
        for panels in [1, 4, 16] {
            let nodes = RefCell::new(None);
            client.certified_simpson(
                &left,
                &right,
                CertifiedSimpsonOptions {
                    panels,
                    max_points: 1 << 12,
                },
                |query| {
                    *nodes.borrow_mut() = Some((
                        query.owner.try_to_vec::<i64>()?,
                        fixture::read_bits(&query.lower)?,
                        fixture::read_bits(&query.upper)?,
                    ));
                    ones(client, query)
                },
                |query| {
                    let (owner, lower, upper) = nodes.borrow_mut().take().expect("values first");
                    let panel_owner = query.owner.try_to_vec::<i64>()?;
                    let panel_lower = fixture::read_bits(&query.lower)?;
                    let panel_upper = fixture::read_bits(&query.upper)?;
                    let stride = 2 * panels + 1;
                    let lanes = owner.len() / stride;
                    assert_eq!(lanes, 3, "zero-width lanes take no queries");
                    assert_eq!(panel_lower.len(), lanes * panels);
                    for lane in 0..lanes {
                        for j in 0..panels {
                            let panel = lane * panels + j;
                            let start = lane * stride + 2 * j;
                            assert_eq!(panel_owner[panel], owner[start]);
                            assert_eq!(panel_lower[panel], lower[start], "panels={panels}");
                            assert_eq!(panel_upper[panel], upper[start + 2], "panels={panels}");
                        }
                    }
                    Ok(Tensor::zeros(query.lower.shape(), dtype, client.device())?)
                },
            )?;
        }
    }
    Ok(())
}

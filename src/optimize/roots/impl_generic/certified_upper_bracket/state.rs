use super::super::certified::native::{self, RootClient};
use crate::DType;
use crate::optimize::OptimizeResult;
use crate::optimize::roots::traits::{CertifiedRootValues, CertifiedUpperBracketStatus};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) struct State<R: Runtime<DType = DType>> {
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
    pub lower_residual: CertifiedRootValues<R>,
    pub upper_residual: CertifiedRootValues<R>,
    pub status: Tensor<R>,
    pub expansions: Tensor<R>,
    pub active: Tensor<R>,
    pub owners: Tensor<R>,
    pub callback_points: usize,
    pub expansion_rounds: usize,
}

impl<R: Runtime<DType = DType>> State<R> {
    pub fn new<C: RootClient<R>>(
        client: &C,
        lower: Tensor<R>,
        upper: Tensor<R>,
        lower_residual: CertifiedRootValues<R>,
        upper_residual: CertifiedRootValues<R>,
        owners: Tensor<R>,
        callback_points: usize,
    ) -> OptimizeResult<Self> {
        let len = lower.numel();
        let status = Tensor::full_scalar(
            &[len],
            DType::I64,
            CertifiedUpperBracketStatus::WorkLimit as i64 as f64,
            client.device(),
        )?;
        let expansions = Tensor::zeros(&[len], DType::I64, client.device())?;
        let active = Tensor::ones(&[len], lower.dtype(), client.device())?;
        Ok(Self {
            lower,
            upper,
            lower_residual,
            upper_residual,
            status,
            expansions,
            active,
            owners,
            callback_points,
            expansion_rounds: 0,
        })
    }

    pub fn finish<C: RootClient<R>>(
        &mut self,
        client: &C,
        mask: &Tensor<R>,
        status: CertifiedUpperBracketStatus,
    ) -> OptimizeResult<()> {
        let selected = client.mul(mask, &self.active)?;
        let indices = native::indices(client, &selected)?;
        if indices.numel() == 0 {
            return Ok(());
        }
        let codes = Tensor::full_scalar(
            &[indices.numel()],
            DType::I64,
            status as i64 as f64,
            client.device(),
        )?;
        self.status = native::replace(client, &self.status, &indices, &codes)?;
        let zeros = Tensor::zeros(&[indices.numel()], self.active.dtype(), client.device())?;
        self.active = native::replace(client, &self.active, &indices, &zeros)?;
        Ok(())
    }
}

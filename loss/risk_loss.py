import torch
import torch.nn as nn


class NetSortinoLoss(nn.Module):
    """
    Turnover-penalized Sortino objective.

    FIX (see GAECO-Net paper, Section 2.5 / Limitations #1):
    `pipeline/train.py` calls this module as
        criterion(weights=weights, next_returns=forward_returns,
                   realized_cov=C_clean, prev_weights=prev_weights)
    but the previous version of this class only accepted
    `forward(self, weights, asset_returns)`, which raised a TypeError
    before a single training step could run. This version accepts the
    full call signature actually used by the training loop, while
    remaining backward compatible with the old 2-positional-argument
    call style (`criterion(weights, asset_returns)`).
    """

    def __init__(
        self,
        fee_rate: float = 0.001,
        return_weight: float = 3.0,
        eps: float = 1e-6,
        risk_lambda: float = 0.0,
        turnover_weight: float | None = None,
        downside_ema_momentum: float = 0.98,
        downside_var_init: float = 1e-4,
    ):
        super().__init__()
        self.fee_rate = fee_rate
        self.return_weight = return_weight
        self.eps = eps
        self.risk_lambda = risk_lambda
        self.turnover_weight = turnover_weight if turnover_weight is not None else fee_rate

        # FIX (root cause of "GAECO and GAECO-Explained both bad / statistically
        # indistinguishable from noise"): pipeline/train.py calls this loss once
        # per single time step (batch size 1). With a single sample,
        # downside_std was computed fresh each step from ONE realized return:
        # whenever that one sample happened to be positive, negative_returns
        # clamps to 0, so downside_std collapsed to sqrt(eps) (~0.001), and
        # sortino_ratio = mean_ret / downside_std spiked to huge values on
        # that step -- dominating return_weight*mean_ret and the turnover
        # penalty and producing wildly unstable, effectively noisy gradient
        # updates. This wasn't a new bug from a prior fix; it's inherent to
        # computing a "Sortino ratio" from n=1 observation.
        #
        # Fix: maintain an exponential moving average of the downside
        # variance across training steps (detached from the autograd graph,
        # like a BatchNorm running statistic) and use THAT as the Sortino
        # denominator. mean_ret in the numerator is still fully
        # differentiable per step; only the (now stable, slowly-evolving)
        # denominator is treated as a constant for gradient purposes. This
        # is the same trick BatchNorm uses for its running variance, applied
        # here because there is no batch to average over within a step.
        self.downside_ema_momentum = downside_ema_momentum
        # FIX: a torch buffer mutated in-place (`.mul_().add_()`) while also
        # being read into the same forward pass's autograd graph (via
        # torch.sqrt(buffer + eps) -> sortino_ratio) can trip PyTorch's
        # in-place-modification version check depending on exactly how the
        # training loop batches/accumulates gradients around this call --
        # this is what threw the error. A plain Python float has no
        # autograd graph and can never trigger that check: the EMA is
        # updated with ordinary Python arithmetic, and only converted to a
        # tensor (or left as a Python scalar, which torch happily
        # broadcasts against) at the point it's divided into mean_ret.
        self._downside_var_ema = float(downside_var_init)

    def reset_downside_ema(self, value: float | None = None):
        """
        Call this between training phases (e.g. after CBB synthetic
        pre-training, before real-data fine-tuning) so the EMA doesn't carry
        over a downside-variance estimate calibrated to one data
        distribution's volatility scale into a differently-scaled one.
        """
        self._downside_var_ema = float(value) if value is not None else 1e-4

    def forward(
        self,
        weights: torch.Tensor,
        next_returns: torch.Tensor | None = None,
        realized_cov: torch.Tensor | None = None,
        prev_weights: torch.Tensor | None = None,
        asset_returns: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        weights:       [Batch, N] portfolio weights proposed for this step.
        next_returns:  [Batch, N] forward (e.g. 10-day cumulative) per-asset
                       return target the weights are being scored against.
                       `asset_returns` is accepted as an alias for backward
                       compatibility with the old call signature; if both
                       are omitted this raises, since a return target is
                       required to compute the objective.
        realized_cov:  [Batch, N, N] the cleaned covariance/correlation
                       matrix (C_clean) produced by the model for this
                       step. Optional: when provided, an explicit
                       ex-ante variance term gamma * w^T C w is added to
                       the loss so the network is penalized for taking on
                       *predicted* risk, not only realized downside
                       deviation within the batch (which, for a batch of
                       size 1, is not a meaningful volatility estimate on
                       its own).
        prev_weights:  [Batch, N] the previous step's (post-drift) target
                       weights, used for the turnover penalty. If omitted,
                       turnover is computed across the batch/time
                       dimension of `weights` instead (old behavior).
        """
        target = next_returns if next_returns is not None else asset_returns
        if target is None:
            raise ValueError(
                "NetSortinoLoss.forward requires either `next_returns` or "
                "`asset_returns` (return target) to be provided."
            )

        # Portfolio return(s) implied by these weights against the target.
        # Supports both a single forward-horizon return per batch element
        # ([Batch, N] . [Batch, N] -> [Batch]) and a full return sequence
        # ([Batch, T, N] -> [Batch, T]) for backward compatibility with
        # callers that pass a multi-day return path.
        if target.dim() == weights.dim():
            port_returns = torch.sum(weights * target, dim=-1)
        else:
            port_returns = torch.sum(weights.unsqueeze(1) * target, dim=-1)

        mean_ret = torch.mean(port_returns)

        # Downside deviation (Sortino denominator), stabilized via EMA --
        # see the __init__ comment for why a per-step, n=1 estimate is
        # unusable here, and why this EMA is kept as a plain Python float
        # rather than a tensor buffer.
        negative_returns = torch.clamp(port_returns, max=0.0)
        batch_downside_var = torch.mean(negative_returns ** 2).item()
        if self.training:
            self._downside_var_ema = (
                self.downside_ema_momentum * self._downside_var_ema
                + (1.0 - self.downside_ema_momentum) * batch_downside_var
            )
        downside_std = (self._downside_var_ema + self.eps) ** 0.5
        sortino_ratio = mean_ret / downside_std

        # Ex-ante variance term from the model's own cleaned covariance
        # estimate, w^T C w, averaged over the batch. This is what lets
        # the model be penalized for predicted risk even when a single
        # realized-return sample can't estimate variance reliably.
        variance_term = torch.zeros((), device=weights.device, dtype=weights.dtype)
        if realized_cov is not None and self.risk_lambda > 0.0:
            w = weights if weights.dim() == realized_cov.dim() - 1 else weights.unsqueeze(1)
            quad = torch.einsum("...i,...ij,...j->...", w, realized_cov, w)
            variance_term = self.risk_lambda * torch.mean(quad)

        # Turnover penalty: prefer the explicit prev_weights argument
        # (drifted target weights from the previous rebalance) when given;
        # otherwise fall back to differencing along the weights tensor's
        # own time/batch dimension (old behavior, used when training is
        # driven by a full sequence rather than one step at a time).
        if prev_weights is not None:
            turnover_penalty = torch.mean(torch.abs(weights - prev_weights)) * self.turnover_weight
        elif weights.dim() >= 2 and weights.size(0) > 1:
            weight_diff = torch.abs(weights[1:] - weights[:-1])
            turnover_penalty = torch.mean(weight_diff) * self.turnover_weight
        else:
            turnover_penalty = torch.zeros((), device=weights.device, dtype=weights.dtype)

        # Negative loss for gradient descent (maximizing return + Sortino,
        # minimizing predicted variance and turnover/fee drag).
        loss = -(self.return_weight * mean_ret + sortino_ratio) + variance_term + turnover_penalty
        return loss
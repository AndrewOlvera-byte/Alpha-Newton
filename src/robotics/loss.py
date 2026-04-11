"""BC loss functions for visuomotor policies."""
import torch


def gaussian_nll_loss(
    pred_mu: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor,
    log_var_clamp: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian negative log-likelihood loss with learned per-dim variance.

    Returns (mean_loss, per_sample_loss) to match the existing API.
    """
    # Clamp log_var for numerical stability
    log_var = pred_log_var.clamp(-log_var_clamp, log_var_clamp)
    # NLL = 0.5 * (log_var + (mu - target)^2 / exp(log_var))
    nll = 0.5 * (log_var + (pred_mu - target).pow(2) / log_var.exp())
    per_sample = nll.mean(dim=(-1, -2))  # [B]
    return per_sample.mean(), per_sample.detach()

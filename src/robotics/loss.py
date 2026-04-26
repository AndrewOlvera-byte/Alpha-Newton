"""Loss functions for robotics BC and PPO policies."""
import torch


def ppo_clip_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Clipped PPO surrogate objective (returns a loss to minimize)."""
    ratio = (log_probs - old_log_probs).exp()
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
    return -torch.min(ratio * advantages, clipped * advantages).mean()


def value_loss(
    values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None = None,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Optionally clipped value function loss."""
    if old_values is not None:
        clipped_values = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
        loss_unclipped = (values - returns).pow(2)
        loss_clipped = (clipped_values - returns).pow(2)
        return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()
    return 0.5 * (values - returns).pow(2).mean()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation.

    Args:
        rewards: [T, N] rewards per step per env
        values: [T, N] value estimates per step per env
        dones: [T, N] episode done flags (1.0 = done)
        last_values: [N] bootstrap values for the last step
        gamma: discount factor
        gae_lambda: GAE lambda

    Returns:
        (advantages, returns) each [T, N]
    """
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        next_values = last_values if t == T - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def gaussian_nll_loss(
    pred_mu: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor,
    log_var_clamp: float = 6.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gaussian negative log likelihood with learned per-dim variance."""

    log_var = pred_log_var.clamp(-log_var_clamp, log_var_clamp)

    nll = 0.5 * (log_var + (pred_mu - target).pow(2) / log_var.exp())
    per_sample = nll.mean(dim=(-1, -2))
    return per_sample.mean(), per_sample.detach()

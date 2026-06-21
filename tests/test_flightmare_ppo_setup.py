import numpy as np
import pytest
import torch

from src.robotics.flightmare_ppo_trainer import (
    FlightmarePPOTrainer,
    FlightmareRolloutBuffer,
    collect_flightmare_rollouts,
)
from src.robotics.loss import compute_gae, ppo_clip_loss, value_loss
from src.robotics.models.flightmare.GaussianActionExpert import GaussianActionExpert


def test_ppo_clipped_surrogate_has_expected_sign_and_clipping():
    old = torch.zeros(4)
    new = torch.log(torch.tensor([1.30, 1.05, 0.70, 0.95]))
    advantages = torch.tensor([2.0, 2.0, -2.0, -2.0])

    loss = ppo_clip_loss(new, old, advantages, clip_eps=0.2)

    expected_terms = torch.tensor([
        min(1.30 * 2.0, 1.20 * 2.0),
        min(1.05 * 2.0, 1.20 * 2.0),
        min(0.70 * -2.0, 0.80 * -2.0),
        min(0.95 * -2.0, 0.95 * -2.0),
    ])
    assert loss == pytest.approx(-expected_terms.mean().item())


def test_value_loss_uses_clipped_value_update_pessimistically():
    values = torch.tensor([2.0])
    returns = torch.tensor([3.0])
    old_values = torch.tensor([0.0])

    loss = value_loss(values, returns, old_values=old_values, clip_eps=0.2)

    unclipped = (2.0 - 3.0) ** 2
    clipped = (0.2 - 3.0) ** 2
    assert loss == pytest.approx(0.5 * max(unclipped, clipped))


def test_gae_respects_done_boundaries_and_bootstraps_last_value_only_when_alive():
    rewards = torch.tensor([[1.0], [1.0], [1.0]])
    values = torch.zeros_like(rewards)
    last_values = torch.tensor([10.0])

    no_done_adv, _ = compute_gae(
        rewards,
        values,
        dones=torch.zeros_like(rewards),
        last_values=last_values,
        gamma=1.0,
        gae_lambda=1.0,
    )
    terminal_adv, _ = compute_gae(
        rewards,
        values,
        dones=torch.tensor([[0.0], [1.0], [0.0]]),
        last_values=last_values,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert no_done_adv.flatten().tolist() == pytest.approx([13.0, 12.0, 11.0])
    assert terminal_adv.flatten().tolist() == pytest.approx([2.0, 1.0, 11.0])


def test_advantage_normalization_centers_and_scales_flat_batch():
    buffer = FlightmareRolloutBuffer(
        n_steps=2,
        n_envs=3,
        state_dim=4,
        action_dim=2,
        device=torch.device("cpu"),
    )
    buffer.advantages = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    buffer.normalize_advantages()

    assert buffer.advantages.mean().item() == pytest.approx(0.0, abs=1e-7)
    assert buffer.advantages.std().item() == pytest.approx(1.0, rel=1e-6)


class _ClipAwareModel(torch.nn.Module):
    action_dim = 4

    def act(self, batch):
        n = batch["state"].shape[0]
        action = torch.full((n, self.action_dim), 2.0, device=batch["state"].device)
        log_prob = torch.full((n,), -999.0, device=batch["state"].device)
        value = torch.zeros(n, device=batch["state"].device)
        return action, log_prob, value

    def evaluate_actions(self, batch, actions):
        log_prob = actions.sum(dim=-1)
        entropy = torch.zeros(actions.shape[0], device=actions.device)
        value = torch.full((actions.shape[0],), 0.5, device=actions.device)
        return log_prob, entropy, value


class _NeverDoneVecEnv:
    def __init__(self, n_envs=2, state_dim=6, action_dim=4):
        self.n_envs = n_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions = []

    def step(self, actions):
        self.actions.append(np.asarray(actions, dtype=np.float32).copy())
        obs = [
            {"state": np.full(self.state_dim, i + 1, dtype=np.float32), "images": {}}
            for i in range(self.n_envs)
        ]
        rewards = np.ones(self.n_envs, dtype=np.float32)
        terms = np.zeros(self.n_envs, dtype=bool)
        truncs = np.zeros(self.n_envs, dtype=bool)
        infos = [
            {
                "action_norm": actions[i].astype(np.float32),
                "speed_mps": 1.0,
                "reward_terms": {"total": 1.0},
            }
            for i in range(self.n_envs)
        ]
        return obs, rewards, terms, truncs, infos

    def reset_at(self, indices, seed=None):
        raise AssertionError("episodes should not end in this test")


def test_rollout_stores_log_prob_and_value_for_applied_clipped_action():
    n_envs = 2
    action_dim = 4
    state_dim = 6
    obs = [{"state": np.zeros(state_dim, dtype=np.float32), "images": {}} for _ in range(n_envs)]
    prev = np.zeros((n_envs, action_dim), dtype=np.float32)
    buffer = FlightmareRolloutBuffer(
        n_steps=1,
        n_envs=n_envs,
        state_dim=state_dim,
        action_dim=action_dim,
        device=torch.device("cpu"),
    )
    vec_env = _NeverDoneVecEnv(n_envs=n_envs, state_dim=state_dim, action_dim=action_dim)

    _, next_prev, stats = collect_flightmare_rollouts(
        vec_env,
        _ClipAwareModel(),
        obs,
        prev,
        buffer,
        n_steps=1,
        device=torch.device("cpu"),
        action_clip=1.0,
    )

    np.testing.assert_allclose(vec_env.actions[0], np.ones((n_envs, action_dim), dtype=np.float32))
    np.testing.assert_allclose(next_prev, np.ones((n_envs, action_dim), dtype=np.float32))
    assert torch.allclose(buffer.actions, torch.ones_like(buffer.actions))
    assert torch.allclose(buffer.log_probs, torch.full((1, n_envs), float(action_dim)))
    assert torch.allclose(buffer.values, torch.full((1, n_envs), 0.5))
    assert stats["action_clip_fraction"] == pytest.approx(1.0)


def test_gaussian_action_head_sample_and_evaluate_are_log_prob_consistent():
    torch.manual_seed(0)
    head = GaussianActionExpert(
        feature_dim=5,
        action_dim=4,
        hidden_dim=8,
        depth=1,
        state_dependent_std=False,
        init_log_std=-2.3,
        log_std_min=-4.0,
        log_std_max=-1.6,
        with_critic=True,
        action_clip=1.35,
        squash_actions=False,
    )
    feature = torch.randn(7, 5)

    action, sample_log_prob, mu, log_std = head.sample(feature)
    eval_log_prob, entropy, eval_mu, eval_log_std = head.evaluate(feature, action)
    value = head.value(feature)

    assert action.shape == (7, 4)
    assert sample_log_prob.shape == (7,)
    assert entropy.shape == (7,)
    assert value.shape == (7,)
    assert torch.allclose(sample_log_prob, eval_log_prob, atol=1e-6)
    assert torch.allclose(mu, eval_mu)
    assert torch.allclose(log_std, eval_log_std)
    assert float(log_std.detach().max()) <= -1.6
    assert float(log_std.detach().min()) >= -4.0


class _TinyActorCritic(torch.nn.Module):
    action_dim = 2

    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Linear(3, 2)
        self.head = torch.nn.Module()
        self.head.critic = torch.nn.Linear(3, 1)


def test_trainer_actor_critic_optimizer_split_is_disjoint_and_complete():
    model = _TinyActorCritic()
    trainer = FlightmarePPOTrainer(
        model=model,
        training_cfg={},
        robotics_cfg={"ppo": {}, "architecture": {}},
        data_cfg={},
    )

    actor_params, critic_params = trainer._split_params()

    actor_ids = {id(p) for p in actor_params}
    critic_ids = {id(p) for p in critic_params}
    all_ids = {id(p) for p in model.parameters() if p.requires_grad}
    assert actor_ids
    assert critic_ids
    assert actor_ids.isdisjoint(critic_ids)
    assert actor_ids | critic_ids == all_ids

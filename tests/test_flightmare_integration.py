import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.flightmare_bc.controllers import QuadParams
from scripts.flightmare_bc.mission import MISSION_DIM
from scripts.flightmare_bc import diag_obs_consistency, diag_policy_stages
from src.core.config import Config
from src.entrypoints.eval_flightmare import _apply_plant_overrides
from src.robotics import flightmare_envs
from src.robotics.curriculum import Curriculum
from src.robotics.flightmare_envs import FlightmareRacingEnv, build_flightmare_env_config
from src.robotics.flightmare_ppo_trainer import (
    FlightmarePPOTrainer,
    FlightmareRolloutBuffer,
    collect_flightmare_rollouts,
)
from src.robotics.rewards import flightmare_racing_v3


def _write_norm_stats(data_dir):
    np.savez(
        data_dir / "norm_stats.npz",
        state_mean=np.zeros(13, dtype=np.float32),
        state_std=np.ones(13, dtype=np.float32),
        mission_mean=np.zeros(MISSION_DIM, dtype=np.float32),
        mission_std=np.ones(MISSION_DIM, dtype=np.float32),
        ctbr_mean=np.array([0.3, 0.0, 0.0, 0.0], dtype=np.float32),
        ctbr_std=np.ones(4, dtype=np.float32),
        ctbr_low=np.array([0.1, -3.0, -4.0, -1.0], dtype=np.float32),
        ctbr_high=np.array([0.5, 3.0, 4.0, 1.0], dtype=np.float32),
        action_normalization=np.array("bounds"),
        proprio_core_mean=np.zeros(9, dtype=np.float32),
        proprio_core_std=np.ones(9, dtype=np.float32),
        gate_mean=np.zeros(24, dtype=np.float32),
        gate_std=np.ones(24, dtype=np.float32),
        aux_mean=np.zeros(3, dtype=np.float32),
        aux_std=np.ones(3, dtype=np.float32),
    )


def test_env_applies_racing_plant_and_action_bounds_override(tmp_path, monkeypatch):
    _write_norm_stats(tmp_path)
    captured = {}

    class FakeFlightmareExpertEnv:
        using_fallback = False

        def __init__(self, **kwargs):
            captured["params"] = kwargs["params"]

        def close(self):
            pass

    monkeypatch.setattr(flightmare_envs, "FlightmareExpertEnv", FakeFlightmareExpertEnv)

    cfg = build_flightmare_env_config(
        data_dir=str(tmp_path),
        obs_schema="v3",
        action_type="ctbr",
        include_mission=False,
        max_collective_thrust_g=5.5,
        action_clip=1.5,
        plant={
            "mass": 0.9,
            "omega_max_body": [18.0, 18.0, 18.0],
            "thrust_map": [1.0e-6, 2.0e-3, -1.0],
        },
        action_bounds_override={
            "ctbr": {
                "low": [0.0, -8.0, -8.0, -4.0],
                "high": [0.8, 8.0, 8.0, 4.0],
            }
        },
    )
    env = FlightmareRacingEnv(cfg)
    try:
        assert captured["params"] is env.params
        assert env.params.mass == pytest.approx(0.9)
        assert env.params.omega_max_body == (18.0, 18.0, 18.0)
        assert env.params.thrust_map == (1.0e-6, 2.0e-3, -1.0)
        assert env.params.max_collective_thrust == pytest.approx(5.5 * 0.9 * env.params.g)
        np.testing.assert_allclose(env.action_space.low, np.full(4, -1.5, dtype=np.float32))
        np.testing.assert_allclose(env.action_space.high, np.full(4, 1.5, dtype=np.float32))
        np.testing.assert_allclose(env.norm.action_low, np.array([0.0, -8.0, -8.0, -4.0]))
        np.testing.assert_allclose(env.norm.action_high, np.array([0.8, 8.0, 8.0, 4.0]))
    finally:
        env.close()


def test_eval_applies_same_plant_overrides_before_thrust_scaling():
    params = _apply_plant_overrides(
        QuadParams(),
        {
            "mass": 0.82,
            "inertia": [0.003, 0.004, 0.005],
            "omega_max_body": [18.0, 17.0, 16.0],
            "thrust_map": [1.1e-6, 0.003, -1.5],
        },
    )
    params.max_collective_thrust = 5.5 * params.mass * params.g

    assert params.mass == pytest.approx(0.82)
    assert params.inertia == (0.003, 0.004, 0.005)
    assert params.omega_max_body == (18.0, 17.0, 16.0)
    assert params.thrust_map == (1.1e-6, 0.003, -1.5)
    assert params.max_collective_thrust == pytest.approx(5.5 * 0.82 * params.g)


def test_flightmare_v3_speed_limit_penalizes_overspeed():
    base_info = {
        "gate_normal_progress_m": 0.1,
        "segment_progress_m": 0.1,
        "gate_center_error_progress": 0.0,
        "gate_center_error_norm_next": 0.0,
        "gate_signed_distance_next_m": -10.0,
        "gate_normal_velocity_mps": 3.0,
        "gate_alignment": 1.0,
        "omega": np.zeros(3, dtype=np.float32),
        "vertical_speed_mps": 0.0,
    }
    slow_info = dict(base_info, speed_mps=7.0)
    fast_info = dict(base_info, speed_mps=10.0)

    slow = flightmare_racing_v3(info=slow_info, speed_limit_mps=7.5, speed_limit_penalty=0.1)
    fast = flightmare_racing_v3(info=fast_info, speed_limit_mps=7.5, speed_limit_penalty=0.1)

    assert slow_info["reward_terms"]["speed_limit"] == pytest.approx(0.0)
    assert fast_info["reward_terms"]["speed_limit"] == pytest.approx(-0.625)
    assert fast < slow


class _DummyRolloutModel(torch.nn.Module):
    action_dim = 4

    def act(self, batch):
        n = batch["state"].shape[0]
        action = torch.full((n, self.action_dim), 2.0, device=batch["state"].device)
        log_prob = torch.zeros(n, device=batch["state"].device)
        value = torch.zeros(n, device=batch["state"].device)
        return action, log_prob, value

    def evaluate_actions(self, batch, actions):
        n = batch["state"].shape[0]
        return (
            torch.zeros(n, device=batch["state"].device),
            torch.zeros(n, device=batch["state"].device),
            torch.zeros(n, device=batch["state"].device),
        )


class _FakeVecEnv:
    def __init__(self, n_envs=2, state_dim=36, action_dim=4):
        self.n_envs = n_envs
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions = []
        self._step = 0

    def step(self, actions):
        self.actions.append(np.asarray(actions, dtype=np.float32).copy())
        self._step += 1
        obs = [
            {"state": np.full(self.state_dim, self._step + i, dtype=np.float32), "images": {}}
            for i in range(self.n_envs)
        ]
        rewards = np.ones(self.n_envs, dtype=np.float32)
        terms = np.zeros(self.n_envs, dtype=bool)
        truncs = np.zeros(self.n_envs, dtype=bool)
        infos = [
            {
                "action_norm": actions[i].astype(np.float32),
                "speed_mps": 8.0,
                "reward_terms": {"total": 1.0, "speed_limit": -0.1},
                "trajectory_replay_used": False,
            }
            for i in range(self.n_envs)
        ]
        return obs, rewards, terms, truncs, infos

    def reset_at(self, indices, seed=None):
        raise AssertionError("test env should not finish episodes")


class _DoneFakeVecEnv(_FakeVecEnv):
    def step(self, actions):
        obs, rewards, _, truncs, infos = super().step(actions)
        terms = np.ones(self.n_envs, dtype=bool)
        for i, info in enumerate(infos):
            info["success"] = bool(i == 0)
            info["gate_completion"] = 0.5
            info["goal_completion"] = 1.0
            info["start_gate_index"] = 2 * i
            info["trajectory_replay_used"] = bool(i == 0)
        return obs, rewards, terms, truncs, infos

    def reset_at(self, indices, seed=None):
        obs = [
            {"state": np.full(self.state_dim, 10 + i, dtype=np.float32), "images": {}}
            for i in indices
        ]
        infos = [
            {"initial_prev_action_norm": np.zeros(self.action_dim, dtype=np.float32)}
            for _ in indices
        ]
        return obs, infos


def test_rollout_collection_clips_actor_actions_and_tracks_prev_action():
    n_envs = 2
    state_dim = 36
    action_dim = 4
    obs = [{"state": np.zeros(state_dim, dtype=np.float32), "images": {}} for _ in range(n_envs)]
    prev = np.zeros((n_envs, action_dim), dtype=np.float32)
    buffer = FlightmareRolloutBuffer(
        n_steps=3,
        n_envs=n_envs,
        state_dim=state_dim,
        action_dim=action_dim,
        device=torch.device("cpu"),
    )
    vec_env = _FakeVecEnv(n_envs=n_envs, state_dim=state_dim, action_dim=action_dim)

    _, next_prev, stats = collect_flightmare_rollouts(
        vec_env,
        _DummyRolloutModel(),
        obs,
        prev,
        buffer,
        n_steps=3,
        device=torch.device("cpu"),
        action_clip=1.0,
    )

    assert stats["action_clip_fraction"] == pytest.approx(1.0)
    assert stats["reward_terms"]["speed_limit"] == pytest.approx(-0.1)
    np.testing.assert_allclose(next_prev, np.ones((n_envs, action_dim), dtype=np.float32))
    for actions in vec_env.actions:
        np.testing.assert_allclose(actions, np.ones((n_envs, action_dim), dtype=np.float32))
    assert torch.allclose(buffer.actions, torch.ones_like(buffer.actions))


def test_rollout_collection_counts_completed_episode_start_gates():
    n_envs = 2
    state_dim = 36
    action_dim = 4
    obs = [{"state": np.zeros(state_dim, dtype=np.float32), "images": {}} for _ in range(n_envs)]
    prev = np.zeros((n_envs, action_dim), dtype=np.float32)
    buffer = FlightmareRolloutBuffer(
        n_steps=2,
        n_envs=n_envs,
        state_dim=state_dim,
        action_dim=action_dim,
        device=torch.device("cpu"),
    )

    _, _, stats = collect_flightmare_rollouts(
        _DoneFakeVecEnv(n_envs=n_envs, state_dim=state_dim, action_dim=action_dim),
        _DummyRolloutModel(),
        obs,
        prev,
        buffer,
        n_steps=2,
        device=torch.device("cpu"),
        action_clip=1.0,
    )

    assert stats["n_episodes"] == 4
    assert stats["start_gate_counts"] == {"0": 2, "2": 2}
    assert stats["trajectory_replay_episodes"] == 2
    assert stats["trajectory_replay_rate"] == pytest.approx(0.5)


def test_reference_state_start_jitter_offsets_position():
    cfg = build_flightmare_env_config(
        data_dir="data/flightmare/bc_v5",
        start_offset_range=[3.0, 3.0],
        start_lateral_range_m=[0.5, 0.5],
        start_vertical_range_m=[-0.25, -0.25],
        start_yaw_noise_rad=0.0,
        start_speed_noise_mps=0.0,
        reference_speed_mps=3.0,
    )
    env = object.__new__(FlightmareRacingEnv)
    env.cfg = cfg
    env.rng = np.random.default_rng(0)
    env._gates = [
        {
            "pos": np.array([10.0, 0.0, 2.0], dtype=np.float64),
            "yaw": 0.0,
            "size": np.array([1.6, 1.6, 1.6], dtype=np.float64),
        }
    ]

    pos, vel, _, omega = env._reference_start_state(0)

    np.testing.assert_allclose(pos, np.array([7.0, 0.5, 1.75]), atol=1e-6)
    np.testing.assert_allclose(vel, np.array([3.0, 0.0, 0.0]), atol=1e-6)
    np.testing.assert_allclose(omega, np.zeros(3), atol=1e-6)


def test_course_start_sample_prob_anchors_reference_stage_to_eval_start(tmp_path, monkeypatch):
    _write_norm_stats(tmp_path)
    calls = {"set_state": 0, "reset_positions": []}

    class FakeFlightmareExpertEnv:
        using_fallback = False

        def __init__(self, **kwargs):
            pass

        def add_gates(self, gates):
            self.gates = gates

        def reset(self, init_pos, yaw):
            calls["reset_positions"].append(np.asarray(init_pos, dtype=np.float64).copy())
            return SimpleNamespace(
                pos=np.asarray(init_pos, dtype=np.float64),
                vel=np.zeros(3, dtype=np.float64),
                quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                omega=np.zeros(3, dtype=np.float64),
                images={},
                done=False,
            )

        def set_state(self, **kwargs):
            calls["set_state"] += 1
            return SimpleNamespace(
                pos=np.asarray(kwargs["pos"], dtype=np.float64),
                vel=np.asarray(kwargs["vel"], dtype=np.float64),
                quat=np.asarray(kwargs["quat"], dtype=np.float64),
                omega=np.asarray(kwargs["omega"], dtype=np.float64),
                images={},
                done=False,
            )

        def close(self):
            pass

    monkeypatch.setattr(flightmare_envs, "FlightmareExpertEnv", FakeFlightmareExpertEnv)

    cfg = build_flightmare_env_config(
        data_dir=str(tmp_path),
        obs_schema="v3",
        action_type="ctbr",
        include_mission=False,
        reset_mode="trajectory_replay_or_reference",
        course_start_sample_prob=1.0,
        start_gate_choices=[4],
        goal_mode="gate_i_to_i+1",
        course_mode="swift_v4",
        num_gates=7,
        gate_size=1.6,
        gate_approach_m=3.0,
        inverted_roll_jitter_rad=0.0,
        backend="numpy",
    )
    env = FlightmareRacingEnv(cfg)
    try:
        _, info = env.reset(seed=7)
    finally:
        env.close()

    assert info["reset_mode"] == "trajectory_replay_or_reference"
    assert info["effective_reset_mode"] == "course_start"
    assert info["course_start_used"] is True
    assert info["trajectory_replay_used"] is False
    assert info["start_gate_index"] == 0
    assert calls["set_state"] == 0
    assert len(calls["reset_positions"]) == 1


def test_trajectory_replay_records_and_samples_post_pass_state():
    cfg = build_flightmare_env_config(
        reset_mode="trajectory_replay_or_reference",
        start_gate_choices=[1],
        goal_mode="gate_i_to_i+1",
        trajectory_replay_capacity_per_gate=2,
        trajectory_replay_sample_prob=1.0,
        trajectory_replay_min_samples_per_gate=1,
        trajectory_replay_pos_noise_m=0.0,
        trajectory_replay_vel_noise_mps=0.0,
        trajectory_replay_omega_noise_radps=0.0,
        trajectory_replay_yaw_noise_rad=0.0,
    )
    env = object.__new__(FlightmareRacingEnv)
    env.cfg = cfg
    env.rng = np.random.default_rng(0)
    env._gates = [object(), object(), object(), object()]
    env._trajectory_replay = {}
    env._goal_terminal_gate_index = 2
    env._obs = SimpleNamespace(
        pos=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        vel=np.array([4.0, 5.0, 6.0], dtype=np.float64),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        omega=np.array([0.1, 0.2, 0.3], dtype=np.float64),
    )

    env._record_trajectory_replay_state(
        start_gate=1,
        action_norm=np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32),
        raw_action=np.array([0.3, -1.0, 2.0, -0.5], dtype=np.float32),
    )
    sample = env._sample_trajectory_replay_start_state()

    assert sample is not None
    start_gate, goal_terminal_gate, pos, vel, quat, omega, prev_action, prev_raw = sample
    assert start_gate == 1
    assert goal_terminal_gate == 2
    np.testing.assert_allclose(pos, env._obs.pos)
    np.testing.assert_allclose(vel, env._obs.vel)
    np.testing.assert_allclose(quat, env._obs.quat)
    np.testing.assert_allclose(omega, env._obs.omega)
    np.testing.assert_allclose(prev_action, np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32))
    np.testing.assert_allclose(prev_raw, np.array([0.3, -1.0, 2.0, -0.5], dtype=np.float32))


def test_trajectory_replay_preserves_remaining_goal_even_outside_reference_start_choices():
    cfg = build_flightmare_env_config(
        reset_mode="trajectory_replay_or_reference",
        start_gate_choices=[0, 1, 2, 3, 4, 5],
        goal_mode="gate_i_to_i+1",
        trajectory_replay_capacity_per_gate=2,
        trajectory_replay_sample_prob=1.0,
        trajectory_replay_min_samples_per_gate=1,
    )
    env = object.__new__(FlightmareRacingEnv)
    env.cfg = cfg
    env.rng = np.random.default_rng(0)
    env._gates = [object() for _ in range(7)]
    env._trajectory_replay = {}
    env._goal_terminal_gate_index = 7
    env._obs = SimpleNamespace(
        pos=np.zeros(3, dtype=np.float64),
        vel=np.ones(3, dtype=np.float64),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        omega=np.zeros(3, dtype=np.float64),
    )

    env._record_trajectory_replay_state(
        start_gate=6,
        action_norm=np.zeros(4, dtype=np.float32),
        raw_action=np.zeros(4, dtype=np.float32),
    )
    sample = env._sample_trajectory_replay_start_state()

    assert sample is not None
    start_gate, goal_terminal_gate, *_ = sample
    assert start_gate == 6
    assert goal_terminal_gate == 7


def test_restore_best_checkpoint_for_transition_does_not_rewind_training_state(tmp_path):
    model = torch.nn.Linear(1, 1)
    model.action_dim = 4
    trainer = FlightmarePPOTrainer(
        model=model,
        training_cfg={"output_dir": str(tmp_path), "save_optimizer_state": True},
        robotics_cfg={"ppo": {}, "architecture": {}},
        data_cfg={},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

    best_dir = tmp_path / "best"
    best_dir.mkdir()
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.bias.fill_(4.0)
    torch.save(model.state_dict(), best_dir / "model.pt")
    torch.save({"optimizer": optimizer.state_dict(), "iteration": 12}, best_dir / "optimizer.pt")
    with open(best_dir / "training_state.json", "w") as f:
        json.dump({"iteration": 12, "best_score": 9.0, "best_iteration": 12}, f)

    with torch.no_grad():
        model.weight.fill_(-1.0)
        model.bias.fill_(-2.0)
    trainer.start_iteration = 77
    trainer.best_score = 1.5
    trainer.best_iteration = 70

    assert trainer._restore_best_checkpoint(model, optimizer, restore_training_state=False)

    assert trainer.start_iteration == 77
    assert trainer.best_score == pytest.approx(1.5)
    assert trainer.best_iteration == 70
    assert model.weight.item() == pytest.approx(3.0)
    assert model.bias.item() == pytest.approx(4.0)


def test_teacher_v10_curriculum_starts_broad_and_tightens_gradually():
    cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_stage12_v10_ppo")
    ppo = cfg.robotics["ppo"]
    arch = cfg.robotics["architecture"]
    stages = ppo["curriculum"]["stages"]

    assert cfg.run["name"] == "paper_teacher_ctbr_stage12_v10_ppo"
    assert arch["init_log_std"] > -2.3
    assert arch["log_std_max"] > -1.6
    assert ppo["actor_lr"] <= 2.0e-6
    assert ppo["clip_eps"] <= 0.08

    assert stages[0]["goal_mode"] == "pass_gate_i"
    assert stages[0]["start_gate_choices"] == [0, 1, 2, 3, 4, 5, 6]
    assert stages[1]["goal_mode"] == "pass_gate_i_align_next"
    assert stages[1]["start_gate_choices"] == [0, 1, 2, 3, 4, 5]
    assert stages[1]["start_lateral_range_m"] == [-0.25, 0.25]

    align_stages = [stage for stage in stages if stage["goal_mode"] == "pass_gate_i_align_next"]
    center_limits = [float(stage["post_pass_max_center_error_norm"]) for stage in align_stages]
    survival_steps = [int(stage["post_pass_success_steps"]) for stage in align_stages]
    signed_limits = [float(stage["post_pass_min_gate_signed_distance_m"]) for stage in align_stages]

    assert center_limits == sorted(center_limits, reverse=True)
    assert survival_steps == sorted(survival_steps)
    assert signed_limits == sorted(signed_limits)
    assert center_limits[-1] >= 2.0
    assert survival_steps[-1] <= 45
    assert signed_limits[-1] <= -8.0

    chain_warmup = next(stage for stage in stages if stage["name"] == "s7_chain_critic_warmup_frozen")
    course_bridge = next(stage for stage in stages if stage["name"] == "s11_course_start_two_gate_bridge")
    assert chain_warmup["actor_lr_override"] < ppo["actor_lr"]
    assert course_bridge["reset_mode"] == "course_start"
    assert course_bridge["goal_mode"] == "gate_i_to_i+1"


def test_curriculum_stage_local_advance_requires_after_iter_and_patience():
    curriculum = Curriculum.from_config(
        {
            "enabled": True,
            "advance_metric": None,
            "stages": [
                {
                    "name": "first",
                    "until_iter": 100,
                    "advance_metric": "success_rate",
                    "advance_threshold": 0.8,
                    "advance_after_iter": 10,
                    "advance_patience": 2,
                    "goal_mode": "pass_gate_i",
                },
                {
                    "name": "second",
                    "until_iter": 200,
                    "goal_mode": "gate_i_to_i+1",
                },
            ],
        }
    )

    assert curriculum.active.name == "first"
    assert curriculum.env_overrides() == {"goal_mode": "pass_gate_i"}
    assert not curriculum.update(9, {"success_rate": 1.0})
    assert not curriculum.update(10, {"success_rate": 1.0})
    assert curriculum.active.name == "first"
    assert curriculum.update(11, {"success_rate": 1.0})
    assert curriculum.active.name == "second"
    assert curriculum.env_overrides() == {"goal_mode": "gate_i_to_i+1"}


def test_curriculum_can_hold_past_timeout_and_use_relative_duration():
    curriculum = Curriculum.from_config(
        {
            "enabled": True,
            "stages": [
                {
                    "name": "must_solve",
                    "until_iter": 5,
                    "advance_on_timeout": False,
                    "advance_metric": "success_rate",
                    "advance_threshold": 0.8,
                    "goal_mode": "gate_i_to_i+1",
                },
                {
                    "name": "critic_warmup",
                    "until_iter": 999,
                    "duration_iters": 3,
                    "actor_lr_override": 5.0e-8,
                    "goal_mode": "gate_i_to_i+2",
                },
                {
                    "name": "train_next",
                    "until_iter": 999,
                    "goal_mode": "gate_i_to_i+2",
                },
            ],
        }
    )

    assert not curriculum.update(6, {"success_rate": 0.0})
    assert curriculum.active.name == "must_solve"
    assert curriculum.update(7, {"success_rate": 1.0})
    assert curriculum.active.name == "critic_warmup"
    assert not curriculum.update(9, None)
    assert curriculum.active.name == "critic_warmup"
    assert curriculum.update(10, None)
    assert curriculum.active.name == "train_next"


def test_teacher_v11_is_simple_gate_count_curriculum_with_metric_gates():
    cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_stage12_v11_ppo")
    ppo = cfg.robotics["ppo"]
    arch = cfg.robotics["architecture"]
    stages = ppo["curriculum"]["stages"]

    assert cfg.run["name"] == "paper_teacher_ctbr_stage12_v11_ppo"
    assert arch["action_clip"] == pytest.approx(ppo["action_clip"])
    assert ppo["action_clip"] == pytest.approx(1.35)
    assert ppo["actor_lr"] <= 1.6e-6
    assert ppo["clip_eps"] <= 0.06
    assert ppo["target_kl"] <= 0.010

    assert [stage["goal_mode"] for stage in stages] == [
        "pass_gate_i",
        "gate_i_to_i+1",
        "gate_i_to_i+2",
        "finish_remaining_course",
        "finish_remaining_course",
    ]
    assert [len(stage["start_gate_choices"]) for stage in stages] == [7, 6, 5, 4, 1]
    assert all("pass_gate_i_align_next" != stage["goal_mode"] for stage in stages)
    assert all("post_pass_success_steps" not in stage for stage in stages)
    assert all(int(stage.get("advance_patience", 1)) >= 8 for stage in stages[:4])

    stage = diag_policy_stages._stage_overrides(cfg, "s1_one_gate_all")
    assert "advance_metric" not in stage
    assert "advance_after_iter" not in stage
    assert stage["goal_mode"] == "pass_gate_i"


def test_teacher_v12_warm_starts_best_checkpoint_and_uses_replay_held_curriculum():
    cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_stage23_v12_ppo")
    ppo = cfg.robotics["ppo"]
    stages = ppo["curriculum"]["stages"]

    assert cfg.run["name"] == "paper_teacher_ctbr_stage23_v12_ppo"
    assert ppo["resume_from"] is None
    assert ppo["reset_best_on_resume"] is False
    assert ppo["bc_checkpoint"] == "outputs/paper_teacher_ctbr_stage23_v11_ppo/best/model.pt"
    assert ppo["n_envs"] == 44
    assert ppo["n_workers"] == 11
    assert ppo["minibatch_size"] == 5632
    assert ppo["actor_lr"] <= 1.5e-6
    assert ppo["target_kl"] <= 0.008
    assert ppo["inverted_roll_jitter_rad"] == pytest.approx(0.0)
    assert ppo["trajectory_replay_capacity_per_gate"] >= 64
    assert ppo["trajectory_replay_sample_prob"] <= 0.20
    assert ppo["trajectory_replay_min_samples_per_gate"] >= 8

    assert [stage["name"] for stage in stages] == [
        "s2_two_gate_replay_finish",
        "s3_three_gate_replay_critic_warmup",
        "s3_three_gate_replay_train",
    ]
    assert stages[0]["reset_mode"] == "trajectory_replay_or_reference"
    assert stages[0]["advance_on_timeout"] is False
    assert stages[0]["advance_metric"] == "success_rate"
    assert stages[0]["advance_threshold"] >= 0.60
    assert stages[0]["trajectory_replay_sample_prob"] <= 0.20
    assert stages[0]["trajectory_replay_min_samples_per_gate"] >= 8
    assert stages[1]["duration_iters"] >= 100
    assert stages[1]["actor_lr_override"] < ppo["actor_lr"] * 0.1
    assert stages[2]["advance_on_timeout"] is False

    stage = diag_policy_stages._stage_overrides(cfg, "s2_two_gate_replay_finish")
    assert "advance_on_timeout" not in stage
    assert "duration_iters" not in stage
    assert stage["reset_mode"] == "trajectory_replay_or_reference"
    assert stage["trajectory_replay_sample_prob"] > 0.0


def test_teacher_v13_promotes_v12_best_with_course_start_anchor_and_best_restore():
    cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_stage23_v13_ppo")
    ppo = cfg.robotics["ppo"]
    arch = cfg.robotics["architecture"]
    stages = ppo["curriculum"]["stages"]

    assert cfg.run["name"] == "paper_teacher_ctbr_stage23_v13_ppo"
    assert arch["bc_checkpoint"] == "outputs/paper_teacher_ctbr_stage23_v12_ppo/best/model.pt"
    assert ppo["bc_checkpoint"] == "outputs/paper_teacher_ctbr_stage23_v12_ppo/best/model.pt"
    assert ppo["resume_from"] is None
    assert ppo["restore_best_on_stage_transition"] is True
    assert ppo["course_start_sample_prob"] == pytest.approx(0.12)
    assert ppo["actor_lr"] < 1.45e-6
    assert ppo["ent_coeff"] < 0.00105
    assert ppo["target_kl"] <= 0.007
    assert ppo["collapse_guard"]["enabled"] is True
    assert ppo["collapse_guard"]["skip_update"] is True

    assert [stage["name"] for stage in stages] == [
        "s2_two_gate_replay_finish",
        "s3_three_gate_replay_critic_warmup",
        "s3_three_gate_replay_train",
    ]
    assert stages[0]["advance_metric"] == "mean_goal_completion"
    assert stages[0]["advance_threshold"] == pytest.approx(0.68)
    assert stages[0]["advance_after_iter"] <= 300
    assert stages[0]["advance_patience"] <= 3
    assert stages[0]["course_start_sample_prob"] == pytest.approx(0.12)
    assert stages[1]["actor_lr_override"] < ppo["actor_lr"] * 0.1
    assert stages[1]["course_start_sample_prob"] > 0.0
    assert stages[2]["advance_metric"] == "mean_goal_completion"
    assert stages[2]["course_start_sample_prob"] == pytest.approx(0.12)

    stage = diag_policy_stages._stage_overrides(cfg, "s2_two_gate_replay_finish")
    assert "advance_metric" not in stage
    assert stage["course_start_sample_prob"] == pytest.approx(0.12)


def test_teacher_stage_diagnostics_preserve_training_env_semantics():
    cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_stage12_v9_ppo")
    base = diag_policy_stages._base_env_kwargs(cfg)
    stage = diag_policy_stages._stage_overrides(cfg, "s5_chain_critic_warmup_frozen")
    merged = diag_policy_stages._merge_stage_env_kwargs(base, stage)
    obs_kwargs = diag_obs_consistency._env_kwargs(cfg)

    assert base["plant"]["omega_max_body"] == [18.0, 18.0, 18.0]
    assert base["action_clip"] == pytest.approx(1.5)
    assert "actor_lr_override" not in stage
    assert merged["goal_mode"] == "gate_i_to_i+1"
    assert merged["reward_kwargs"]["speed_limit_mps"] == pytest.approx(7.5)
    assert merged["reward_kwargs"]["centerline_progress_scale"] == pytest.approx(4.8)
    assert obs_kwargs["plant"]["omega_max_body"] == [18.0, 18.0, 18.0]
    assert obs_kwargs["gate_approach_m"] == pytest.approx(3.0)


def test_trainer_curriculum_merge_keeps_base_reward_and_plant_overrides():
    cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_stage12_v9_ppo")
    model = torch.nn.Linear(1, 1)
    model.action_dim = 4
    trainer = FlightmarePPOTrainer(
        model=model,
        training_cfg=cfg.training,
        robotics_cfg=cfg.robotics,
        data_cfg=cfg.data,
    )
    trainer.curriculum.update(2000, None)

    merged = trainer._apply_curriculum_overrides(trainer._env_kwargs())

    assert merged["goal_mode"] == "gate_i_to_i+1"
    assert merged["plant"]["omega_max_body"] == [18.0, 18.0, 18.0]
    assert merged["reward_kwargs"]["speed_limit_mps"] == pytest.approx(7.5)
    assert merged["reward_kwargs"]["centerline_error_penalty"] == pytest.approx(0.07)
    assert trainer.ent_coeff == pytest.approx(0.0010)


def test_trainer_curriculum_reapplies_configured_optimizer_lrs_after_resume():
    cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_stage23_v12_ppo")
    model = torch.nn.Linear(1, 1)
    model.action_dim = 4
    trainer = FlightmarePPOTrainer(
        model=model,
        training_cfg=cfg.training,
        robotics_cfg=cfg.robotics,
        data_cfg=cfg.data,
    )
    trainer.curriculum = Curriculum.from_config(
        {
            "enabled": True,
            "stages": [
                {
                    "name": "critic_warmup",
                    "until_iter": 1,
                    "actor_lr_override": 5.0e-8,
                },
                {"name": "train", "until_iter": 100},
            ],
        }
    )
    trainer._optimizer = torch.optim.AdamW([
        {"params": [model.weight], "lr": 9.0e-4},
        {"params": [model.bias], "lr": 8.0e-4},
    ])

    trainer._apply_curriculum_overrides(trainer._env_kwargs())

    assert trainer._optimizer.param_groups[0]["lr"] == pytest.approx(5.0e-8)
    assert trainer._optimizer.param_groups[1]["lr"] == pytest.approx(trainer.critic_lr)

    trainer._optimizer.param_groups[0]["lr"] = 9.0e-4
    trainer._optimizer.param_groups[1]["lr"] = 8.0e-4
    assert trainer.curriculum.update(2, None)
    trainer._apply_curriculum_overrides(trainer._env_kwargs())

    assert trainer._optimizer.param_groups[0]["lr"] == pytest.approx(trainer.actor_lr)
    assert trainer._optimizer.param_groups[1]["lr"] == pytest.approx(trainer.critic_lr)


def test_teacher_config_exposes_racing_plant_authority_and_speed_penalty():
    old_cfg = Config.from_experiment("flightmare/paper/ctbr_stage12_ppo")
    teacher_cfg = Config.from_experiment("flightmare/paper/teacher/ctbr_stage12_v9_ppo")
    old_ppo = old_cfg.robotics["ppo"]
    teacher_arch = teacher_cfg.robotics["architecture"]
    teacher_ppo = teacher_cfg.robotics["ppo"]

    assert old_ppo.get("plant") is None
    assert teacher_ppo["plant"]["omega_max_body"] == [18.0, 18.0, 18.0]
    assert teacher_arch["action_clip"] == pytest.approx(teacher_ppo["action_clip"])
    assert teacher_ppo["action_clip"] > 1.0
    assert teacher_ppo["reward_kwargs"]["speed_limit_mps"] == pytest.approx(7.5)
    assert teacher_ppo["reward_kwargs"]["speed_limit_penalty"] > 0.0

    reward = dict(teacher_ppo["reward_kwargs"])
    for stage in teacher_ppo["curriculum"]["stages"]:
        effective_reward = {**reward, **(stage.get("reward_kwargs") or {})}
        if float(stage.get("reference_speed_mps", teacher_ppo["reference_speed_mps"])) <= 3.0:
            assert effective_reward["speed_limit_mps"] <= 7.5
            assert effective_reward["speed_limit_penalty"] > 0.0

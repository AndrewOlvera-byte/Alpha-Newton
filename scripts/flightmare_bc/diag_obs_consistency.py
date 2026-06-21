"""Check PPO v3 observations against direct raw-state recomputation.

Example:
  python -m scripts.flightmare_bc.diag_obs_consistency \
      --exp flightmare/paper/ctbr_curriculum_ppo --episodes 3 --steps 200
"""
from __future__ import annotations

import argparse

import numpy as np

from scripts.flightmare_bc.obs_v3 import LOOKAHEAD_GATES_V3, build_proprio_core, encode_aux, encode_gate_corners
from src.core.config import Config
from src.robotics.flightmare_envs import build_flightmare_env_config, FlightmareRacingEnv


def _env_kwargs(cfg: Config) -> dict:
    ppo = (cfg.robotics or {}).get("ppo", {}) or {}
    data = cfg.data or {}
    keys = {
        "control_hz", "horizon", "scene", "render", "backend", "course_mode",
        "gate_layout", "random_start_gate", "fixed_gate_pos_noise",
        "fixed_gate_pos_noise_xyz",
        "fixed_gate_yaw_noise", "num_gates", "gate_spacing_range",
        "gate_lateral_jitter", "gate_z_range", "gate_yaw_step",
        "gate_yaw_noise", "gate_size", "gate_approach_m", "inverted_roll_jitter_rad",
        "z_min", "gate_vehicle_radius",
        "max_world_radius", "min_z", "terminate_on_gate_miss",
        "terminate_on_crash", "reward_fn", "reward_kwargs", "max_body_rate",
        "max_waypoint_speed", "max_collective_thrust_g", "action_clip",
        "plant", "action_bounds_override",
        "reset_mode", "course_start_sample_prob", "start_gate_index", "start_gate_choices",
        "start_offset_m", "start_offset_range",
        "start_lateral_range_m", "start_vertical_range_m",
        "start_yaw_noise_rad", "start_speed_noise_mps",
        "reference_speed_mps",
        "goal_mode", "goal_gate_span", "post_pass_success_steps",
        "post_pass_max_speed_mps", "post_pass_max_body_rate",
        "post_pass_max_center_error_norm", "post_pass_min_gate_signed_distance_m",
        "trajectory_replay_capacity_per_gate", "trajectory_replay_sample_prob",
        "trajectory_replay_min_samples_per_gate", "trajectory_replay_pos_noise_m",
        "trajectory_replay_vel_noise_mps", "trajectory_replay_omega_noise_radps",
        "trajectory_replay_yaw_noise_rad",
    }
    out = {k: v for k, v in ppo.items() if k in keys}
    out.update(
        data_dir=data.get("data_dir", "data/flightmare/bc_v4"),
        action_type=data.get("action_type", ppo.get("action_type", "ctbr")),
        obs_schema=data.get("obs_schema", ppo.get("obs_schema", "v3")),
        include_mission=data.get("include_mission", True),
        normalize_state=data.get("normalize_state", True),
        normalize_mission=data.get("normalize_mission", True),
        normalize_action=data.get("normalize_action", True),
        action_normalization=data.get("action_normalization", "auto"),
        seed=int(ppo.get("seed", 0)),
    )
    return out


def _direct_normalized(env: FlightmareRacingEnv) -> np.ndarray:
    current = 0 if env._strict_tracker is None else env._strict_tracker.current_index
    time_since_pass = float(env._step_count - env._last_pass_step) * env.dt
    proprio = build_proprio_core(env._obs.vel, env._obs.quat, env._obs.omega)
    gate = encode_gate_corners(env._obs.pos, env._obs.quat, env._gate_specs, current, LOOKAHEAD_GATES_V3)
    aux = encode_aux(env._obs.pos, env._gate_specs, current, time_since_pass)
    return env.norm.normalize_observation_v3(proprio, gate, aux)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", type=str, required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()

    cfg = Config.from_experiment(args.exp)
    kwargs = _env_kwargs(cfg)
    kwargs["obs_schema"] = "v3"
    kwargs["render"] = False

    env = FlightmareRacingEnv(build_flightmare_env_config(**kwargs))
    max_err = 0.0
    checked = 0
    try:
        for ep in range(int(args.episodes)):
            obs, _ = env.reset(seed=int(kwargs.get("seed", 0)) + ep)
            for _ in range(int(args.steps)):
                direct = _direct_normalized(env)
                err = float(np.max(np.abs(obs["state"] - direct)))
                max_err = max(max_err, err)
                if err > args.atol:
                    raise SystemExit(f"obs mismatch at episode={ep} step={env._step_count}: {err}")
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                obs, _, term, trunc, _ = env.step(action)
                checked += 1
                if term or trunc:
                    break
    finally:
        env.close()

    print(f"[diag-obs] exp={args.exp} samples={checked} max_abs_err={max_err:.6g}")


if __name__ == "__main__":
    main()

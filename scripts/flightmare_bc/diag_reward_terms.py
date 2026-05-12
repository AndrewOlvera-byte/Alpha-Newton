"""Roll out a Flightmare PPO env briefly and summarize reward term logging."""
from __future__ import annotations

import argparse

import numpy as np

from scripts.flightmare_bc.diag_obs_consistency import _env_kwargs
from src.core.config import Config
from src.robotics.flightmare_envs import FlightmareRacingEnv, build_flightmare_env_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp", type=str, required=True)
    p.add_argument("--stage", type=str, default=None, help="Optional label printed with the summary.")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()

    cfg = Config.from_experiment(args.exp)
    kwargs = _env_kwargs(cfg)
    if args.stage:
        stages = ((cfg.robotics or {}).get("ppo", {}) or {}).get("curriculum", {}).get("stages", []) or []
        for i, stage in enumerate(stages):
            if args.stage in {str(stage.get("name", "")), str(i), f"stage_{i}"}:
                for key, value in stage.items():
                    if key not in {"name", "until_iter", "ent_coeff_override"}:
                        kwargs[key] = value
                break
    kwargs["reward_fn"] = kwargs.get("reward_fn", "flightmare_racing_v2")
    env = FlightmareRacingEnv(build_flightmare_env_config(**kwargs))
    sums: dict[str, float] = {}
    count = 0
    try:
        for ep in range(int(args.episodes)):
            env.reset(seed=int(kwargs.get("seed", 0)) + ep)
            for _ in range(int(args.steps)):
                action = np.zeros(env.action_space.shape, dtype=np.float32)
                _, _, term, trunc, info = env.step(action)
                terms = info.get("reward_terms", {})
                if terms:
                    count += 1
                    for k, v in terms.items():
                        sums[k] = sums.get(k, 0.0) + float(v)
                if term or trunc:
                    break
    finally:
        env.close()

    print(f"[diag-reward] exp={args.exp} stage={args.stage or '-'} samples={count}")
    for key in sorted(sums):
        print(f"  {key:24s}: {sums[key] / max(1, count): .6f}")
    if not sums:
        raise SystemExit("no reward_terms found; use reward_fn=flightmare_racing_v2")


if __name__ == "__main__":
    main()

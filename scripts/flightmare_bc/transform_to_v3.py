"""Overlay obs v3 onto an existing bc_v4 dataset (in-place).

Reads each ``episodes/ep_*.h5`` file, joins it with the per-episode gate list
in the dataset's ``index.json``, and writes three new datasets next to the
existing ``obs/state``:

    obs/proprio_core   (T, 9)   float32
    obs/gate           (T, 24)  float32
    obs/aux            (T, 3)   float32

Augments ``index.json`` with an ``obs_v3`` block describing the layout, and
adds ``proprio_core_{mean,std}``, ``gate_{mean,std}``, ``aux_{mean,std}``
into ``norm_stats.npz`` (preserving the existing v2 keys).

Idempotent: if the v3 keys already exist on every episode and
``obs_v3`` is present in ``index.json`` the script is a no-op unless run with
``--force``.

Usage:
  python -m scripts.flightmare_bc.transform_to_v3 \
      --data-dir data/flightmare/bc_v4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.flightmare_bc.obs_v3 import (
    AUX_DIM,
    GATE_DIM,
    LOOKAHEAD_GATES_V3,
    PROPRIO_CORE_DIM,
    GateSpec,
    build_proprio_core,
    encode_aux,
    encode_gate_corners,
)


_V3_KEYS = ("obs/proprio_core", "obs/gate", "obs/aux")


def _gates_from_index_entry(entry: dict) -> list[GateSpec]:
    """Coerce the index.json gate dicts into ``GateSpec`` objects."""
    out: list[GateSpec] = []
    for g in entry["gates"]:
        size = g.get("size", [1.6, 1.6, 1.6])
        out.append(
            GateSpec(
                pos=np.asarray(g["pos"], dtype=np.float64),
                yaw=float(g.get("yaw", 0.0)),
                size=np.asarray(size, dtype=np.float64),
                quat=(np.asarray(g["quat"], dtype=np.float64) if g.get("quat") is not None else None),
            )
        )
    return out


def _episode_has_v3(f: h5py.File) -> bool:
    return all(k in f for k in _V3_KEYS)


def _delete_existing_v3(f: h5py.File) -> None:
    for k in _V3_KEYS:
        if k in f:
            del f[k]


def _time_since_pass(gate_index: np.ndarray, dt: float) -> np.ndarray:
    """Seconds since the last gate-index transition. Resets to 0 at each
    advance. Returns 0 for all steps before the first transition."""
    T = gate_index.shape[0]
    out = np.zeros(T, dtype=np.float32)
    last = 0
    seen_any = False
    for t in range(T):
        if t > 0 and gate_index[t] != gate_index[t - 1]:
            last = t
            seen_any = True
        out[t] = (t - last) * dt if seen_any else 0.0
    return out


def transform_episode(
    h5_path: Path,
    gates: list[GateSpec],
    dt: float,
    force: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add v3 keys to a single episode file. Returns (proprio, gate, aux)."""
    with h5py.File(h5_path, "r+") as f:
        if _episode_has_v3(f) and not force:
            return (
                f["obs/proprio_core"][...],
                f["obs/gate"][...],
                f["obs/aux"][...],
            )

        state = f["obs/state"][...]                    # (T, 13)
        gate_index = f["mission/gate_index"][...]      # (T,)
        T = state.shape[0]

        pos = state[:, 0:3]
        vel = state[:, 3:6]
        quat = state[:, 6:10]
        omega = state[:, 10:13]

        proprio = np.empty((T, PROPRIO_CORE_DIM), dtype=np.float32)
        gate = np.empty((T, GATE_DIM), dtype=np.float32)
        aux = np.empty((T, AUX_DIM), dtype=np.float32)

        tsp = _time_since_pass(gate_index, dt)

        for t in range(T):
            proprio[t] = build_proprio_core(vel[t], quat[t], omega[t])
            idx = int(gate_index[t])
            gate[t] = encode_gate_corners(pos[t], quat[t], gates, idx, LOOKAHEAD_GATES_V3)
            aux[t] = encode_aux(pos[t], gates, idx, float(tsp[t]))

        if force:
            _delete_existing_v3(f)
        f.create_dataset("obs/proprio_core", data=proprio, compression="gzip", compression_opts=4)
        f.create_dataset("obs/gate", data=gate, compression="gzip", compression_opts=4)
        f.create_dataset("obs/aux", data=aux, compression="gzip", compression_opts=4)
    return proprio, gate, aux


def update_index_json(index_path: Path) -> None:
    idx = json.loads(index_path.read_text())
    idx["obs_v3"] = {
        "lookahead_gates": LOOKAHEAD_GATES_V3,
        "proprio_core": {
            "dim": PROPRIO_CORE_DIM,
            "layout": ["v_body(3)", "omega_body(3)", "gravity_body(3)"],
        },
        "gate": {
            "dim": GATE_DIM,
            "layout": (
                f"{LOOKAHEAD_GATES_V3} gates x 4 corners (TL,TR,BR,BL) x 3 (body-frame xyz)"
            ),
        },
        "aux": {
            "dim": AUX_DIM,
            "layout": ["progress", "dist_to_current", "time_since_pass"],
        },
        "total_state_dim_no_prev_action": PROPRIO_CORE_DIM + GATE_DIM + AUX_DIM,
    }
    index_path.write_text(json.dumps(idx, indent=2))


def update_norm_stats(
    stats_path: Path, proprio: np.ndarray, gate: np.ndarray, aux: np.ndarray
) -> None:
    """Merge v3 mean/std into the existing norm_stats.npz."""
    existing = {}
    if stats_path.exists():
        with np.load(stats_path) as s:
            existing = {k: s[k] for k in s.files}
    eps = 1e-6
    existing["proprio_core_mean"] = proprio.mean(axis=0).astype(np.float32)
    existing["proprio_core_std"] = (proprio.std(axis=0) + eps).astype(np.float32)
    existing["gate_mean"] = gate.mean(axis=0).astype(np.float32)
    existing["gate_std"] = (gate.std(axis=0) + eps).astype(np.float32)
    existing["aux_mean"] = aux.mean(axis=0).astype(np.float32)
    existing["aux_std"] = (aux.std(axis=0) + eps).astype(np.float32)
    np.savez(stats_path, **existing)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data/flightmare/bc_v4", type=str)
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing v3 keys.")
    ap.add_argument("--max-episodes", type=int, default=None,
                    help="Limit episodes processed (for smoke runs).")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    index_path = data_dir / "index.json"
    stats_path = data_dir / "norm_stats.npz"

    idx = json.loads(index_path.read_text())
    dt = float(idx.get("control_hz") and 1.0 / float(idx["control_hz"]) or 0.01)

    episodes = idx["episodes"]
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]

    print(f"[v3-overlay] {len(episodes)} episodes, dt={dt}s, force={args.force}")

    proprio_chunks: list[np.ndarray] = []
    gate_chunks: list[np.ndarray] = []
    aux_chunks: list[np.ndarray] = []

    for ep in episodes:
        ep_id = ep["episode_id"]
        h5_path = data_dir / "episodes" / f"ep_{ep_id:06d}.h5"
        if not h5_path.exists():
            print(f"  [skip] missing {h5_path.name}")
            continue
        gates = _gates_from_index_entry(ep)
        p, g, a = transform_episode(h5_path, gates, dt, args.force)
        proprio_chunks.append(p)
        gate_chunks.append(g)
        aux_chunks.append(a)
        if (ep_id + 1) % 100 == 0:
            print(f"  [{ep_id + 1}/{len(episodes)}] last shape: "
                  f"proprio={p.shape}, gate={g.shape}, aux={a.shape}")

    if not proprio_chunks:
        print("[v3-overlay] no episodes processed; abort")
        return

    proprio_all = np.concatenate(proprio_chunks, axis=0)
    gate_all = np.concatenate(gate_chunks, axis=0)
    aux_all = np.concatenate(aux_chunks, axis=0)

    print(
        f"[v3-overlay] aggregated stats: "
        f"proprio={proprio_all.shape}, gate={gate_all.shape}, aux={aux_all.shape}"
    )

    update_norm_stats(stats_path, proprio_all, gate_all, aux_all)
    print(f"[v3-overlay] norm_stats.npz updated -> {stats_path}")

    update_index_json(index_path)
    print(f"[v3-overlay] index.json augmented with obs_v3 block -> {index_path}")


if __name__ == "__main__":
    main()

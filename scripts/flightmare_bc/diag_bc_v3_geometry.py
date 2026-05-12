"""Diagnose BC v3 gate-corner geometry and gate quaternion metadata.

Example:
  python -m scripts.flightmare_bc.diag_bc_v3_geometry \
      --data-dir data/flightmare/bc_v4 --episodes 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.flightmare_bc.obs_v3 import LOOKAHEAD_GATES_V3, GateSpec, encode_gate_corners


def _gates(ep: dict, *, yaw_only: bool = False) -> list[GateSpec]:
    out: list[GateSpec] = []
    for g in ep.get("gates", []):
        quat = None if yaw_only else g.get("quat")
        out.append(
            GateSpec(
                pos=np.asarray(g["pos"], dtype=np.float64),
                yaw=float(g.get("yaw", 0.0)),
                size=np.asarray(g.get("size", [1.6, 1.6, 1.6]), dtype=np.float64),
                quat=None if quat is None else np.asarray(quat, dtype=np.float64),
            )
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/flightmare/bc_v4"))
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--fail-on-missing-quat", action="store_true")
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()

    index_path = args.data_dir / "index.json"
    idx = json.loads(index_path.read_text())
    episodes = idx.get("episodes", [])[: max(0, int(args.episodes))]
    if not episodes:
        raise SystemExit(f"No episodes found in {index_path}")

    missing_quat = 0
    quat_diff_eps = 0
    stored_mismatch = 0
    checked = 0
    max_stored_err = 0.0
    max_yaw_quat_diff = 0.0

    for ep in episodes:
        gates = _gates(ep, yaw_only=False)
        yaw_gates = _gates(ep, yaw_only=True)
        missing_quat += sum(1 for g in ep.get("gates", []) if g.get("quat") is None)
        h5_path = args.data_dir / ep["path"]
        with h5py.File(h5_path, "r") as h:
            state = h["obs/state"][...]
            gate_index = h["mission/gate_index"][...]
            stored_gate = h["obs/gate"][...] if "obs/gate" in h else None
            sample_ts = np.linspace(0, state.shape[0] - 1, min(16, state.shape[0])).round().astype(int)
            for t in sample_ts:
                pos = state[t, 0:3]
                quat = state[t, 6:10]
                gi = int(gate_index[t])
                encoded = encode_gate_corners(pos, quat, gates, gi, LOOKAHEAD_GATES_V3)
                yaw_encoded = encode_gate_corners(pos, quat, yaw_gates, gi, LOOKAHEAD_GATES_V3)
                d = float(np.max(np.abs(encoded - yaw_encoded)))
                max_yaw_quat_diff = max(max_yaw_quat_diff, d)
                if d > args.atol:
                    quat_diff_eps += 1
                if stored_gate is not None:
                    err = float(np.max(np.abs(encoded - stored_gate[t])))
                    max_stored_err = max(max_stored_err, err)
                    if err > args.atol:
                        stored_mismatch += 1
                checked += 1

    print(f"[diag-bc-v3] data_dir={args.data_dir}")
    print(f"  episodes_checked        : {len(episodes)}")
    print(f"  samples_checked         : {checked}")
    print(f"  gates_missing_quat      : {missing_quat}")
    print(f"  yaw_vs_quat_diff_samples: {quat_diff_eps}")
    print(f"  max_yaw_vs_quat_diff    : {max_yaw_quat_diff:.6g}")
    print(f"  stored_v3_mismatches    : {stored_mismatch}")
    print(f"  max_stored_v3_error     : {max_stored_err:.6g}")

    if args.fail_on_missing_quat and missing_quat:
        raise SystemExit("missing gate quat metadata")
    if stored_mismatch:
        raise SystemExit("stored obs/gate does not match quat-aware v3 encoder")


if __name__ == "__main__":
    main()

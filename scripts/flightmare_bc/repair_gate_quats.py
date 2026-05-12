"""Patch existing Flightmare BC manifests with gate quaternion metadata.

This is a repair path for datasets collected before ``index.json`` persisted
``gate["quat"]``. For ``swift_v4`` the original per-episode inverted-roll
jitter cannot be recovered from the old manifest, so the inverted gate is
patched with canonical 180-degree roll about the gate forward axis.

After running this, regenerate v3 overlays with:
  python -m scripts.flightmare_bc.transform_to_v3 --data-dir data/flightmare/bc_v4 --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.flightmare_bc.collect import _quat_from_yaw_pitch_roll


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data/flightmare/bc_v4"))
    p.add_argument("--inverted-gate-index", type=int, default=2)
    p.add_argument("--force", action="store_true", help="Overwrite existing gate quats too.")
    args = p.parse_args()

    index_path = args.data_dir / "index.json"
    manifest = json.loads(index_path.read_text())
    course_mode = str(manifest.get("course_mode", ""))
    patched = 0
    already = 0

    for ep in manifest.get("episodes", []):
        for i, gate in enumerate(ep.get("gates", [])):
            if gate.get("quat") is not None and not args.force:
                already += 1
                continue
            yaw = float(gate.get("yaw", 0.0))
            roll = 0.0
            if course_mode == "swift_v4" and i == int(args.inverted_gate_index):
                roll = 3.141592653589793
            gate["quat"] = _quat_from_yaw_pitch_roll(yaw, 0.0, roll).astype(float).tolist()
            patched += 1

    manifest["gate_quat_repair"] = {
        "method": "yaw_quat_plus_canonical_inverted_roll",
        "course_mode": course_mode,
        "inverted_gate_index": int(args.inverted_gate_index),
        "note": "Original inverted-roll jitter cannot be recovered if the old manifest omitted quat.",
    }
    index_path.write_text(json.dumps(manifest, indent=2))
    print(f"[repair-gate-quats] patched={patched} already_present={already} -> {index_path}")


if __name__ == "__main__":
    main()

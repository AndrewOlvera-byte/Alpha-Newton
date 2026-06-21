"""Dump swift_v4 gate geometry and the turn angle the drone must make leaving
each start gate, so the curriculum can order start_gate_choices easy->sharp.

For start gate i in gate_i_to_i+1: the drone is reset aligned through gate i
(heading = gate i forward). The hard part is turning from gate i's forward onto
the line into gate i+1. We report:
  - heading_turn_deg: angle between gate i forward and the (i -> i+1) direction
  - gate_axis_turn_deg: angle between gate i forward and gate i+1 forward
  - seg_len: distance i -> i+1 (shorter segment = less room to make the turn)
"""
import numpy as np
from src.robotics.flightmare_autonomy_fsw.nodes import CourseConfig
from src.robotics.flightmare_autonomy_fsw.gates import gate_frame
from scripts.flightmare_bc.collect import sample_gate_course


def ang(u, v):
    u = u / (np.linalg.norm(u) + 1e-9)
    v = v / (np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))))


def main():
    cfg = CourseConfig(course_mode="swift_v4", num_gates=7, gate_size=1.6,
                       gate_approach_m=3.0, z_min=1.0)
    # Fixed seed; swift_v4 with zero noise should be deterministic anyway.
    rng = np.random.default_rng(0)
    gates = sample_gate_course(rng, cfg)
    n = len(gates)
    frames = [gate_frame(g) for g in gates]  # (center, forward, right, up)
    print(f"num_gates={n}")
    rows = []
    for i in range(n - 1):
        c_i, f_i, _, _ = frames[i]
        c_j, f_j, _, _ = frames[i + 1]
        to_next = np.asarray(c_j) - np.asarray(c_i)
        heading_turn = ang(f_i, to_next)
        axis_turn = ang(f_i, f_j)
        seg_len = float(np.linalg.norm(to_next))
        rows.append((i, heading_turn, axis_turn, seg_len))
        print(f"  start_gate {i} -> {i+1}: heading_turn={heading_turn:6.1f}deg "
              f"axis_turn={axis_turn:6.1f}deg seg_len={seg_len:5.1f}m  "
              f"center_i={np.round(c_i,1)}")
    order = [r[0] for r in sorted(rows, key=lambda r: r[1])]
    print("\nstart_gate_choices easy->hard by heading_turn:", order)
    print("heading_turn sorted:", [round(r[1], 1) for r in sorted(rows, key=lambda r: r[1])])


if __name__ == "__main__":
    main()

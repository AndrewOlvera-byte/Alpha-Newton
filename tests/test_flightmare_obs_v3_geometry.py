import json
from types import SimpleNamespace

import h5py
import numpy as np

from scripts.flightmare_bc.collect import _gate_record, sample_gate_course
from scripts.flightmare_bc.obs_v3 import GateSpec, encode_gate_corners
from scripts.flightmare_bc.transform_to_v3 import transform_episode


def test_swift_v4_gate_records_persist_quaternions():
    cfg = SimpleNamespace(
        course_mode="swift_v4",
        num_gates=7,
        gate_size=1.6,
        z_min=1.0,
        fixed_gate_pos_noise=0.0,
        fixed_gate_yaw_noise=0.0,
        inverted_gate_index=2,
        inverted_roll_jitter_rad=0.0,
    )
    gates = sample_gate_course(np.random.default_rng(0), cfg)
    records = [_gate_record(g) for g in gates]

    assert all(r["quat"] is not None for r in records)
    # JSON round-trip matters because index.json is the source used by v3 overlays.
    round_tripped = json.loads(json.dumps(records))
    assert round_tripped[2]["quat"] is not None

    inverted = GateSpec(
        pos=np.asarray(round_tripped[2]["pos"], dtype=np.float64),
        yaw=float(round_tripped[2]["yaw"]),
        size=np.asarray(round_tripped[2]["size"], dtype=np.float64),
        quat=np.asarray(round_tripped[2]["quat"], dtype=np.float64),
    )
    yaw_only = GateSpec(pos=inverted.pos, yaw=inverted.yaw, size=inverted.size, quat=None)
    pos = inverted.pos - np.array([3.0, 0.0, 0.0], dtype=np.float64)
    drone_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    quat_corners = encode_gate_corners(pos, drone_quat, [inverted], 0)
    yaw_corners = encode_gate_corners(pos, drone_quat, [yaw_only], 0)
    assert np.max(np.abs(quat_corners - yaw_corners)) > 0.5


def test_transform_to_v3_uses_same_quat_aware_encoder(tmp_path):
    gate = GateSpec(
        pos=np.array([3.0, 0.0, 2.0], dtype=np.float64),
        yaw=0.0,
        size=np.array([1.6, 1.6, 1.6], dtype=np.float64),
        quat=np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64),
    )
    h5_path = tmp_path / "ep_000000.h5"
    state = np.array(
        [[0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    with h5py.File(h5_path, "w") as h:
        h.create_dataset("obs/state", data=state)
        h.create_dataset("mission/gate_index", data=np.array([0], dtype=np.int32))

    _, gate_block, _ = transform_episode(h5_path, [gate], dt=0.01, force=False)
    expected = encode_gate_corners(state[0, 0:3], state[0, 6:10], [gate], 0)
    np.testing.assert_allclose(gate_block[0], expected, atol=1e-6)

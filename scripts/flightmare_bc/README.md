# Flightmare BC dataset collection

Collects a privileged-expert imitation-learning dataset in Flightmare's
quadrotor dynamics environment. Each step records proprioceptive state, RGB
camera frames when rendering is enabled, and all three action labels
(waypoint+speed, CTBR, and motor commands) so the same dataset can train the
action-output ablations of
`MLPFusionGaussianExpertActor`.

This repo patches Flightmare's upstream `flightgym` wheel during the Docker
build. The stock binding still exists as `flightgym.QuadrotorEnv_v1` for fast
state-only motor rollouts. The patch also adds
`flightgym.VisualRacingEnv_v0`, which exposes direct state reset, gate
insertion, CTBR/motor stepping, Unity rendering, and RGB frame readback. Use
`--no-render` for cheap dynamics/debug runs; use `--launch-unity` without
`--no-render` when you want real visual BC frames.

## Why this expert (which github did we pull from?)

Real drone-racing BC datasets are produced by one of three classical experts:

| Expert | Repo | Trade-off |
|---|---|---|
| **NMPC tracker** (Foehn, Romero, Scaramuzza) | [`uzh-rpg/rpg_mpc`](https://github.com/uzh-rpg/rpg_mpc) | Highest fidelity; ROS / acados toolchain, heavy to install |
| **Privileged sampler + expert** | [`uzh-rpg/agile_autonomy`](https://github.com/uzh-rpg/agile_autonomy) | Used in *Learning High-Speed Flight in the Wild* (Sci. Rob. 2021); ROS-coupled |
| **Min-jerk polynomial + SE(3) geometric controller** | [`spencerfolk/rotorpy`](https://github.com/spencerfolk/rotorpy), [`utiasDSL/gym-pybullet-drones`](https://github.com/utiasDSL/gym-pybullet-drones) | Pure Python, ~150 LOC, well-tested; standard reference (Lee, Leok, McClamroch 2010) |

For BC warmup we want high throughput, no ROS, and trajectories that span the
relevant flight envelope. The **min-jerk + SE(3) geometric** stack wins on all
three counts. It is the same controller family used inside `rotorpy` and
`gym-pybullet-drones` and is also the inner-loop tracker for several
agile_autonomy variants. We adapt it (no external code dependency required) in
`controllers.py` and `trajectories.py`.

References:
* T. Lee, M. Leok, N. H. McClamroch. *Geometric Tracking Control of a Quadrotor UAV on SE(3)*. CDC 2010.
* D. Mellinger, V. Kumar. *Minimum Snap Trajectory Generation and Control for Quadrotors*. ICRA 2011.

## Pipeline

`collect.py` runs `N_episodes` episodes:

1. Sample either a random waypoint path or a randomized gate course
   (`--course-mode random|gates`).
2. Fit a **minimum-jerk** polynomial through them at a target average speed
   (`--speed-range`).
3. At each control tick (default 100 Hz) the **SE(3) geometric controller**
   converts the current reference state into a CTBR command
   `(thrust, ω_x, ω_y, ω_z)`.
4. The command is stepped through Flightmare as CTBR on the visual backend, or
   mixed into per-rotor motor thrusts for the stock state-only backend.
5. Per step we record the **observation** plus all action labels:
   * **waypoint+speed** = the next look-ahead waypoint expressed in the body
     frame, plus the trajectory's instantaneous tangential speed
     (4-dim: Δx_b, Δy_b, Δz_b, v).
   * **CTBR** = the controller's `(T_norm, ω_x, ω_y, ω_z)` command (4-dim).
   * **motor** = the per-rotor normalized thrusts derived from CTBR via the
     mixer (4-dim, optional alternative low-level head).

Each episode is saved as a single HDF5 file (`hdf5_writer.py`) with chunking
chosen for fast random per-step access from a multi-worker PyTorch DataLoader.

## Quick start

```bash
# (inside docker / venv with flightmare's flightgym + h5py installed)
python -m scripts.flightmare_bc.collect \
    --out data/flightmare/expert_v1 \
    --episodes 500 \
    --image-size 224 \
    --control-hz 100 \
    --cameras forward \
    --no-render \
    --seed 0
```

Rendered gate-course collection:

```bash
python -m scripts.flightmare_bc.collect \
    --out data/flightmare/gates_visual_v1 \
    --episodes 500 \
    --course-mode gates \
    --num-gates 8 \
    --scene warehouse \
    --image-size 224 \
    --control-hz 100 \
    --launch-unity \
    --seed 0
```

Quick video inspection:

```bash
python -m scripts.flightmare_bc.export_video \
    --data-dir data/flightmare/gates_visual_v1 \
    --episode-index 0 \
    --out data/flightmare/gates_visual_v1/ep0.mp4 \
    --fps 30
```

Useful diversity knobs:

* `--scene industrial|warehouse|garage|natureforest|tunnels`: selects the
  Flightmare Unity scene when rendering is enabled.
* `--bbox X Y Z`: samples waypoint offsets around the actual reset position.
* `--speed-range MIN MAX`: varies trajectory aggressiveness.
* `--n-waypoints`: changes course length and curvature.
* `--lookahead-s`: changes the high-level waypoint target horizon.
* `--num-gates`, `--gate-spacing-range`, `--gate-lateral-jitter`,
  `--gate-z-range`, `--gate-yaw-step`, `--gate-yaw-noise`, `--gate-size`:
  control randomized gate-course geometry.
* `--seed`: controls deterministic episode/course generation.

## Gate waypoint policy

For sim-to-real racing, the most reusable hierarchy is:

1. Perception estimates gates or near-term free-space structure from images.
2. A high-level policy/planner turns that into body-frame waypoint+speed
   commands.
3. A tracker converts waypoints to CTBR or motor commands.

That is why every collected step stores all three labels. The `waypoint` label
is the most transferable across Flightmare, AirSim Drone Racing Lab, and real
platforms because a detector/SLAM module can provide the same gate-relative
input. `ctbr` is a useful end-to-end visual-control ablation when the vehicle
inner loop accepts collective thrust plus body rates. `motor` is mainly a
sim-specific low-level ablation and should be treated as least transferable.

Then point a BC config at the dataset:

```yaml
data:
  type: "flightmare_bc"
  data_dir: "data/flightmare/expert_v1"
  action_type: "ctbr"          # or "waypoint" / "motor"
  image_size: 224
  augment: true
robotics:
  architecture:
    type: "flightmare_ctbr_bc"
    state_dim: 13
    action_dim: 4
    backbone: "resnet18"
```

## Schema

```
data/flightmare/expert_v1/
├── index.json                # episode list, lengths, action stats, splits
├── norm_stats.npz            # per-action-type mean/std (from training split)
└── episodes/
    └── ep_NNNNNN.h5
        ├── attrs: episode_id, length, dt, controller_name
        ├── obs/
        │   ├── state           [T, state_dim]      f32  chunks=(64, *)
        │   └── image_<cam>     [T, H, W, 3]        u8   chunks=(1, H, W, 3)
        ├── action/
        │   ├── waypoint        [T, 4]              f32  chunks=(256, 4)
        │   ├── ctbr            [T, 4]              f32  chunks=(256, 4)
        │   └── motor           [T, 4]              f32  chunks=(256, 4)
        ├── reference/
        │   ├── pos_des         [T, 3]              f32
        │   ├── vel_des         [T, 3]              f32
        │   └── yaw_des         [T]                 f32
        └── meta/done           [T]                 bool
```

Design choices:
* **One file per episode** → workers open distinct files, no SWMR / lock
  contention.
* **`chunks=(1, H, W, 3)` images, LZF compression** → each frame is a stand-
  alone chunk; one decompress per random-access read; LZF is roughly 2× faster
  than gzip for negligible size penalty on rendered Unity frames.
* **Raw uint8, not JPEG** → no per-frame CPU decode; rendered visual
  collection stores real Unity frames, while `--no-render` debug runs store
  blank frames by design.
* **All action labels per step** → run the expert once, train waypoint, CTBR,
  and motor ablations with no re-collection cost.

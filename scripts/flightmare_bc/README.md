# Flightmare BC dataset collection

Collects a privileged-expert imitation-learning dataset in the Flightmare
photo-realistic drone simulator. Each step records the visual + proprioceptive
observation **and both action labels** (waypoint+speed *and* CTBR / motor
commands) so the same dataset can train both action-output ablations of
`MLPFusionGaussianExpertActor`.

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

1. Sample a random sequence of waypoints inside a configurable bounding box.
2. Fit a **minimum-jerk** polynomial through them at a target average speed
   (`--speed-range`).
3. At each control tick (default 100 Hz) the **SE(3) geometric controller**
   converts the current reference state into a CTBR command
   `(thrust, ω_x, ω_y, ω_z)`.
4. The Flightmare quadrotor is stepped with that CTBR and the Unity bridge
   renders the requested camera frames.
5. Per step we record the **observation** plus *both* action labels:
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
    --seed 0
```

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
        ├── attrs: episode_id, length, dt, controller, action_clipped
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
* **Raw uint8, not JPEG** → no per-frame CPU decode; matters when the
  DataLoader is the bottleneck (which it usually is for visuomotor BC).
* **Both action labels per step** → run the expert once, train *both*
  ablations, no re-collection cost.

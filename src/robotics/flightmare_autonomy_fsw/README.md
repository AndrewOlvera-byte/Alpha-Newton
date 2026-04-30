# Flightmare Autonomy FSW

This package is an in-process ROS-style autonomy graph for state-only Flightmare
evaluation. It keeps the ROS boundaries explicit without adding a hard
`rclpy` dependency to benchmark runs:

- `FlightmareStateNode`: owns Flightmare, publishes state, applies CTBR commands.
- `MissionWorldModelNode`: updates `MissionWrapper` from state and gate map.
- `PolicyPlannerNode`: calls the registered actor's `model.predict(batch)`.
- `BaseControllerNode`: converts planner output to CTBR commands.
  - CTBR policies are clipped and passed through as collective thrust/body rates.
  - Waypoint policies are tracked by a closed-form double-integrator LQR gain
    feeding the geometric CTBR controller.

Run inside the Docker service:

```bash
docker compose exec alpha-newton python -m src.entrypoints.eval_flightmare \
  --exp flightmare/flightmare_ctbr_bc_state \
  --episodes 20
```

or:

```bash
docker compose exec alpha-newton python -m src.entrypoints.eval_flightmare \
  --config configs/exp/flightmare/flightmare_waypoint_bc_state.yaml \
  --episodes 20
```

Artifacts are written to:

```text
outputs/<run.name>/eval/<timestamp>_stats.json
outputs/<run.name>/eval/<timestamp>_trajectories.png
```

If the running container was built before `matplotlib` was added to
`requirements.txt`, install it once in the live container:

```bash
docker compose exec alpha-newton pip install 'matplotlib>=3.7.0'
```

Flightmare bindings are still supplied by the existing Dockerfile. If the
evaluator prints that it is using the numpy fallback, rebuild the image or run
inside the Flightmare-enabled container before treating metrics as real racing
results.

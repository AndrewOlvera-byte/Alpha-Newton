"""Schema-aware HDF5 writer for Flightmare BC episodes.

One file per episode. Within each file the per-step datasets are *chunked*
to match the access pattern of a multi-worker PyTorch ``DataLoader``:

* Images: ``chunks=(1, H, W, 3)`` so a single random-access read decompresses
  exactly one frame; LZF compression for fast decode.
* State / actions / references: ``chunks=(256, ...)`` - dense, batch-friendly.

See ``README.md`` for the full schema layout.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


# Compression: LZF is ~2-3x faster than gzip-1 for ~10% larger files;
# this trade is the right one for image-heavy datasets that are I/O bound.
_IMG_KW = dict(compression="lzf", shuffle=True)
_FLOAT_KW = dict(compression="lzf", shuffle=True)


class EpisodeWriter:
    """Streaming append-only writer for one episode.

    Usage:
        with EpisodeWriter(path, image_size, cameras, state_dim) as w:
            for t in range(T):
                w.append(state, images, actions, reference)
    """

    def __init__(
        self,
        path: str | Path,
        image_size: int,
        cameras: Iterable[str],
        state_dim: int,
        controller_name: str,
        dt: float,
        seed: int,
        episode_id: int,
        chunk_state: int = 256,
        chunk_action: int = 256,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cameras = tuple(cameras)
        self.image_size = int(image_size)
        self.state_dim = int(state_dim)
        self._h5 = h5py.File(self.path, "w")
        self._h5.attrs["episode_id"] = int(episode_id)
        self._h5.attrs["dt"] = float(dt)
        self._h5.attrs["seed"] = int(seed)
        self._h5.attrs["controller_name"] = str(controller_name)
        self._h5.attrs["image_size"] = int(image_size)
        self._h5.attrs["cameras"] = np.array(list(cameras), dtype=h5py.string_dtype())

        H = W = self.image_size
        self._ds_state = self._h5.create_dataset(
            "obs/state", shape=(0, state_dim), maxshape=(None, state_dim),
            chunks=(chunk_state, state_dim), dtype=np.float32, **_FLOAT_KW,
        )
        self._ds_images = {
            cam: self._h5.create_dataset(
                f"obs/image_{cam}", shape=(0, H, W, 3), maxshape=(None, H, W, 3),
                chunks=(1, H, W, 3), dtype=np.uint8, **_IMG_KW,
            )
            for cam in cameras
        }
        self._ds_action = {
            name: self._h5.create_dataset(
                f"action/{name}", shape=(0, 4), maxshape=(None, 4),
                chunks=(chunk_action, 4), dtype=np.float32, **_FLOAT_KW,
            )
            for name in ("waypoint", "ctbr", "motor")
        }
        self._ds_prev_action = None
        # Mission/perception-prior channels: next-K-gates in body frame +
        # progress + distance. Dim is set when first sample is appended.
        self._ds_mission = None
        self._ds_gate_index = self._h5.create_dataset(
            "mission/gate_index", shape=(0,), maxshape=(None,),
            chunks=(chunk_action,), dtype=np.int32,
        )
        self._ds_ref_pos = self._h5.create_dataset(
            "reference/pos_des", shape=(0, 3), maxshape=(None, 3),
            chunks=(chunk_action, 3), dtype=np.float32, **_FLOAT_KW,
        )
        self._ds_ref_vel = self._h5.create_dataset(
            "reference/vel_des", shape=(0, 3), maxshape=(None, 3),
            chunks=(chunk_action, 3), dtype=np.float32, **_FLOAT_KW,
        )
        self._ds_ref_yaw = self._h5.create_dataset(
            "reference/yaw_des", shape=(0,), maxshape=(None,),
            chunks=(chunk_action,), dtype=np.float32, **_FLOAT_KW,
        )
        self._ds_done = self._h5.create_dataset(
            "meta/done", shape=(0,), maxshape=(None,), chunks=(chunk_action,), dtype=bool,
        )
        self._ds_expert = None
        self._t = 0

    @staticmethod
    def _grow(ds, n: int) -> int:
        old = ds.shape[0]
        ds.resize(old + n, axis=0)
        return old

    def append(
        self,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        actions: dict[str, np.ndarray],
        ref_pos: np.ndarray,
        ref_vel: np.ndarray,
        ref_yaw: float,
        done: bool,
        mission: np.ndarray | None = None,
        gate_index: int = -1,
        prev_actions: dict[str, np.ndarray] | None = None,
        expert: dict | None = None,
    ) -> None:
        i = self._grow(self._ds_state, 1)
        self._ds_state[i] = state.astype(np.float32, copy=False)
        for cam, ds in self._ds_images.items():
            self._grow(ds, 1)
            ds[i] = images[cam]
        for name, ds in self._ds_action.items():
            self._grow(ds, 1)
            ds[i] = actions[name].astype(np.float32, copy=False)
        if prev_actions is not None:
            if self._ds_prev_action is None:
                self._ds_prev_action = {
                    name: self._h5.create_dataset(
                        f"action/{name}_prev", shape=(0, 4), maxshape=(None, 4),
                        chunks=(256, 4), dtype=np.float32, **_FLOAT_KW,
                    )
                    for name in ("waypoint", "ctbr", "motor")
                }
            for name, ds in self._ds_prev_action.items():
                self._grow(ds, 1)
                value = prev_actions.get(name)
                if value is None:
                    value = np.zeros(4, dtype=np.float32)
                ds[i] = np.asarray(value, dtype=np.float32)
        elif self._ds_prev_action is not None:
            for _, ds in self._ds_prev_action.items():
                self._grow(ds, 1)
                ds[i] = np.zeros(4, dtype=np.float32)
        if mission is not None:
            mission = np.asarray(mission, dtype=np.float32)
            if self._ds_mission is None:
                self._ds_mission = self._h5.create_dataset(
                    "mission/vec", shape=(0, mission.shape[0]),
                    maxshape=(None, mission.shape[0]),
                    chunks=(256, mission.shape[0]), dtype=np.float32,
                    **_FLOAT_KW,
                )
            self._grow(self._ds_mission, 1)
            self._ds_mission[i] = mission
        self._grow(self._ds_gate_index, 1)
        self._ds_gate_index[i] = int(gate_index)
        self._grow(self._ds_ref_pos, 1)
        self._ds_ref_pos[i] = ref_pos.astype(np.float32, copy=False)
        self._grow(self._ds_ref_vel, 1)
        self._ds_ref_vel[i] = ref_vel.astype(np.float32, copy=False)
        self._grow(self._ds_ref_yaw, 1)
        self._ds_ref_yaw[i] = float(ref_yaw)
        self._grow(self._ds_done, 1)
        self._ds_done[i] = bool(done)
        if expert is not None:
            self._append_expert(i, expert)
        elif self._ds_expert is not None:
            self._append_expert(i, {})
        self._t += 1

    def _ensure_expert_datasets(self) -> None:
        if self._ds_expert is not None:
            return
        str_dtype = h5py.string_dtype(encoding="utf-8")
        self._ds_expert = {
            "weight": self._h5.create_dataset(
                "expert/weight", shape=(0,), maxshape=(None,),
                chunks=(256,), dtype=np.float32, **_FLOAT_KW,
            ),
            "confidence": self._h5.create_dataset(
                "expert/confidence", shape=(0,), maxshape=(None,),
                chunks=(256,), dtype=np.float32, **_FLOAT_KW,
            ),
            "action_log_std": self._h5.create_dataset(
                "expert/action_log_std", shape=(0, 4), maxshape=(None, 4),
                chunks=(256, 4), dtype=np.float32, **_FLOAT_KW,
            ),
            "cost": self._h5.create_dataset(
                "expert/cost", shape=(0,), maxshape=(None,),
                chunks=(256,), dtype=np.float32, **_FLOAT_KW,
            ),
            "cost_margin": self._h5.create_dataset(
                "expert/cost_margin", shape=(0,), maxshape=(None,),
                chunks=(256,), dtype=np.float32, **_FLOAT_KW,
            ),
            "saturation_frac": self._h5.create_dataset(
                "expert/saturation_frac", shape=(0,), maxshape=(None,),
                chunks=(256,), dtype=np.float32, **_FLOAT_KW,
            ),
            "valid": self._h5.create_dataset(
                "expert/valid", shape=(0,), maxshape=(None,),
                chunks=(256,), dtype=bool,
            ),
            "source": self._h5.create_dataset(
                "expert/source", shape=(0,), maxshape=(None,),
                chunks=(256,), dtype=str_dtype,
            ),
            "safety_status": self._h5.create_dataset(
                "expert/safety_status", shape=(0,), maxshape=(None,),
                chunks=(256,), dtype=str_dtype,
            ),
        }
        if self._t > 0:
            for name, ds in self._ds_expert.items():
                ds.resize(self._t, axis=0)
                if name == "weight":
                    ds[:] = 1.0
                elif name == "confidence":
                    ds[:] = 1.0
                elif name == "action_log_std":
                    ds[:] = np.asarray([-2.0, -2.0, -2.0, -2.0], dtype=np.float32)
                elif name == "valid":
                    ds[:] = True
                elif name in {"source", "safety_status"}:
                    ds[:] = ""
                else:
                    ds[:] = np.nan

    def _append_expert(self, i: int, expert: dict) -> None:
        self._ensure_expert_datasets()
        assert self._ds_expert is not None
        for ds in self._ds_expert.values():
            if ds.shape[0] <= i:
                ds.resize(i + 1, axis=0)
        self._ds_expert["weight"][i] = float(expert.get("weight", 1.0))
        self._ds_expert["confidence"][i] = float(expert.get("confidence", 1.0))
        self._ds_expert["action_log_std"][i] = np.asarray(
            expert.get("action_log_std", [-2.0, -2.0, -2.0, -2.0]),
            dtype=np.float32,
        )
        self._ds_expert["cost"][i] = float(expert.get("cost", np.nan))
        self._ds_expert["cost_margin"][i] = float(expert.get("cost_margin", np.nan))
        self._ds_expert["saturation_frac"][i] = float(expert.get("saturation_frac", 0.0))
        self._ds_expert["valid"][i] = bool(expert.get("valid", True))
        self._ds_expert["source"][i] = str(expert.get("source", ""))
        self._ds_expert["safety_status"][i] = str(expert.get("safety_status", ""))

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.attrs["length"] = int(self._t)
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

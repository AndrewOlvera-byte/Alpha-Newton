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
    ) -> None:
        i = self._grow(self._ds_state, 1)
        self._ds_state[i] = state.astype(np.float32, copy=False)
        for cam, ds in self._ds_images.items():
            self._grow(ds, 1)
            ds[i] = images[cam]
        for name, ds in self._ds_action.items():
            self._grow(ds, 1)
            ds[i] = actions[name].astype(np.float32, copy=False)
        self._grow(self._ds_ref_pos, 1)
        self._ds_ref_pos[i] = ref_pos.astype(np.float32, copy=False)
        self._grow(self._ds_ref_vel, 1)
        self._ds_ref_vel[i] = ref_vel.astype(np.float32, copy=False)
        self._grow(self._ds_ref_yaw, 1)
        self._ds_ref_yaw[i] = float(ref_yaw)
        self._grow(self._ds_done, 1)
        self._ds_done[i] = bool(done)
        self._t += 1

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.attrs["length"] = int(self._t)
            self._h5.close()
            self._h5 = None

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

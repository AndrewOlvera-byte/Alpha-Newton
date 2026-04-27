"""Reusable Unity-per-worker launcher for Flightmare visual envs.

Used by:
  * BC data collection (``parallel_collect.py``)
  * Future visual RL vec-env (one Unity per sub-process worker)

Design
------
Each worker is a separate OS process. Inside the worker we:
  1. Allocate a pub/sub ZMQ port pair from a base offset (worker_id * 2).
  2. Launch the Unity binary with ``-input-port``/``-output-port`` matching.
  3. Set ``FLIGHTMARE_PUB_PORT`` / ``FLIGHTMARE_SUB_PORT`` env vars in the
     same process so the patched ``unity_bridge.cpp`` reads them when
     ``flightgym`` is imported.
  4. Run the worker's task (collect a shard, or roll out RL transitions).
  5. On exit, terminate the Unity child process.

Prerequisites
-------------
``unity_bridge.cpp`` must be patched (see ``patch_unity_bridge.py``) and the
flightlib wheel rebuilt. Without that, the C++ side ignores the env vars and
all workers race for ports 10253/10254.
"""
from __future__ import annotations

import os
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


DEFAULT_BASE_PUB_PORT = 10253
DEFAULT_BASE_SUB_PORT = 10254
DEFAULT_PORT_STRIDE = 4  # leave 4 ports per worker so future bridge channels don't clash
DEFAULT_UNITY_BIN = "/opt/flightmare/flightrender/RPG_Flightmare.x86_64"


def allocate_ports(worker_id: int, base_pub: int = DEFAULT_BASE_PUB_PORT,
                   base_sub: int = DEFAULT_BASE_SUB_PORT,
                   stride: int = DEFAULT_PORT_STRIDE) -> tuple[int, int]:
    """Deterministic port pair for ``worker_id``."""
    return base_pub + worker_id * stride, base_sub + worker_id * stride


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def configure_env_for_worker(worker_id: int, **kwargs) -> dict[str, str]:
    """Return an env dict (current env + Flightmare port overrides) for this worker.

    Caller can pass it to ``subprocess.Popen(env=...)`` or update ``os.environ``.
    """
    pub, sub = allocate_ports(worker_id, **kwargs)
    env = dict(os.environ)
    env["FLIGHTMARE_PUB_PORT"] = str(pub)
    env["FLIGHTMARE_SUB_PORT"] = str(sub)
    return env


def launch_unity(
    pub_port: int,
    sub_port: int,
    unity_bin: str | Path = DEFAULT_UNITY_BIN,
    headless: bool = True,
    log_path: str | Path | None = None,
    startup_s: float = 5.0,
) -> subprocess.Popen:
    """Spawn a single Unity instance on a chosen port pair.

    The Flightmare Unity build accepts ``-input-port`` / ``-output-port`` plus
    standard Unity headless flags. If the user's binary uses different flag
    names, override via the ``FLIGHTMARE_UNITY_ARGS`` env var (space-separated).
    """
    unity_bin = str(unity_bin)
    if not Path(unity_bin).exists():
        raise FileNotFoundError(f"Unity binary not found: {unity_bin}")
    if not _port_free(pub_port):
        raise RuntimeError(f"port {pub_port} already in use; another Unity may be running")

    args: list[str] = [unity_bin]
    extra = os.environ.get("FLIGHTMARE_UNITY_ARGS", "").split()
    if extra:
        args.extend(extra)
    args.extend([
        "-input-port", str(pub_port),
        "-output-port", str(sub_port),
    ])
    if headless:
        args.extend(["-batchmode", "-nographics"])

    log = open(log_path, "w") if log_path else subprocess.DEVNULL
    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
    time.sleep(max(0.0, float(startup_s)))
    if proc.poll() is not None:
        raise RuntimeError(
            f"Unity exited immediately (code={proc.returncode}). "
            f"Inspect {log_path or 'stdout'}."
        )
    return proc


@contextmanager
def unity_for_worker(
    worker_id: int,
    *,
    unity_bin: str | Path = DEFAULT_UNITY_BIN,
    headless: bool = True,
    log_dir: str | Path | None = None,
    startup_s: float = 5.0,
):
    """Context manager: brings up a Unity instance + sets the port env vars.

    Usage:
        with unity_for_worker(worker_id=2, log_dir="logs") as info:
            # info = {"pub_port": ..., "sub_port": ..., "proc": <Popen>}
            run_collect_shard(...)

    Inside the ``with`` block, ``os.environ`` has the right port pair, so any
    ``import flightgym`` after this point picks them up.
    """
    pub, sub = allocate_ports(worker_id)
    os.environ["FLIGHTMARE_PUB_PORT"] = str(pub)
    os.environ["FLIGHTMARE_SUB_PORT"] = str(sub)
    log_path = None
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"unity_w{worker_id}.log"
    proc = launch_unity(pub, sub, unity_bin=unity_bin, headless=headless,
                        log_path=log_path, startup_s=startup_s)
    try:
        yield {"pub_port": pub, "sub_port": sub, "proc": proc, "log_path": log_path}
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

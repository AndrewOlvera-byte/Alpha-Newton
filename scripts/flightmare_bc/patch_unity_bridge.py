"""Patch Flightmare's UnityBridge so ZMQ ports come from env vars.

Flightmare's upstream ``unity_bridge.cpp`` hardcodes the pub/sub ports to
10253/10254. That makes it impossible to run more than one Unity-rendered env
on the same network namespace. This patch replaces the constructor's port
initializers with reads from ``FLIGHTMARE_PUB_PORT`` / ``FLIGHTMARE_SUB_PORT``
(falling back to the upstream defaults), so each worker process can set its
own port pair before importing flightgym.

The patch is idempotent. After running it, rebuild the flightlib wheel:

    python /workspace/scripts/flightmare_bc/patch_unity_bridge.py
    python /workspace/scripts/flightmare_bc/patch_flightmare_pybind.py
    pip install --no-deps --force-reinstall /opt/flightmare/flightlib

A successful patch lets ``parallel_collect.py`` spawn N workers each with its
own Unity instance on a unique port pair.
"""
from __future__ import annotations

from pathlib import Path


SENTINEL = "// FLIGHTMARE_PORT_ENV_PATCH"

OLD = '''pub_port_("10253"),
    sub_port_("10254"),'''

NEW = f'''pub_port_(std::getenv("FLIGHTMARE_PUB_PORT") ? std::getenv("FLIGHTMARE_PUB_PORT") : "10253"),  {SENTINEL}
    sub_port_(std::getenv("FLIGHTMARE_SUB_PORT") ? std::getenv("FLIGHTMARE_SUB_PORT") : "10254"),'''


def main() -> None:
    src = Path("/opt/flightmare/flightlib/src/bridges/unity_bridge.cpp")
    if not src.exists():
        raise SystemExit(f"unity_bridge.cpp not found at {src}")
    text = src.read_text()
    if SENTINEL in text:
        print(f"[patch] {src} already patched.")
        return
    if OLD not in text:
        raise SystemExit(
            "[patch] could not find the expected port-initializer block. "
            "Either Flightmare's upstream changed or the file was already "
            "modified by hand. Inspect manually."
        )
    src.write_text(text.replace(OLD, NEW))
    print(f"[patch] applied env-var port override to {src}")
    print("[patch] rebuild with: pip install --no-deps --force-reinstall /opt/flightmare/flightlib")


if __name__ == "__main__":
    main()

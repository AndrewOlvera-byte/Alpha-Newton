"""Quality + diversity analysis for a procedural BC dataset (index.json + h5).

    docker compose exec -T alpha-newton python -m scratch.analysis.analyze_bc_v7_dataset \
        --data-dir data/flightmare/bc_v7 --plot-dir scratch/analysis/bc_v7_report
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def _stat(xs):
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return "n=0"
    q = np.percentile(xs, [5, 50, 95])
    return f"n={xs.size:5d} min={xs.min():6.2f} p5={q[0]:6.2f} med={q[1]:6.2f} p95={q[2]:6.2f} max={xs.max():6.2f}"


def _heading_changes(centers):
    out = []
    for i in range(1, len(centers) - 1):
        a, b = centers[i] - centers[i - 1], centers[i + 1] - centers[i]
        na, nb = np.linalg.norm(a[:2]), np.linalg.norm(b[:2])
        if na < 1e-6 or nb < 1e-6:
            continue
        out.append(np.degrees(np.arccos(np.clip(np.dot(a[:2], b[:2]) / (na * nb), -1, 1))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--plot-dir", type=Path, default=None)
    ap.add_argument("--max-h5", type=int, default=200, help="how many episodes to open for label stats")
    args = ap.parse_args()

    manifest = json.load(open(args.data_dir / "index.json"))
    eps = manifest["episodes"]
    print(f"# Dataset: {args.data_dir}  ({len(eps)} episodes)")
    print(f"# controller={manifest.get('controller')} speed_range={manifest.get('speed_range')}")

    # ---- composition ----
    kinds = Counter(e.get("sample_kind", "expert") for e in eps)
    splits = Counter(e.get("split") for e in eps)
    fams = Counter((e.get("scenario") or {}).get("scenario_family", "none") for e in eps)
    print("\n## Composition")
    print("  sample_kinds:", dict(kinds))
    print("  splits      :", dict(splits))
    print("  families    :", dict(fams))

    # ---- quality (expert episodes only: strict closed-loop completion) ----
    exp = [e for e in eps if e.get("sample_kind", "expert") == "expert"]
    comp = [float(e.get("gate_completion", e.get("strict_gate_completion", 0.0))) for e in exp]
    disc = [e for e in eps if e.get("split") == "discard"]
    print("\n## Expert label quality (strict aperture validation)")
    print(f"  expert episodes        : {len(exp)}")
    if comp:
        print(f"  strict gate_completion : {_stat(comp)}")
        print(f"  fully completed (==1)  : {np.mean(np.asarray(comp) >= 0.999) * 100:.1f}%")
    print(f"  discarded episodes      : {len(disc)} ({100*len(disc)/max(1,len(eps)):.1f}%)")
    print(f"  expert mean_track_err  : {_stat([e.get('mean_track_err', 0.0) for e in exp])}")
    print(f"  expert max_track_err   : {_stat([e.get('max_track_err', 0.0) for e in exp])}")

    # ---- per-family quality + diversity ----
    print("\n## Per-family quality + geometry diversity")
    by_fam = defaultdict(list)
    for e in eps:
        by_fam[(e.get("scenario") or {}).get("scenario_family", "none")].append(e)
    hdr = f"  {'family':20s} {'n':>4s} {'gc(exp)':>8s} {'len_m':>7s} {'gates':>6s} {'inv':>5s} {'turn_deg':>9s} {'speed':>6s}"
    print(hdr)
    for fam, group in sorted(by_fam.items()):
        gexp = [e for e in group if e.get("sample_kind", "expert") == "expert"]
        gc = np.mean([float(e.get("gate_completion", 0.0)) for e in gexp]) if gexp else float("nan")
        lengths, ng, ninv, turns, spd = [], [], [], [], []
        for e in group:
            sc = e.get("scenario") or {}
            lengths.append(sc.get("path_length_m", 0.0))
            ninv.append(sc.get("num_inverted_gates", 0))
            spd.append(e.get("avg_speed", 0.0))
            centers = np.array([g["pos"] for g in e.get("gates", [])], dtype=np.float64)
            ng.append(len(centers))
            turns.extend(_heading_changes(centers))
        print(f"  {fam:20s} {len(group):4d} {gc:8.3f} {np.mean(lengths):7.1f} "
              f"{np.mean(ng):6.1f} {np.mean(ninv):5.2f} {np.mean(turns) if turns else 0:9.1f} {np.mean(spd):6.2f}")

    # ---- start-direction diversity (gate0 heading) ----
    headings = []
    for e in eps:
        gates = e.get("gates", [])
        if gates:
            headings.append(np.degrees(float(gates[0].get("yaw", 0.0))) % 360)
    if headings:
        hist, _ = np.histogram(headings, bins=8, range=(0, 360))
        print("\n## Start-direction diversity (gate0 yaw, 8 bins over 360deg)")
        print("  counts:", hist.tolist(), "(uniform-ish == diverse directions)")

    # ---- label distributions from h5 ----
    try:
        import h5py
        files = [args.data_dir / e["path"] for e in eps if e.get("split") == "train"][: args.max_h5]
        thr, wmax, spd_lbl = [], [], []
        for f in files:
            with h5py.File(f, "r") as h:
                if "action/ctbr" in h:
                    a = h["action/ctbr"][:]
                    thr.append(a[:, 0]); wmax.append(np.abs(a[:, 1:4]))
                if "action/waypoint" in h:
                    spd_lbl.append(h["action/waypoint"][:, 3])
        print(f"\n## CTBR label distributions ({len(files)} episodes)")
        if thr:
            print("  thrust_norm   :", _stat(np.concatenate(thr)))
            print("  |body_rate|   :", _stat(np.concatenate(wmax).reshape(-1)))
        if spd_lbl:
            print("  waypoint speed:", _stat(np.concatenate(spd_lbl)))
    except Exception as exc:  # noqa: BLE001
        print("  (label stats skipped:", exc, ")")

    # ---- plots ----
    if args.plot_dir is not None:
        _plots(eps, by_fam, args.plot_dir)


def _plots(eps, by_fam, plot_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    families = [f for f in by_fam if f != "none"]
    # Top-down shapes: a few sample courses per family.
    n = len(families)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8))
    axes = np.atleast_1d(axes).ravel()
    rng = np.random.default_rng(0)
    for ax, fam in zip(axes, families):
        group = by_fam[fam]
        for e in [group[i] for i in rng.choice(len(group), size=min(6, len(group)), replace=False)]:
            c = np.array([g["pos"] for g in e["gates"]], dtype=np.float64)
            c = c - c[0]  # align starts at origin for shape comparison
            ax.plot(c[:, 0], c[:, 1], "-o", ms=3, lw=1, alpha=0.7)
        ax.set_title(fam); ax.set_aspect("equal"); ax.grid(alpha=0.3)
    for ax in axes[len(families):]:
        ax.axis("off")
    fig.suptitle("Top-down course shapes (start-aligned) per scenario family")
    fig.tight_layout()
    fig.savefig(plot_dir / "course_shapes_topdown.png", dpi=110)
    plt.close(fig)

    # Altitude profiles.
    fig, ax = plt.subplots(figsize=(8, 4))
    for fam in families:
        for e in by_fam[fam][:8]:
            c = np.array([g["pos"] for g in e["gates"]], dtype=np.float64)
            s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
            ax.plot(s, c[:, 2], "-", lw=0.8, alpha=0.5)
    ax.set_xlabel("path length (m)"); ax.set_ylabel("altitude z (m)")
    ax.set_title("Altitude profiles along the course"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(plot_dir / "altitude_profiles.png", dpi=110)
    plt.close(fig)
    print(f"\n## Plots -> {plot_dir}/course_shapes_topdown.png, altitude_profiles.png")


if __name__ == "__main__":
    main()

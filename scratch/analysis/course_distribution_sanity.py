"""Sanity audit for the procedural course distribution.

Samples N courses across all scenario families, asserts 100% pass the geometry
validator, and prints distribution stats (path length, gate counts, inverted
counts, spacing, altitude, heading change) so we can confirm realistic,
physically-flyable shapes before a real collection run.

    docker compose exec -T alpha-newton python -m scratch.analysis.course_distribution_sanity --n 2000
"""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from src.robotics.flightmare_courses import (
    default_course_distribution,
    sample_course,
    validate_course,
)
from src.robotics.flightmare_courses.validator import path_length


def _heading_changes(centers: np.ndarray) -> list[float]:
    out = []
    for i in range(1, len(centers) - 1):
        a = centers[i] - centers[i - 1]
        b = centers[i + 1] - centers[i]
        na, nb = np.linalg.norm(a[:2]), np.linalg.norm(b[:2])
        if na < 1e-6 or nb < 1e-6:
            continue
        cos = float(np.clip(np.dot(a[:2], b[:2]) / (na * nb), -1.0, 1.0))
        out.append(np.degrees(np.arccos(cos)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dist = default_course_distribution()
    fams = Counter()
    invalid = 0
    lengths, n_gates, n_inv, spac, zmin, zmax, hmax = [], [], [], [], [], [], []
    for _ in range(args.n):
        gc = sample_course(rng, dist)
        fams[gc.scenario_family] += 1
        ok, reasons = validate_course(gc.gates, dist.bounds, waypoints=gc.waypoints)
        if not ok:
            invalid += 1
            print("INVALID", gc.scenario_name, reasons[:3])
            continue
        centers = np.stack([g.pos for g in gc.gates])
        lengths.append(path_length(gc.gates))
        n_gates.append(len(gc.gates))
        n_inv.append(gc.num_inverted_gates)
        if len(centers) > 1:
            spac.extend(np.linalg.norm(np.diff(centers, axis=0), axis=1).tolist())
        zmin.append(centers[:, 2].min())
        zmax.append(centers[:, 2].max())
        hc = _heading_changes(centers)
        hmax.append(max(hc) if hc else 0.0)

    def stat(name, xs):
        xs = np.asarray(xs)
        print(f"  {name:16s} min={xs.min():7.2f} mean={xs.mean():7.2f} max={xs.max():7.2f}")

    print(f"sampled {args.n} courses; invalid={invalid}")
    print("family mix:", dict(fams))
    stat("path_length_m", lengths)
    stat("num_gates", n_gates)
    stat("num_inverted", n_inv)
    stat("gate_spacing_m", spac)
    stat("min_alt_m", zmin)
    stat("max_alt_m", zmax)
    stat("max_heading_deg", hmax)
    assert invalid == 0, f"{invalid} invalid courses sampled"
    print("OK: all sampled courses pass the geometry validator")


if __name__ == "__main__":
    main()

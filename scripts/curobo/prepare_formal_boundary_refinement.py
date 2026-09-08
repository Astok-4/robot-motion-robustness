#!/usr/bin/env python3
"""Prepare the next deterministic formal boundary-bisection round."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_formal/rq1_rq2_robustness_v1"
sys.path.insert(0, str(ROOT / "scripts/curobo"))
from prepare_canonical_formal_rq1_rq2_trials import (  # noqa: E402
    DIRECTIONS, EXPECTED, JOINT_NAMES, MATRIX, evaluate, load_trajectory_spheres, sha256, write_csv,
)


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truth(value: str) -> bool:
    return value.lower() == "true"


def main() -> None:
    for path, expected in EXPECTED.items():
        if sha256(path) != expected:
            raise RuntimeError(f"frozen asset SHA mismatch: {path}")
    coarse = rows(OUT / "coarse_trials.csv")
    if len(coarse) != 76800:
        raise RuntimeError("coarse dataset must be complete")
    existing_results = rows(OUT / "boundary_refinement_trials.csv")
    existing_prepared = rows(OUT / "prepared_boundary_refinement_trials.csv")
    result_by_base: dict[tuple[str, str], list[tuple[float, bool]]] = {}
    for row in existing_results:
        result_by_base.setdefault((row["condition_id"], row["direction_id"]), []).append(
            (float(row["magnitude_mm"]), truth(row["physx_collision"]))
        )
    coarse_by_pair: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in coarse:
        coarse_by_pair.setdefault((row["condition_id"], row["direction_id"]), []).append(row)
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    condition_map = {c["condition_id"]: c for c in matrix["conditions"]}
    with DIRECTIONS.open(newline="", encoding="utf-8") as stream:
        direction_map = {
            r["direction_id"]: np.asarray([float(r["x"]), float(r["y"]), float(r["z"])])
            for r in csv.DictReader(stream)
        }
    new_specs: list[tuple[str, str, float]] = []
    observed = 0
    complete = 0
    for key, values in coarse_by_pair.items():
        values.sort(key=lambda r: float(r["magnitude_mm"]))
        prior_mag, prior_collision = 0.0, False
        bracket = None
        for row in values:
            magnitude, collision = float(row["magnitude_mm"]), truth(row["physx_collision"])
            if not prior_collision and collision:
                bracket = [prior_mag, magnitude]
                break
            prior_mag, prior_collision = magnitude, collision
        if bracket is None:
            continue
        observed += 1
        for magnitude, collision in sorted(result_by_base.get(key, [])):
            if bracket[0] < magnitude < bracket[1]:
                bracket[1 if collision else 0] = magnitude
        if bracket[1] - bracket[0] <= 0.5 + 1e-12:
            complete += 1
            continue
        midpoint = (bracket[0] + bracket[1]) / 2.0
        new_specs.append((key[0], key[1], midpoint))
    prepared_ids = {row["trial_id"] for row in existing_prepared}
    prepared = list(existing_prepared)
    sphere_cache = {}
    for condition_id, direction_id, magnitude in new_specs:
        trial_id = f"{condition_id}__{direction_id}__refine_{magnitude:07.3f}mm"
        if trial_id in prepared_ids:
            continue
        condition = condition_map[condition_id]
        if condition_id not in sphere_cache:
            trajectory = np.asarray(condition["trajectory"]["position"], dtype=np.float32)
            sphere_cache[condition_id] = load_trajectory_spheres(trajectory, JOINT_NAMES)
        row = evaluate(condition, sphere_cache[condition_id], direction_id, direction_map[direction_id], magnitude)
        row["trial_id"] = trial_id
        prepared.append(row)
        prepared_ids.add(trial_id)
    if prepared:
        write_csv(OUT / "prepared_boundary_refinement_trials.csv", prepared)
    print(
        f"FORMAL_REFINEMENT_PREP observed_pairs={observed} completed_pairs={complete} "
        f"new_trials={len(prepared)-len(existing_prepared)} cumulative_trials={len(prepared)}"
    )


if __name__ == "__main__":
    main()

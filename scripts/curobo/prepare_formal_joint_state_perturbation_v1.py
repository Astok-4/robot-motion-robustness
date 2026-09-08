#!/usr/bin/env python3
"""Prepare formal constant single-joint bias coarse trials; no planning."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from prepare_joint_state_perturbation_pilot_v0 import JOINT_NAMES, evaluate, load_limits

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/formal_joint_state_perturbation_v1.json"
MATRIX = ROOT / "configs/experiments/formal_safety_margin_matrix_v1.json"
DIRECTIONS = ROOT / "configs/experiments/formal_joint_state_direction_set_v1.json"
OUT = ROOT / "results_formal/joint_state_perturbation_v1"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    direction_asset = json.loads(DIRECTIONS.read_text(encoding="utf-8"))
    directions = direction_asset["directions"]
    magnitudes = config["coarse_nonzero_magnitudes_deg"]
    conditions = [row for row in matrix["conditions"] if row["inflation_mm"] == 0]
    if config["status"] != "FROZEN" or matrix["matrix_status"] != "FROZEN":
        raise RuntimeError("formal assets must be frozen")
    if len(conditions) != 4 or len(directions) != 14 or magnitudes != [0.5,1.0,2.0,3.0,4.0,6.0,8.0]:
        raise RuntimeError("formal population differs from frozen protocol")
    if [d["direction_id"] for d in directions] != [
        f"{sign}_J{joint}" for joint in range(1,8) for sign in ("positive","negative")
    ]:
        raise RuntimeError("direction order mismatch")
    limits = load_limits()
    rows: list[dict[str, Any]] = []
    trajectories: dict[str, list[list[float]]] = {}
    obstacles: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        scene = condition["scene_id"]
        nominal = np.asarray(condition["trajectory"]["position"], dtype=np.float32)
        center = condition["nominal_obstacle_center_m"]
        size = condition["nominal_evaluation_obstacle_size_m"]
        if nominal.shape != (41,7) or size != [0.1,0.1,0.1]:
            raise RuntimeError(f"invalid formal condition: {scene}")
        obstacles[scene] = {"nominal_position_m": center, "size_m": size}
        c, clearance, idx, sphere = evaluate(nominal, center, size)
        if c != condition["nominal_curobo_collision"] or not math.isclose(clearance, condition["c_curobo_m"], abs_tol=1e-7):
            raise RuntimeError(f"nominal cuRobo mismatch: {scene}")
        nominal_id = f"{scene}__joint_nominal__000deg"
        rows.append({
            "trial_id": nominal_id, "scene_id": scene, "direction_id": "nominal",
            "joint_index": "", "joint_name": "nominal", "sign": "nominal",
            "magnitude_deg": 0.0, "magnitude_rad": 0.0, "joint_limit_valid": True,
            "first_invalid_state_index": "", "curobo_collision": c,
            "curobo_min_clearance_m": clearance, "curobo_min_state_index": idx,
            "curobo_min_sphere_index": sphere, "trajectory_state_count": 41,
        })
        trajectories[nominal_id] = nominal.tolist()
        for direction in directions:
            joint_index = int(direction["joint_index"])
            joint_name = direction["joint_name"]
            lower, upper = limits[joint_name]
            for magnitude_deg in magnitudes:
                magnitude_rad = math.radians(magnitude_deg)
                perturbed = nominal.copy()
                perturbed[:,joint_index] += float(direction["axis_value"]) * magnitude_rad
                invalid = np.flatnonzero((perturbed[:,joint_index] < lower) | (perturbed[:,joint_index] > upper))
                valid = len(invalid) == 0
                trial_id = f"{scene}__{direction['direction_id']}__{magnitude_deg:05.2f}deg"
                if valid:
                    pc, pd, pi, ps = evaluate(perturbed, center, size)
                else:
                    pc, pd, pi, ps = None, None, None, None
                rows.append({
                    "trial_id": trial_id, "scene_id": scene,
                    "direction_id": direction["direction_id"], "joint_index": joint_index,
                    "joint_name": joint_name, "sign": direction["sign"],
                    "magnitude_deg": magnitude_deg, "magnitude_rad": magnitude_rad,
                    "joint_limit_valid": valid,
                    "first_invalid_state_index": int(invalid[0]) if len(invalid) else "",
                    "curobo_collision": pc, "curobo_min_clearance_m": pd,
                    "curobo_min_state_index": pi, "curobo_min_sphere_index": ps,
                    "trajectory_state_count": 41,
                })
                trajectories[trial_id] = perturbed.tolist()
    if len(rows) != 396 or sum(float(r["magnitude_deg"]) > 0 for r in rows) != 392:
        raise RuntimeError("formal coarse population mismatch")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "prepared_coarse_trials.csv", rows)
    payload = {
        "joint_names": JOINT_NAMES,
        "joint_limits_rad": {k:list(v) for k,v in limits.items()},
        "scene_obstacles": obstacles,
        "trajectories": trajectories,
    }
    encoded = json.dumps(payload, separators=(",",":"), allow_nan=False).encode()
    (OUT / "prepared_coarse_trajectories.json").write_bytes(encoded)
    metadata = {
        "experiment_id": "formal_joint_state_perturbation_v1",
        "status": "prepared_sanity_pass", "experiment_role": "SECONDARY FORMAL EXTENSION",
        "config_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "matrix_sha256": hashlib.sha256(MATRIX.read_bytes()).hexdigest(),
        "direction_set_sha256": hashlib.sha256(DIRECTIONS.read_bytes()).hexdigest(),
        "scene_count": 4, "direction_count": 14, "coarse_nonzero_count": 392,
        "nominal_count": 4, "trajectory_shape": [41,7],
        "trajectory_frozen": True, "obstacle_unchanged": True,
        "motion_planner_invoked": False, "replanned_after_perturbation": False,
        "random_sampling_used": False, "combined_perturbation_used": False,
        "prepared_states_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("FORMAL_JOINT_STATE_V1_PREPARE_PASS coarse=392 nominal=4 replanning=false")


if __name__ == "__main__": main()

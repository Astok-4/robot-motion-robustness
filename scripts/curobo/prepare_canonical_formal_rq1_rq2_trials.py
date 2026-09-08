#!/usr/bin/env python3
"""Prepare the frozen formal trial population and cuRobo analytical results."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_random_environment_pilot import load_trajectory_spheres, sphere_box_clearance_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_formal/rq1_rq2_robustness_v1"
BASELINE = ROOT / "configs/experiments/formal_baseline_set_v1.json"
MATRIX = ROOT / "configs/experiments/formal_safety_margin_matrix_v1.json"
PROTOCOL = ROOT / "configs/experiments/formal_rq1_rq2_protocol_v1.json"
DIRECTIONS = ROOT / "configs/experiments/formal_direction_set_v1.csv"
EXPECTED = {
    BASELINE: "3a3b4806e2a407253006be509063ccf4b584c873bc5e8e8f9110949de94e8f6b",
    MATRIX: "8869b9709cf6c84274d77e2d28a2bf8354eb15f3e3740362da0b2d8871880d04",
    PROTOCOL: "eb270c2219460b44a1e9ff445b66dc40b18d1f4690383c66d6478419dba3022f",
    DIRECTIONS: "4ee274983d2e3db633a1543816e698733e0a07773016e80fa0bfd815d15f3ada",
}
MAGNITUDES = list(range(2, 61, 2))
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, separators=(",", ":")) if isinstance(v, list) else v for k, v in row.items()})


def evaluate(
    condition: dict[str, Any], spheres: torch.Tensor, direction_id: str,
    vector: np.ndarray, magnitude_mm: float,
) -> dict[str, Any]:
    delta = vector * magnitude_mm / 1000.0
    nominal = np.asarray(condition["nominal_obstacle_center_m"], dtype=np.float64)
    actual = nominal + delta
    clearance = sphere_box_clearance_matrix(
        spheres, actual.tolist(), condition["nominal_evaluation_obstacle_size_m"]
    )
    waypoint = torch.min(clearance, dim=-1).values
    minimum, index = torch.min(waypoint, dim=0)
    suffix = "nominal" if direction_id == "nominal" else f"{direction_id}__{int(magnitude_mm):03d}mm"
    return {
        "trial_id": f"{condition['condition_id']}__{suffix}",
        "condition_id": condition["condition_id"], "scene_id": condition["scene_id"],
        "inflation_mm": condition["inflation_mm"], "direction_id": direction_id,
        "magnitude_mm": magnitude_mm,
        "direction_x": float(vector[0]), "direction_y": float(vector[1]), "direction_z": float(vector[2]),
        "delta_x_m": float(delta[0]), "delta_y_m": float(delta[1]), "delta_z_m": float(delta[2]),
        "actual_obstacle_position_m": actual.tolist(),
        "curobo_minimum_clearance_m": float(minimum.item()),
        "curobo_minimum_clearance_index": int(index.item()),
        "curobo_collision": bool(minimum.item() <= 0.0),
        "trajectory_frozen": True, "replanned_after_perturbation": False,
    }


def main() -> None:
    for path, expected in EXPECTED.items():
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"frozen asset SHA mismatch: {path}: {actual}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for cuRobo kinematics-only evaluation")
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if matrix["matrix_status"] != "FROZEN" or len(matrix["conditions"]) != 20:
        raise ValueError("invalid frozen matrix")
    if protocol["protocol_status"] != "FROZEN" or protocol["perturbation_protocol"]["coarse_nonzero_magnitudes_mm"] != MAGNITUDES:
        raise ValueError("invalid frozen protocol magnitude grid")
    with DIRECTIONS.open(newline="", encoding="utf-8") as stream:
        raw_directions = list(csv.DictReader(stream))
    directions = []
    for row in raw_directions:
        vector = np.asarray([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float64)
        if not math.isclose(float(np.linalg.norm(vector)), 1.0, abs_tol=1e-12):
            raise ValueError(f"non-unit direction {row['direction_id']}")
        directions.append((row["direction_id"], vector))
    if len(directions) != 128:
        raise ValueError("expected 128 formal directions")

    nominal_rows: list[dict[str, Any]] = []
    coarse_rows: list[dict[str, Any]] = []
    for condition_index, condition in enumerate(matrix["conditions"], start=1):
        trajectory = np.asarray(condition["trajectory"]["position"], dtype=np.float32)
        if trajectory.shape != (41, 7) or condition["joint_names"] != JOINT_NAMES:
            raise ValueError(f"invalid trajectory {condition['condition_id']}")
        spheres = load_trajectory_spheres(trajectory, JOINT_NAMES)
        nominal = evaluate(condition, spheres, "nominal", np.zeros(3), 0.0)
        if not math.isclose(nominal["curobo_minimum_clearance_m"], condition["c_curobo_m"], abs_tol=1e-7):
            raise RuntimeError(f"nominal cuRobo clearance mismatch {condition['condition_id']}")
        nominal_rows.append(nominal)
        for direction_id, vector in directions:
            for magnitude in MAGNITUDES:
                coarse_rows.append(evaluate(condition, spheres, direction_id, vector, float(magnitude)))
        print(f"FORMAL_CUROBO_PREP_PROGRESS conditions={condition_index}/20 trials={len(coarse_rows)}", flush=True)
    if len(nominal_rows) != 20 or len(coarse_rows) != 76800:
        raise RuntimeError("unexpected formal trial population")
    trial_ids = [row["trial_id"] for row in nominal_rows + coarse_rows]
    if len(set(trial_ids)) != 76820:
        raise RuntimeError("duplicate formal trial IDs")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "prepared_nominal_trials.csv", nominal_rows)
    write_csv(OUT / "prepared_coarse_trials.csv", coarse_rows)
    metadata = {
        "experiment_id": "canonical_formal_rq1_rq2_robustness_v1",
        "status": "prepared", "backend": "sequential PhysX",
        "nominal_trial_count": 20, "coarse_trial_count": 76800,
        "condition_count": 20, "direction_count": 128, "coarse_magnitude_count": 30,
        "trajectory_shape": [41, 7], "trajectory_frozen": True,
        "replanned_after_perturbation": False, "motion_planner_invoked": False,
        "collision_criterion": "minimum reported separation <= 0.0 m",
        "frozen_sha256": {str(path.relative_to(ROOT)): value for path, value in EXPECTED.items()},
        "candidate_005_20mm_bottleneck_switch_retained": True,
    }
    (OUT / "prepared_experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("FORMAL_RQ1_RQ2_TRIAL_PREPARATION_PASS nominal=20 coarse=76800 randomness=false replanning=false")


if __name__ == "__main__":
    main()

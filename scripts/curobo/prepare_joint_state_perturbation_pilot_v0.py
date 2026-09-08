#!/usr/bin/env python3
"""Prepare deterministic constant-bias joint-state Pilot V0; no planning."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import torch

from run_random_environment_pilot import load_trajectory_spheres, sphere_box_clearance_matrix

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "configs/experiments/formal_baseline_v1.json"
URDF = ROOT / "curobo/curobo/content/assets/robot/franka_description/franka_panda.urdf"
OUT = ROOT / "results_pilot/joint_state_perturbation_v0"
MAGNITUDES_DEG = [0.25, 0.5, 1.0, 2.0, 3.0]
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_limits() -> dict[str, tuple[float, float]]:
    root = ET.parse(URDF).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in JOINT_NAMES:
            continue
        limit = joint.find("limit")
        if limit is None:
            raise RuntimeError(f"missing joint limit: {name}")
        limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    if list(limits) != JOINT_NAMES:
        raise RuntimeError(f"unexpected Franka joint limits: {limits}")
    return limits


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def evaluate(
    trajectory: np.ndarray,
    center: list[float],
    size: list[float],
) -> tuple[bool, float, int, int]:
    spheres = load_trajectory_spheres(trajectory, JOINT_NAMES)
    matrix = sphere_box_clearance_matrix(spheres, center, size)
    flat_index = int(torch.argmin(matrix).item())
    state_index, sphere_index = np.unravel_index(flat_index, tuple(matrix.shape))
    minimum = float(matrix[state_index, sphere_index].item())
    return minimum <= 0.0, minimum, int(state_index), int(sphere_index)


def main() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    if baseline.get("baseline_id") != "formal_baseline_v1" or baseline.get("baseline_status") != "FROZEN":
        raise RuntimeError("Pilot V0 requires frozen formal_baseline_v1")
    trajectory = np.asarray(baseline["nominal_trajectory"]["position"], dtype=np.float32)
    if trajectory.shape != (41, 7) or baseline["joint_names"] != JOINT_NAMES:
        raise RuntimeError("unexpected frozen trajectory")
    limits = load_limits()
    obstacle = baseline["obstacle"]
    nominal_collision, nominal_clearance, nominal_index, nominal_sphere = evaluate(
        trajectory, obstacle["nominal_position_m"], obstacle["size_m"]
    )
    expected_collision = bool(baseline["curobo_validation"]["nominal_collision"])
    expected_clearance = float(baseline["curobo_validation"]["minimum_clearance_m"])
    if nominal_collision != expected_collision or not math.isclose(nominal_clearance, expected_clearance, abs_tol=1e-7):
        raise RuntimeError("nominal cuRobo sanity check failed")

    specs: list[dict[str, Any]] = [{
        "trial_id": "formal_baseline_v1__joint_nominal__000deg",
        "scene_id": "formal_baseline_v1", "joint_index": "", "joint_name": "nominal",
        "sign": "nominal", "magnitude_deg": 0.0, "magnitude_rad": 0.0,
        "valid_joint_limits": True, "first_invalid_state_index": "",
        "curobo_collision": nominal_collision,
        "curobo_min_clearance_m": nominal_clearance,
        "curobo_collision_or_min_index": nominal_index,
        "curobo_min_sphere_index": nominal_sphere,
        "trajectory_state_count": 41,
    }]
    trajectories: dict[str, list[list[float]]] = {specs[0]["trial_id"]: trajectory.tolist()}
    first_nonzero_checked = False
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        lower, upper = limits[joint_name]
        for sign_name, sign_value in (("positive", 1.0), ("negative", -1.0)):
            for magnitude_deg in MAGNITUDES_DEG:
                magnitude_rad = math.radians(magnitude_deg)
                perturbed = trajectory.copy()
                perturbed[:, joint_index] += sign_value * magnitude_rad
                invalid = np.flatnonzero(
                    (perturbed[:, joint_index] < lower) | (perturbed[:, joint_index] > upper)
                )
                valid = len(invalid) == 0
                trial_id = (
                    f"formal_baseline_v1__joint_{joint_index + 1}_{sign_name}__"
                    f"{magnitude_deg:05.2f}deg"
                )
                if not first_nonzero_checked:
                    changed = perturbed - trajectory
                    if not np.allclose(changed[:, joint_index], sign_value * magnitude_rad, atol=1e-8):
                        raise RuntimeError("constant-bias sanity check failed")
                    if np.count_nonzero(np.delete(changed, joint_index, axis=1)) != 0:
                        raise RuntimeError("non-target joints changed")
                    first_nonzero_checked = True
                if valid:
                    collision, clearance, minimum_index, sphere_index = evaluate(
                        perturbed, obstacle["nominal_position_m"], obstacle["size_m"]
                    )
                else:
                    collision, clearance, minimum_index, sphere_index = None, None, None, None
                specs.append({
                    "trial_id": trial_id, "scene_id": "formal_baseline_v1",
                    "joint_index": joint_index, "joint_name": joint_name, "sign": sign_name,
                    "magnitude_deg": magnitude_deg, "magnitude_rad": magnitude_rad,
                    "valid_joint_limits": valid,
                    "first_invalid_state_index": int(invalid[0]) if len(invalid) else "",
                    "curobo_collision": collision,
                    "curobo_min_clearance_m": clearance,
                    "curobo_collision_or_min_index": minimum_index,
                    "curobo_min_sphere_index": sphere_index,
                    "trajectory_state_count": 41,
                })
                trajectories[trial_id] = perturbed.tolist()
    if len(specs) != 71 or len(trajectories) != 71:
        raise RuntimeError("Pilot V0 population mismatch")

    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "prepared_joint_state_pilot_v0.csv", specs)
    trajectory_payload = {
        "scene_id": "formal_baseline_v1",
        "obstacle": obstacle,
        "joint_names": JOINT_NAMES,
        "joint_limits_rad": {name: list(values) for name, values in limits.items()},
        "trajectories": trajectories,
    }
    encoded = json.dumps(trajectory_payload, separators=(",", ":"), allow_nan=False).encode()
    (OUT / "prepared_trajectories.json").write_bytes(encoded)
    metadata = {
        "experiment_id": "joint_state_perturbation_pilot_v0",
        "status": "prepared_sanity_pass",
        "label": "PRELIMINARY / FEASIBILITY PILOT — NOT FORMAL PAPER EVIDENCE",
        "baseline_file": str(BASELINE.relative_to(ROOT)),
        "baseline_sha256": hashlib.sha256(BASELINE.read_bytes()).hexdigest(),
        "trajectory_shape": [41, 7],
        "direction_count": 14, "nonzero_magnitude_count": 5,
        "nonzero_condition_count": 70, "nominal_condition_count": 1,
        "magnitudes_deg": MAGNITUDES_DEG,
        "bias_semantics": "same deterministic single-joint offset added to all 41 states",
        "obstacle_unchanged": True, "trajectory_frozen_before_bias": True,
        "replanned_after_perturbation": False, "motion_planner_invoked": False,
        "random_sampling_used": False,
        "prepared_trajectory_sha256": sha256_bytes(encoded),
        "sanity_checks": {
            "nominal_curobo_reproduced": True,
            "constant_bias_applied_to_all_states": True,
            "only_target_joint_changed": True,
            "obstacle_pose_unchanged": True,
            "trajectory_state_count_41": True,
            "no_replanning_code_path": True,
        },
    }
    (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("JOINT_STATE_PILOT_V0_PREPARE_SANITY_PASS trials=71 replanning=false randomness=false")


if __name__ == "__main__":
    main()

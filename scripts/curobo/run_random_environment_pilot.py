#!/usr/bin/env python3
"""Run a deterministic small XY obstacle-position perturbation pilot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


RANDOM_SEED = 20260816
MAGNITUDES_MM = [0.0, 2.0, 5.0, 10.0, 15.0, 20.0]
NONZERO_TRIAL_COUNT = 16


def parse_args() -> argparse.Namespace:
    """Parse frozen baseline and isolated pilot output paths."""
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline",
        type=Path,
        default=project_dir / "configs" / "experiments" / "formal_baseline_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results_pilot" / "random_environment",
    )
    return parser.parse_args()


def load_baseline(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    """Load and validate Formal Baseline V1."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        baseline = json.load(stream)
    if baseline.get("baseline_id") != "formal_baseline_v1" or baseline.get("baseline_status") != "FROZEN":
        raise ValueError("pilot requires FROZEN formal_baseline_v1")
    if baseline.get("source_candidate") != "candidate_001":
        raise ValueError("Formal Baseline V1 must come from candidate_001")
    trajectory = np.asarray(baseline["nominal_trajectory"]["position"], dtype=np.float32)
    if trajectory.shape != (41, 7) or not np.isfinite(trajectory).all():
        raise ValueError("frozen trajectory must be finite with shape (41, 7)")
    if baseline["nominal_trajectory"].get("frozen") is not True:
        raise ValueError("nominal trajectory is not marked frozen")
    return baseline, trajectory


def generate_trials(seed: int) -> list[dict[str, Any]]:
    """Generate the complete deterministic pilot design from one fixed seed."""
    rng = np.random.default_rng(seed)
    trials: list[dict[str, Any]] = []
    global_index = 0
    for magnitude_mm in MAGNITUDES_MM:
        count = 1 if magnitude_mm == 0.0 else NONZERO_TRIAL_COUNT
        for magnitude_trial_index in range(count):
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            radius_m = magnitude_mm / 1000.0
            dx = float(radius_m * math.cos(theta))
            dy = float(radius_m * math.sin(theta))
            trials.append(
                {
                    "trial_index": global_index,
                    "magnitude_trial_index": magnitude_trial_index,
                    "random_seed": seed,
                    "perturbation_magnitude_mm": magnitude_mm,
                    "sampled_theta_rad": theta,
                    "dx_m": dx,
                    "dy_m": dy,
                    "dz_m": 0.0,
                }
            )
            global_index += 1
    return trials


def sphere_box_clearance_matrix(
    spheres: torch.Tensor, center: list[float], box_size: list[float]
) -> torch.Tensor:
    """Compute signed clearances from every collision sphere to one AABB."""
    center_t = torch.tensor(center, device=spheres.device, dtype=spheres.dtype)
    half_t = torch.tensor(box_size, device=spheres.device, dtype=spheres.dtype) / 2.0
    delta = torch.abs(spheres[..., :3] - center_t) - half_t
    outside = torch.linalg.vector_norm(torch.clamp(delta, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(delta, dim=-1).values, max=0.0)
    return outside + inside - spheres[..., 3]


def load_trajectory_spheres(trajectory: np.ndarray, joint_names: list[str]) -> torch.Tensor:
    """Load only cuRobo kinematics and compute spheres; no planner is constructed."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    config = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    robot = Kinematics(config)
    if list(robot.joint_names) != joint_names:
        raise ValueError(f"Franka active joint names differ: {robot.joint_names}")
    q = torch.as_tensor(trajectory, device="cuda", dtype=torch.float32)
    state = robot.compute_kinematics(JointState.from_position(q, joint_names=joint_names))
    spheres = state.robot_spheres.squeeze(1)
    if spheres.ndim != 3 or spheres.shape[0] != 41 or spheres.shape[-1] != 4:
        raise RuntimeError(f"unexpected robot sphere shape: {spheres.shape}")
    return spheres


def evaluate_trials(
    design: list[dict[str, Any]], baseline: dict[str, Any], spheres: torch.Tensor
) -> list[dict[str, Any]]:
    """Evaluate every position against the same frozen trajectory spheres."""
    nominal = np.asarray(baseline["obstacle"]["nominal_position_m"], dtype=np.float64)
    box_size = baseline["obstacle"]["size_m"]
    results: list[dict[str, Any]] = []
    for trial in design:
        position = nominal + np.asarray([trial["dx_m"], trial["dy_m"], 0.0])
        matrix = sphere_box_clearance_matrix(spheres, position.tolist(), box_size)
        waypoint_clearance = torch.min(matrix, dim=-1).values
        minimum, minimum_index = torch.min(waypoint_clearance, dim=0)
        result = {
            **trial,
            "actual_obstacle_position_x_m": float(position[0]),
            "actual_obstacle_position_y_m": float(position[1]),
            "actual_obstacle_position_z_m": float(position[2]),
            "minimum_clearance_m": float(minimum.item()),
            "minimum_clearance_trajectory_index": int(minimum_index.item()),
            "collision_detected": bool(minimum.item() <= 0.0),
            "trajectory_frozen": True,
            "replanned": False,
            "status": "completed",
            "error": None,
        }
        results.append(result)
    return results


def summarize(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate empirical collision counts by pilot magnitude."""
    summary: list[dict[str, Any]] = []
    for magnitude in MAGNITUDES_MM:
        group = [item for item in results if item["perturbation_magnitude_mm"] == magnitude]
        collisions = sum(item["collision_detected"] for item in group)
        collision_indices = [
            item["minimum_clearance_trajectory_index"]
            for item in group
            if item["collision_detected"]
        ]
        summary.append(
            {
                "perturbation_magnitude_mm": magnitude,
                "number_of_trials": len(group),
                "number_of_collisions": collisions,
                "empirical_collision_rate": collisions / len(group),
                "collision_minimum_indices": collision_indices,
            }
        )
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    """Write JSON-compatible records to CSV."""
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, separators=(",", ":")) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def main() -> None:
    """Run and save the deterministic small pilot."""
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cuRobo kinematics")
    baseline, trajectory = load_baseline(args.baseline.resolve())
    design = generate_trials(RANDOM_SEED)
    if design != generate_trials(RANDOM_SEED):
        raise RuntimeError("fixed-seed perturbation generation is not exactly reproducible")
    spheres = load_trajectory_spheres(trajectory, baseline["joint_names"])
    results = evaluate_trials(design, baseline, spheres)
    nominal_result = results[0]
    expected_nominal = float(baseline["curobo_validation"]["minimum_clearance_m"])
    if not math.isclose(nominal_result["minimum_clearance_m"], expected_nominal, abs_tol=1e-7):
        raise RuntimeError(
            "recomputed nominal clearance differs from frozen baseline: "
            f"{nominal_result['minimum_clearance_m']} vs {expected_nominal}"
        )
    summary = summarize(results)
    payload = {
        "schema_version": 1,
        "status": "completed",
        "error": None,
        "experiment_id": "formal_baseline_v1_random_xy_pilot",
        "baseline_id": baseline["baseline_id"],
        "source_candidate": baseline["source_candidate"],
        "baseline_file": "configs/experiments/formal_baseline_v1.json",
        "trajectory_frozen": True,
        "replanned": False,
        "random_seed": RANDOM_SEED,
        "random_generator": "numpy.random.Generator(PCG64)",
        "random_design_reproducibility_check": "PASS_EXACT",
        "distribution": "theta uniform on [0, 2*pi); fixed XY radius; dz=0",
        "pilot_magnitudes_mm": MAGNITUDES_MM,
        "trials_per_nonzero_magnitude": NONZERO_TRIAL_COUNT,
        "zero_magnitude_trial_count": 1,
        "total_trial_count": len(results),
        "collision_validator": "cuRobo Franka collision spheres versus analytic AABB signed clearance",
        "collision_criterion": "minimum signed clearance <= 0.0 m",
        "isaac_sim_random_trial_validation_run": False,
        "interpretation_limit": (
            "Empirical collision rate under this synthetic fixed-radius random-direction distribution; "
            "not a real-robot collision probability."
        ),
        "summary": summary,
        "trials": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "experiment.json"
    trials_path = args.output_dir / "trials.csv"
    summary_path = args.output_dir / "summary.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
    write_csv(trials_path, results)
    write_csv(summary_path, summary)
    print("RANDOM_XY_REPRODUCIBILITY_PASS_EXACT")
    for item in summary:
        print("PILOT_SUMMARY " + json.dumps(item, sort_keys=True))
    print(f"PILOT_EXPERIMENT_JSON {json_path}")
    print(f"PILOT_TRIALS_CSV {trials_path}")
    print(f"PILOT_SUMMARY_CSV {summary_path}")
    print("RANDOM_ENVIRONMENT_PILOT_PASS")


if __name__ == "__main__":
    main()

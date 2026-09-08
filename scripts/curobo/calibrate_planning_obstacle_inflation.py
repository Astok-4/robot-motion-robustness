#!/usr/bin/env python3
"""Calibrate isotropic planning-time box inflation on two frozen scenes.

The enlarged box is used only by MotionPlanner. Every successful trajectory is
then evaluated twice with cuRobo collision spheres: once against the enlarged
planning box and once against the original frozen nominal box.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from calibrate_multiscene_candidates import sphere_link_map
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import JointState


EXPECTED_SCENES = ["formal_baseline_v1", "multiscene_candidate_001"]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=project_dir / "configs/experiments/planning_obstacle_inflation_calibration_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results_calibration/planning_obstacle_inflation_v1",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def validate_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = load_json(config_path)
    if config.get("experiment_id") != "planning_obstacle_inflation_calibration_v1":
        raise ValueError("unexpected experiment config")
    if config.get("scene_ids") != EXPECTED_SCENES:
        raise ValueError("calibration scene IDs differ from the fixed two-scene design")
    if config.get("inflation_levels_mm") != [0, 2, 5, 10, 15, 20]:
        raise ValueError("inflation levels differ from the fixed calibration design")
    if config.get("planning_repetitions") != 2:
        raise ValueError("calibration requires exactly two planning repetitions")
    project_dir = config_path.resolve().parents[2]
    baseline_path = project_dir / config["baseline_set"]
    baseline = load_json(baseline_path)
    if (
        baseline.get("baseline_set_id") != "formal_baseline_set_v1"
        or baseline.get("baseline_set_status") != "FROZEN"
        or baseline.get("trajectory_frozen") is not True
    ):
        raise ValueError("calibration requires FROZEN formal_baseline_set_v1")
    scenes = {item["scene_id"]: item for item in baseline.get("scenes", [])}
    if not all(scene_id in scenes for scene_id in EXPECTED_SCENES):
        raise ValueError("required scenes are absent from frozen baseline set")
    for scene_id in EXPECTED_SCENES:
        scene = scenes[scene_id]
        if scene.get("trajectory_frozen") is not True:
            raise ValueError(f"{scene_id}: baseline trajectory is not frozen")
        if scene["obstacle"]["size_m"] != [0.1, 0.1, 0.1]:
            raise ValueError(f"{scene_id}: nominal box size is not 100 mm per axis")
        for key in ("start_joint_configuration", "goal_joint_configuration"):
            values = np.asarray(scene[key], dtype=np.float64)
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError(f"{scene_id}: invalid {key}")
    return config, baseline, baseline_path


def box_scene(center: list[float], size: list[float]) -> Scene:
    return Scene(
        cuboid=[Cuboid(name="calibration_box", dims=size, pose=[*center, 1.0, 0.0, 0.0, 0.0])]
    )


def trajectory_spheres(planner: MotionPlanner, trajectory: torch.Tensor) -> torch.Tensor:
    names = planner.kinematics.config.kinematics_config.joint_names
    state = planner.kinematics.compute_kinematics(JointState.from_position(trajectory, names))
    return state.robot_spheres.squeeze(1)


def clearance_details(
    spheres: torch.Tensor,
    center: list[float],
    box_size: list[float],
    link_map: dict[int, str],
) -> dict[str, Any]:
    center_t = torch.tensor(center, device=spheres.device, dtype=spheres.dtype)
    half_t = torch.tensor(box_size, device=spheres.device, dtype=spheres.dtype) / 2.0
    delta = torch.abs(spheres[..., :3] - center_t) - half_t
    outside = torch.linalg.vector_norm(torch.clamp(delta, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(delta, dim=-1).values, max=0.0)
    matrix = outside + inside - spheres[..., 3]
    waypoint, sphere_indices = torch.min(matrix, dim=-1)
    minimum, trajectory_index_t = torch.min(waypoint, dim=0)
    trajectory_index = int(trajectory_index_t.item())
    sphere_index = int(sphere_indices[trajectory_index].item())
    return {
        "minimum_clearance_m": float(minimum.item()),
        "minimum_clearance_trajectory_index": trajectory_index,
        "minimum_clearance_sphere_index": sphere_index,
        "minimum_clearance_link": link_map[sphere_index],
        "collision": bool(minimum.item() <= 0.0),
        "clearance_profile_m": [float(value) for value in waypoint.detach().cpu().tolist()],
    }


def trajectory_length(trajectory: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(trajectory[1:] - trajectory[:-1], dim=-1).sum().item())


def plan_once(
    planner: MotionPlanner,
    start_q: list[float],
    goal_q: list[float],
    max_attempts: int,
) -> tuple[bool, torch.Tensor | None, dict[str, Any]]:
    names = planner.joint_names
    start = JointState.from_position(torch.tensor([start_q], device="cuda"), names)
    goal = JointState.from_position(torch.tensor([goal_q], device="cuda"), names)
    started = time.perf_counter()
    try:
        result = planner.plan_cspace(goal, start, max_attempts=max_attempts)
        wall_time = time.perf_counter() - started
        success = bool(result is not None and result.success.any().item())
        timing = {
            "planning_reported_solve_time_s": float(result.solve_time) if result is not None else None,
            "planning_reported_total_time_s": float(result.total_time) if result is not None else None,
            "measured_wall_time_s": wall_time,
        }
        if not success:
            return False, None, {**timing, "failure_reason": "planner_returned_no_success"}
        active = planner.kinematics.get_active_js(result.get_interpolated_plan())
        trajectory = active.position.reshape(-1, 7).contiguous()
        if not torch.isfinite(trajectory).all():
            raise RuntimeError("planner returned non-finite trajectory")
        return True, trajectory, {**timing, "failure_reason": None}
    except Exception as error:  # preserve every failed condition instead of aborting the series
        return False, None, {
            "planning_reported_solve_time_s": None,
            "planning_reported_total_time_s": None,
            "measured_wall_time_s": time.perf_counter() - started,
            "failure_reason": f"{type(error).__name__}: {error}",
        }


def compare_runs(run1: dict[str, Any], run2: dict[str, Any]) -> dict[str, Any]:
    base = {
        "scene_id": run1["scene_id"],
        "condition_id": run1["condition_id"],
        "inflation_mm": run1["inflation_mm"],
        "run1_success": run1["planning_success"],
        "run2_success": run2["planning_success"],
        "success_consistent": run1["planning_success"] == run2["planning_success"],
    }
    if not run1["planning_success"] or not run2["planning_success"]:
        return {
            **base,
            "trajectory_shapes_equal": None,
            "max_absolute_joint_difference_rad": None,
            "trajectory_length_difference_rad": None,
            "nominal_clearance_difference_m": None,
            "exact_trajectory_match": None,
            "tolerance_reproducible": run1["planning_success"] == run2["planning_success"],
        }
    first = np.asarray(run1["trajectory_positions"], dtype=np.float64)
    second = np.asarray(run2["trajectory_positions"], dtype=np.float64)
    shapes_equal = first.shape == second.shape
    max_difference = float(np.max(np.abs(first - second))) if shapes_equal else None
    return {
        **base,
        "trajectory_shapes_equal": shapes_equal,
        "max_absolute_joint_difference_rad": max_difference,
        "trajectory_length_difference_rad": abs(
            run1["trajectory_length_rad"] - run2["trajectory_length_rad"]
        ),
        "nominal_clearance_difference_m": abs(
            run1["clearance_nominal_geometry"]["minimum_clearance_m"]
            - run2["clearance_nominal_geometry"]["minimum_clearance_m"]
        ),
        "exact_trajectory_match": bool(shapes_equal and np.array_equal(first, second)),
        "tolerance_reproducible": bool(
            shapes_equal
            and max_difference is not None
            and max_difference <= 1.0e-7
            and abs(run1["trajectory_length_rad"] - run2["trajectory_length_rad"]) <= 1.0e-7
            and abs(
                run1["clearance_nominal_geometry"]["minimum_clearance_m"]
                - run2["clearance_nominal_geometry"]["minimum_clearance_m"]
            )
            <= 1.0e-7
        ),
    }


def pointwise_comparison(canonical: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    row = {
        "scene_id": canonical["scene_id"],
        "condition_id": canonical["condition_id"],
        "inflation_mm": canonical["inflation_mm"],
        "planning_success": canonical["planning_success"],
        "reference_condition_id": reference["condition_id"],
    }
    if not canonical["planning_success"] or not reference["planning_success"]:
        return {**row, "comparison_status": "not_applicable_planning_failure"}
    current = np.asarray(canonical["trajectory_positions"], dtype=np.float64)
    baseline = np.asarray(reference["trajectory_positions"], dtype=np.float64)
    if current.shape != baseline.shape:
        return {
            **row,
            "comparison_status": "direct_pointwise_comparison_not_applicable",
            "trajectory_shape": list(current.shape),
            "reference_trajectory_shape": list(baseline.shape),
        }
    absolute = np.abs(current - baseline)
    return {
        **row,
        "comparison_status": "pointwise_comparison_available",
        "trajectory_shape": list(current.shape),
        "reference_trajectory_shape": list(baseline.shape),
        "maximum_absolute_joint_deviation_rad": float(np.max(absolute)),
        "mean_absolute_joint_deviation_rad": float(np.mean(absolute)),
        "trajectory_length_difference_rad": canonical["trajectory_length_rad"]
        - reference["trajectory_length_rad"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for planning obstacle inflation calibration")
    config, baseline, baseline_path = validate_inputs(args.config)
    planner_cfg = config["planner"]
    scenes = {item["scene_id"]: item for item in baseline["scenes"]}
    initial_scene = scenes[EXPECTED_SCENES[0]]
    torch.manual_seed(int(planner_cfg["random_seed"]))
    planner_config = MotionPlannerCfg.create(
        robot=planner_cfg["robot_config"],
        scene_model=box_scene(
            initial_scene["obstacle"]["nominal_position_m"],
            initial_scene["obstacle"]["size_m"],
        ),
        random_seed=int(planner_cfg["random_seed"]),
        optimizer_collision_activation_distance=float(
            planner_cfg["optimizer_collision_activation_distance_m"]
        ),
    )
    planner = MotionPlanner(planner_config)
    planner.warmup(
        enable_graph=True, num_warmup_iterations=int(planner_cfg["warmup_iterations"])
    )
    link_map = sphere_link_map(planner)

    runs: list[dict[str, Any]] = []
    for scene_id in EXPECTED_SCENES:
        scene = scenes[scene_id]
        nominal_size = [float(value) for value in scene["obstacle"]["size_m"]]
        center = [float(value) for value in scene["obstacle"]["nominal_position_m"]]
        for inflation_mm in config["inflation_levels_mm"]:
            delta_m = float(inflation_mm) / 1000.0
            planning_size = [value + 2.0 * delta_m for value in nominal_size]
            condition_id = f"{scene_id}__inflation_{int(inflation_mm):03d}mm"
            planner.clear_scene_cache()
            planner.update_world(box_scene(center, planning_size))
            for run_index in range(1, config["planning_repetitions"] + 1):
                planner.reset_seed()
                torch.manual_seed(int(planner_cfg["random_seed"]))
                success, trajectory, planning = plan_once(
                    planner,
                    scene["start_joint_configuration"],
                    scene["goal_joint_configuration"],
                    int(planner_cfg["max_attempts"]),
                )
                record: dict[str, Any] = {
                    "scene_id": scene_id,
                    "condition_id": condition_id,
                    "run_index": run_index,
                    "canonical_run": run_index == 1,
                    "inflation_mm": inflation_mm,
                    "inflation_per_face_m": delta_m,
                    "nominal_obstacle_center_m": center,
                    "nominal_obstacle_size_m": nominal_size,
                    "planning_obstacle_size_m": planning_size,
                    "planning_success": success,
                    **planning,
                }
                if success and trajectory is not None:
                    spheres = trajectory_spheres(planner, trajectory)
                    record.update(
                        {
                            "trajectory_shape": list(trajectory.shape),
                            "trajectory_positions": trajectory.detach().cpu().tolist(),
                            "trajectory_length_rad": trajectory_length(trajectory),
                            "clearance_planning_geometry": clearance_details(
                                spheres, center, planning_size, link_map
                            ),
                            "clearance_nominal_geometry": clearance_details(
                                spheres, center, nominal_size, link_map
                            ),
                        }
                    )
                else:
                    record.update(
                        {
                            "trajectory_shape": None,
                            "trajectory_positions": None,
                            "trajectory_length_rad": None,
                            "clearance_planning_geometry": None,
                            "clearance_nominal_geometry": None,
                        }
                    )
                runs.append(record)
                nominal_clearance = (
                    record["clearance_nominal_geometry"]["minimum_clearance_m"]
                    if success
                    else None
                )
                print(
                    "INFLATION_PLANNING_RUN "
                    + json.dumps(
                        {
                            "condition_id": condition_id,
                            "run_index": run_index,
                            "success": success,
                            "trajectory_shape": record["trajectory_shape"],
                            "nominal_clearance_m": nominal_clearance,
                        },
                        sort_keys=True,
                    )
                )

    grouped = {
        (scene_id, inflation): [
            row
            for row in runs
            if row["scene_id"] == scene_id and row["inflation_mm"] == inflation
        ]
        for scene_id in EXPECTED_SCENES
        for inflation in config["inflation_levels_mm"]
    }
    reproducibility = [compare_runs(group[0], group[1]) for group in grouped.values()]
    canonical = [group[0] for group in grouped.values()]
    comparisons: list[dict[str, Any]] = []
    for scene_id in EXPECTED_SCENES:
        scene_rows = [row for row in canonical if row["scene_id"] == scene_id]
        reference = next(row for row in scene_rows if row["inflation_mm"] == 0)
        comparisons.extend(pointwise_comparison(row, reference) for row in scene_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "trajectories.json").open("w", encoding="utf-8") as stream:
        json.dump({"runs": runs}, stream, indent=2, allow_nan=False)
    condition_fields = [
        "scene_id", "condition_id", "run_index", "canonical_run", "inflation_mm",
        "inflation_per_face_m", "nominal_obstacle_center_m", "nominal_obstacle_size_m",
        "planning_obstacle_size_m", "planning_success", "failure_reason",
        "trajectory_shape", "trajectory_length_rad", "planning_reported_solve_time_s",
        "planning_reported_total_time_s", "measured_wall_time_s",
    ]
    write_csv(args.output_dir / "conditions.csv", runs, condition_fields)
    clearance_rows: list[dict[str, Any]] = []
    for row in canonical:
        planning_clearance = row["clearance_planning_geometry"] or {}
        nominal_clearance = row["clearance_nominal_geometry"] or {}
        clearance_rows.append(
            {
                **{field: row.get(field) for field in condition_fields},
                "clearance_planning_geometry_m": planning_clearance.get("minimum_clearance_m"),
                "planning_geometry_minimum_index": planning_clearance.get(
                    "minimum_clearance_trajectory_index"
                ),
                "planning_geometry_bottleneck_sphere": planning_clearance.get(
                    "minimum_clearance_sphere_index"
                ),
                "planning_geometry_bottleneck_link": planning_clearance.get(
                    "minimum_clearance_link"
                ),
                "clearance_nominal_geometry_m": nominal_clearance.get("minimum_clearance_m"),
                "nominal_geometry_minimum_index": nominal_clearance.get(
                    "minimum_clearance_trajectory_index"
                ),
                "nominal_geometry_bottleneck_sphere": nominal_clearance.get(
                    "minimum_clearance_sphere_index"
                ),
                "nominal_geometry_bottleneck_link": nominal_clearance.get(
                    "minimum_clearance_link"
                ),
            }
        )
    clearance_fields = condition_fields + [
        "clearance_planning_geometry_m", "planning_geometry_minimum_index",
        "planning_geometry_bottleneck_sphere", "planning_geometry_bottleneck_link",
        "clearance_nominal_geometry_m", "nominal_geometry_minimum_index",
        "nominal_geometry_bottleneck_sphere", "nominal_geometry_bottleneck_link",
    ]
    write_csv(args.output_dir / "clearance_summary.csv", clearance_rows, clearance_fields)
    reproducibility_fields = list(reproducibility[0])
    write_csv(args.output_dir / "reproducibility.csv", reproducibility, reproducibility_fields)
    comparison_fields = sorted({key for row in comparisons for key in row})
    write_csv(args.output_dir / "trajectory_comparison.csv", comparisons, comparison_fields)

    curobo_commit = subprocess.run(
        ["git", "-C", str(args.config.resolve().parents[2] / "curobo"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    experiment = {
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "planning_conditions_attempted": len(canonical),
        "planning_runs_attempted": len(runs),
        "scene_ids": EXPECTED_SCENES,
        "inflation_levels_mm": config["inflation_levels_mm"],
        "inflation_semantics": config["inflation_semantics"],
        "planning_geometry": config["planning_geometry"],
        "evaluation_geometry": config["evaluation_geometry"],
        "canonical_run_selection": config["canonical_run_selection"],
        "new_robustness_or_isaac_sim_run": False,
        "source_hashes_sha256": {
            str(args.config.relative_to(args.config.resolve().parents[2])): sha256(args.config),
            str(baseline_path.relative_to(args.config.resolve().parents[2])): sha256(baseline_path),
        },
        "software": {"curobo_commit": curobo_commit},
        "successful_canonical_conditions": sum(row["planning_success"] for row in canonical),
        "failed_canonical_conditions": sum(not row["planning_success"] for row in canonical),
        "reproducibility_all_tolerance_pass": all(
            row["tolerance_reproducible"] for row in reproducibility
        ),
        "notes": [
            "Planning times are recorded for audit only; two repetitions do not support a planning-cost claim.",
            "Canonical trajectories are always run 1, never selected by clearance or timing.",
        ],
    }
    with (args.output_dir / "experiment.json").open("w", encoding="utf-8") as stream:
        json.dump(experiment, stream, indent=2, allow_nan=False)
    print(
        "PLANNING_OBSTACLE_INFLATION_CALIBRATION_PASS "
        + json.dumps(
            {
                "conditions": len(canonical),
                "runs": len(runs),
                "successful": experiment["successful_canonical_conditions"],
                "reproducibility": experiment["reproducibility_all_tolerance_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

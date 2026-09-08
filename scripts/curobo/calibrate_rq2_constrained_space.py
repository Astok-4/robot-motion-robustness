#!/usr/bin/env python3
"""Calibrate a deterministic two-box gate for the RQ2 constrained-space study."""

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


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/rq2_constrained_space_calibration_v1.json")
    parser.add_argument("--output-dir", type=Path, default=root / "results_calibration/rq2_constrained_space_v1")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def gate_boxes(config: dict[str, Any], width_mm: int, inflation_mm: int) -> list[dict[str, Any]]:
    gate = config["gate"]
    half_gap = width_mm / 2000.0
    inflation = inflation_mm / 1000.0
    inner_left = gate["opening_center_m"] - half_gap
    inner_right = gate["opening_center_m"] + half_gap
    span = gate["box_outer_span_x_m"]
    nominal_dims = [span, gate["box_depth_y_m"], gate["box_height_z_m"]]
    planning_dims = [value + 2.0 * inflation for value in nominal_dims]
    return [
        {
            "name": "corridor_left_box",
            "center_m": [inner_left - span / 2.0, gate["plane_position_m"], gate["box_center_z_m"]],
            "nominal_size_m": nominal_dims,
            "planning_size_m": planning_dims,
        },
        {
            "name": "corridor_right_box",
            "center_m": [inner_right + span / 2.0, gate["plane_position_m"], gate["box_center_z_m"]],
            "nominal_size_m": nominal_dims,
            "planning_size_m": planning_dims,
        },
    ]


def make_scene(boxes: list[dict[str, Any]], planning: bool) -> Scene:
    size_key = "planning_size_m" if planning else "nominal_size_m"
    return Scene(cuboid=[Cuboid(name=box["name"], dims=box[size_key], pose=[*box["center_m"], 1.0, 0.0, 0.0, 0.0]) for box in boxes])


def trajectory_state(planner: MotionPlanner, trajectory: torch.Tensor):
    names = planner.kinematics.config.kinematics_config.joint_names
    return planner.kinematics.compute_kinematics(JointState.from_position(trajectory, names))


def trajectory_length(trajectory: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(trajectory[1:] - trajectory[:-1], dim=-1).sum().item())


def clearance_details(spheres: torch.Tensor, boxes: list[dict[str, Any]], planning: bool, link_map: dict[int, str]) -> dict[str, Any]:
    size_key = "planning_size_m" if planning else "nominal_size_m"
    matrices = []
    for box in boxes:
        center = torch.tensor(box["center_m"], device=spheres.device, dtype=spheres.dtype)
        half = torch.tensor(box[size_key], device=spheres.device, dtype=spheres.dtype) / 2.0
        delta = torch.abs(spheres[..., :3] - center) - half
        outside = torch.linalg.vector_norm(torch.clamp(delta, min=0.0), dim=-1)
        inside = torch.clamp(torch.max(delta, dim=-1).values, max=0.0)
        matrices.append(outside + inside - spheres[..., 3])
    matrix = torch.stack(matrices, dim=-1)  # T, S, obstacles
    flat = matrix.reshape(matrix.shape[0], -1)
    per_point, flat_indices = torch.min(flat, dim=-1)
    minimum, trajectory_index_t = torch.min(per_point, dim=0)
    trajectory_index = int(trajectory_index_t.item())
    flat_index = int(flat_indices[trajectory_index].item())
    obstacle_index = flat_index % len(boxes)
    sphere_index = flat_index // len(boxes)
    return {
        "minimum_clearance_m": float(minimum.item()),
        "minimum_clearance_trajectory_index": trajectory_index,
        "minimum_clearance_sphere_index": sphere_index,
        "minimum_clearance_link": link_map[sphere_index],
        "minimum_clearance_obstacle": boxes[obstacle_index]["name"],
        "collision": bool(minimum.item() <= 0.0),
        "clearance_profile_m": [float(v) for v in per_point.detach().cpu().tolist()],
    }


def end_effector_positions(planner: MotionPlanner, state: Any) -> np.ndarray:
    frames = planner.kinematics.config.kinematics_config.tool_frames
    if not frames:
        raise RuntimeError("Franka tool frame is unavailable")
    return state.tool_poses.get_link_pose(frames[0]).position.detach().cpu().numpy()


def passage_audit(positions: np.ndarray, config: dict[str, Any], width_mm: int) -> dict[str, Any]:
    gate = config["gate"]
    plane = float(gate["plane_position_m"])
    low = float(gate["opening_center_m"] - width_mm / 2000.0)
    high = float(gate["opening_center_m"] + width_mm / 2000.0)
    z_low = float(gate["box_center_z_m"] - gate["box_height_z_m"] / 2.0)
    z_high = float(gate["box_center_z_m"] + gate["box_height_z_m"] / 2.0)
    candidates = []
    for index in range(len(positions) - 1):
        y0, y1 = float(positions[index, 1]), float(positions[index + 1, 1])
        if (y0 - plane) * (y1 - plane) <= 0.0 and not math.isclose(y0, y1, abs_tol=1e-12):
            alpha = (plane - y0) / (y1 - y0)
            crossing = positions[index] + alpha * (positions[index + 1] - positions[index])
            candidates.append((index, crossing))
    valid = [
        (index, point)
        for index, point in candidates
        if low <= float(point[0]) <= high and z_low <= float(point[2]) <= z_high
    ]
    chosen = valid[0] if valid else (candidates[0] if candidates else None)
    start_side = float(positions[0, 1] - plane)
    goal_side = float(positions[-1, 1] - plane)
    crosses_sides = start_side * goal_side < 0.0
    passage_used = bool(crosses_sides and valid)
    crossing_index = chosen[0] if chosen else None
    crossing_position = chosen[1].tolist() if chosen else None
    return {
        "passage_used": passage_used,
        "bypass_detected": bool(crosses_sides and not passage_used),
        "crosses_gate_plane": bool(crosses_sides),
        "crossing_index": crossing_index,
        "crossing_position_m": crossing_position,
        "opening_lower_x_m": low,
        "opening_upper_x_m": high,
        "gate_post_lower_z_m": z_low,
        "gate_post_upper_z_m": z_high,
        "start_gate_side_m": start_side,
        "goal_gate_side_m": goal_side,
    }


def plan_once(planner: MotionPlanner, start_q: list[float], goal_q: list[float], max_attempts: int):
    start = JointState.from_position(torch.tensor([start_q], device="cuda"), planner.joint_names)
    goal = JointState.from_position(torch.tensor([goal_q], device="cuda"), planner.joint_names)
    began = time.perf_counter()
    try:
        result = planner.plan_cspace(goal, start, max_attempts=max_attempts)
        wall = time.perf_counter() - began
        success = bool(result is not None and result.success.any().item())
        timing = {
            "planning_reported_solve_time_s": float(result.solve_time) if result is not None else None,
            "planning_reported_total_time_s": float(result.total_time) if result is not None else None,
            "measured_wall_time_s": wall,
        }
        if not success:
            return False, None, {**timing, "failure_reason": "planner_returned_no_success"}
        trajectory = planner.kinematics.get_active_js(result.get_interpolated_plan()).position.reshape(-1, 7).contiguous()
        if not torch.isfinite(trajectory).all():
            raise RuntimeError("non-finite trajectory")
        return True, trajectory, {**timing, "failure_reason": None}
    except Exception as error:
        return False, None, {
            "planning_reported_solve_time_s": None,
            "planning_reported_total_time_s": None,
            "measured_wall_time_s": time.perf_counter() - began,
            "failure_reason": f"{type(error).__name__}: {error}",
        }


def compare_runs(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    row = {
        "condition_id": first["condition_id"],
        "corridor_width_mm": first["corridor_width_mm"],
        "inflation_mm": first["inflation_mm"],
        "run1_success": first["planning_success"],
        "run2_success": second["planning_success"],
        "success_consistent": first["planning_success"] == second["planning_success"],
    }
    if not first["planning_success"] or not second["planning_success"]:
        return {**row, "trajectory_shapes_equal": None, "max_absolute_joint_difference_rad": None, "trajectory_length_difference_rad": None, "exact_trajectory_match": None, "tolerance_reproducible": row["success_consistent"]}
    one = np.asarray(first["trajectory_positions"], dtype=np.float64)
    two = np.asarray(second["trajectory_positions"], dtype=np.float64)
    same_shape = one.shape == two.shape
    maximum = float(np.max(np.abs(one - two))) if same_shape else None
    length_difference = abs(float(first["trajectory_length_rad"]) - float(second["trajectory_length_rad"]))
    return {
        **row,
        "trajectory_shapes_equal": same_shape,
        "max_absolute_joint_difference_rad": maximum,
        "trajectory_length_difference_rad": length_difference,
        "exact_trajectory_match": bool(same_shape and np.array_equal(one, two)),
        "tolerance_reproducible": bool(same_shape and maximum is not None and maximum <= 1e-7 and length_difference <= 1e-7),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    config = load_json(args.config)
    matrix_path = args.config.resolve().parents[2] / config["source_matrix"]
    matrix_hash_before = sha256(matrix_path)
    matrix = load_json(matrix_path)
    source = next(row for row in matrix["conditions"] if row["condition_id"] == config["source_condition_id"])
    start_q = [float(v) for v in source["start_configuration"]]
    goal_q = [float(v) for v in source["goal_configuration"]]
    planner_cfg = config["planner"]
    initial_boxes = gate_boxes(config, config["initial_corridor_widths_mm"][0], 0)
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=config["robot_config"], scene_model=make_scene(initial_boxes, True),
        random_seed=planner_cfg["random_seed"],
        optimizer_collision_activation_distance=planner_cfg["optimizer_collision_activation_distance_m"],
    ))
    planner.warmup(enable_graph=True, num_warmup_iterations=planner_cfg["warmup_iterations"])
    link_map = sphere_link_map(planner)

    runs: list[dict[str, Any]] = []
    tested_widths: list[int] = []

    def run_width(width_mm: int) -> None:
        tested_widths.append(width_mm)
        for inflation_mm in config["inflation_levels_mm"]:
            boxes = gate_boxes(config, width_mm, inflation_mm)
            condition_id = f"corridor_{width_mm:03d}mm__inflation_{inflation_mm:03d}mm"
            planner.clear_scene_cache()
            planner.update_world(make_scene(boxes, True))
            for run_index in range(1, config["planning_repetitions"] + 1):
                planner.reset_seed()
                torch.manual_seed(planner_cfg["random_seed"])
                success, trajectory, timing = plan_once(planner, start_q, goal_q, planner_cfg["max_attempts"])
                record: dict[str, Any] = {
                    "condition_id": condition_id, "corridor_width_mm": width_mm,
                    "inflation_mm": inflation_mm,
                    "effective_planning_gap_mm": width_mm - 2 * inflation_mm,
                    "trivially_blocked_geometry": width_mm - 2 * inflation_mm <= 0,
                    "run_index": run_index, "canonical_run": run_index == config["canonical_run"],
                    "planning_success": success, "failure_reason": timing["failure_reason"],
                    "planning_reported_solve_time_s": timing["planning_reported_solve_time_s"],
                    "planning_reported_total_time_s": timing["planning_reported_total_time_s"],
                    "measured_wall_time_s": timing["measured_wall_time_s"],
                    "boxes": boxes,
                }
                if success and trajectory is not None:
                    state = trajectory_state(planner, trajectory)
                    spheres = state.robot_spheres.squeeze(1)
                    record.update({
                        "trajectory_shape": list(trajectory.shape),
                        "trajectory_positions": trajectory.detach().cpu().tolist(),
                        "trajectory_length_rad": trajectory_length(trajectory),
                        "trajectory_finite": bool(torch.isfinite(trajectory).all()),
                        "planning_geometry_clearance": clearance_details(spheres, boxes, True, link_map),
                        "nominal_geometry_clearance": clearance_details(spheres, boxes, False, link_map),
                        "passage_audit": passage_audit(end_effector_positions(planner, state), config, width_mm),
                    })
                else:
                    record.update({"trajectory_shape": None, "trajectory_positions": None, "trajectory_length_rad": None, "trajectory_finite": None, "planning_geometry_clearance": None, "nominal_geometry_clearance": None, "passage_audit": None})
                runs.append(record)
                print("CONSTRAINED_PLANNING_RUN " + json.dumps({"condition_id": condition_id, "run": run_index, "success": success, "passage": record["passage_audit"]["passage_used"] if success else None, "nominal_clearance_m": record["nominal_geometry_clearance"]["minimum_clearance_m"] if success else None}, sort_keys=True), flush=True)

    for width in config["initial_corridor_widths_mm"]:
        run_width(width)

    def canonical_for(width: int) -> list[dict[str, Any]]:
        return [r for r in runs if r["corridor_width_mm"] == width and r["canonical_run"]]

    initial = [r for width in config["initial_corridor_widths_mm"] for r in canonical_for(width)]
    if all(r["planning_success"] and r["passage_audit"]["passage_used"] for r in initial):
        for width in config["narrow_extension_widths_mm"]:
            run_width(width)
            rows = canonical_for(width)
            if any((not r["planning_success"]) or (not r["passage_audit"]["passage_used"]) for r in rows):
                break
    elif not (canonical_for(220)[0]["planning_success"] and canonical_for(220)[0]["passage_audit"]["passage_used"]):
        for width in config["wide_extension_widths_mm"]:
            run_width(width)
            row = canonical_for(width)[0]
            if row["planning_success"] and row["passage_audit"]["passage_used"]:
                break

    canonical = [r for r in runs if r["canonical_run"]]
    reproducibility = []
    for row in canonical:
        pair = [r for r in runs if r["condition_id"] == row["condition_id"]]
        reproducibility.append(compare_runs(pair[0], pair[1]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trajectories.json").write_text(json.dumps({"runs": runs}, indent=2, allow_nan=False), encoding="utf-8")
    scene_geometry = {
        "corridor_definition": config["corridor_definition"], "gate": config["gate"],
        "tested_corridor_widths_mm": tested_widths, "start_configuration": start_q,
        "goal_configuration": goal_q, "obstacle_orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "geometry_by_width": {str(w): gate_boxes(config, w, 0) for w in tested_widths},
    }
    (args.output_dir / "scene_geometry.json").write_text(json.dumps(scene_geometry, indent=2), encoding="utf-8")
    common_fields = ["condition_id", "corridor_width_mm", "inflation_mm", "effective_planning_gap_mm", "trivially_blocked_geometry", "run_index", "canonical_run", "planning_success", "failure_reason", "trajectory_shape", "trajectory_length_rad", "trajectory_finite", "planning_reported_solve_time_s", "planning_reported_total_time_s", "measured_wall_time_s"]
    write_csv(args.output_dir / "planning_runs.csv", runs, common_fields)
    write_csv(args.output_dir / "conditions.csv", canonical, common_fields)
    write_csv(args.output_dir / "reproducibility.csv", reproducibility)
    passage_rows = [{"condition_id": r["condition_id"], "corridor_width_mm": r["corridor_width_mm"], "inflation_mm": r["inflation_mm"], **(r["passage_audit"] or {"passage_used": None, "bypass_detected": None, "crosses_gate_plane": None, "crossing_index": None, "crossing_position_m": None, "opening_lower_x_m": None, "opening_upper_x_m": None, "start_gate_side_m": None, "goal_gate_side_m": None})} for r in canonical]
    write_csv(args.output_dir / "passage_usage_audit.csv", passage_rows)
    cu_rows = []
    for r in canonical:
        clearance = r["nominal_geometry_clearance"] or {}
        cu_rows.append({"condition_id": r["condition_id"], "corridor_width_mm": r["corridor_width_mm"], "inflation_mm": r["inflation_mm"], "planning_success": r["planning_success"], "nominal_curobo_collision": clearance.get("collision"), "nominal_curobo_safe": (not clearance.get("collision")) if clearance else None, "c_curobo_m": clearance.get("minimum_clearance_m"), "minimum_index": clearance.get("minimum_clearance_trajectory_index"), "bottleneck_sphere": clearance.get("minimum_clearance_sphere_index"), "bottleneck_link": clearance.get("minimum_clearance_link"), "bottleneck_obstacle": clearance.get("minimum_clearance_obstacle")})
    write_csv(args.output_dir / "curobo_nominal_validation.csv", cu_rows)
    experiment = {
        "experiment_id": config["experiment_id"], "status": "planning_and_curobo_nominal_complete_physx_pending",
        "calibration_only": True, "formal_matrix_frozen": False,
        "tested_corridor_widths_mm": tested_widths, "inflation_levels_mm": config["inflation_levels_mm"],
        "conditions_attempted": len(canonical), "planning_attempts": len(runs),
        "canonical_run_selection": "run_1_without_cherry_picking",
        "timing_interpretation": "calibration_timing_observation_only",
        "robustness_experiment_run": False,
        "source_hashes_before": {str(matrix_path.relative_to(args.config.resolve().parents[2])): matrix_hash_before},
        "source_hashes_after": {str(matrix_path.relative_to(args.config.resolve().parents[2])): sha256(matrix_path)},
        "config_sha256": sha256(args.config),
        "curobo_commit": subprocess.run(["git", "-C", str(args.config.resolve().parents[2] / "curobo"), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
    }
    (args.output_dir / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    if matrix_hash_before != sha256(matrix_path):
        raise RuntimeError("frozen formal matrix changed during calibration")
    print("RQ2_CONSTRAINED_SPACE_PLANNING_CALIBRATION_COMPLETE " + json.dumps({"widths": tested_widths, "conditions": len(canonical), "runs": len(runs)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Offline gate-local geometry and strengthened passage audit for all 55 paths."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/curobo"))

from calibrate_multiscene_candidates import sphere_link_map  # noqa: E402
from calibrate_rq2_constrained_space import (  # noqa: E402
    end_effector_positions,
    gate_boxes,
    make_scene,
    trajectory_state,
)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402


GATE_LOCAL_HALF_THICKNESS_M = 0.04


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-results", type=Path, default=ROOT / "results_calibration/rq2_constrained_space_v1")
    parser.add_argument("--diagnostic-results", type=Path, default=ROOT / "results_calibration/rq2_constrained_space_diagnostic_v1")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/rq2_constrained_space_calibration_v1.json")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def nearest_surface(point: np.ndarray, center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, str]:
    low, high = center - size / 2.0, center + size / 2.0
    closest = np.clip(point, low, high)
    outside_axes = np.where((point < low) | (point > high))[0]
    labels = ["x", "y", "z"]
    if len(outside_axes):
        parts = []
        for axis in outside_axes:
            parts.append(f"{labels[axis]}_{'min' if point[axis] < low[axis] else 'max'}")
        return closest, "+".join(parts)
    distances = np.concatenate([point - low, high - point])
    face = int(np.argmin(distances))
    axis = face % 3
    closest = point.copy()
    if face < 3:
        closest[axis] = low[axis]
        return closest, f"{labels[axis]}_min"
    closest[axis] = high[axis]
    return closest, f"{labels[axis]}_max"


def clearance_matrix(spheres: torch.Tensor, boxes: list[dict[str, Any]]) -> torch.Tensor:
    matrices = []
    for box in boxes:
        center = torch.tensor(box["center_m"], device=spheres.device, dtype=spheres.dtype)
        half = torch.tensor(box["nominal_size_m"], device=spheres.device, dtype=spheres.dtype) / 2.0
        delta = torch.abs(spheres[..., :3] - center) - half
        outside = torch.linalg.vector_norm(torch.clamp(delta, min=0.0), dim=-1)
        inside = torch.clamp(torch.max(delta, dim=-1).values, max=0.0)
        matrices.append(outside + inside - spheres[..., 3])
    return torch.stack(matrices, dim=-1)


def clearance_record(matrix: torch.Tensor, spheres: torch.Tensor, boxes: list[dict[str, Any]], indices: list[int], link_map: dict[int, str]) -> dict[str, Any]:
    if not indices:
        raise RuntimeError("gate-local state set is empty")
    selected = matrix[indices]
    flat_index = int(torch.argmin(selected).item())
    sphere_count, obstacle_count = selected.shape[1], selected.shape[2]
    local_t = flat_index // (sphere_count * obstacle_count)
    remainder = flat_index % (sphere_count * obstacle_count)
    sphere_index = remainder // obstacle_count
    obstacle_index = remainder % obstacle_count
    trajectory_index = indices[local_t]
    value = float(selected[local_t, sphere_index, obstacle_index].item())
    sphere = spheres[trajectory_index, sphere_index].detach().cpu().numpy()
    box = boxes[obstacle_index]
    closest, surface = nearest_surface(sphere[:3], np.asarray(box["center_m"]), np.asarray(box["nominal_size_m"]))
    return {
        "clearance_m": value, "trajectory_index": trajectory_index,
        "sphere_index": sphere_index, "link": link_map[sphere_index],
        "obstacle": box["name"], "obstacle_surface": surface,
        "sphere_center_m": sphere[:3].tolist(), "sphere_radius_m": float(sphere[3]),
        "obstacle_nearest_surface_point_m": closest.tolist(),
    }


def strengthened_passage(positions: np.ndarray, config: dict[str, Any], width_mm: int) -> dict[str, Any]:
    gate = config["gate"]
    plane = float(gate["plane_position_m"])
    center = np.asarray([gate["opening_center_m"], plane, gate["box_center_z_m"]], dtype=np.float64)
    x_low = gate["opening_center_m"] - width_mm / 2000.0
    x_high = gate["opening_center_m"] + width_mm / 2000.0
    z_low = gate["box_center_z_m"] - gate["box_height_z_m"] / 2.0
    z_high = gate["box_center_z_m"] + gate["box_height_z_m"] / 2.0
    crossings = []
    for index in range(len(positions) - 1):
        y0, y1 = positions[index, 1], positions[index + 1, 1]
        if (y0 - plane) * (y1 - plane) <= 0 and not math.isclose(float(y0), float(y1), abs_tol=1e-12):
            alpha = (plane - y0) / (y1 - y0)
            crossings.append((index, positions[index] + alpha * (positions[index + 1] - positions[index])))
    crossing = crossings[0] if crossings else None
    if crossing is None:
        category = "bypass_unknown_side"
        inside = False
        point = None
    else:
        point = crossing[1]
        inside_x = x_low <= point[0] <= x_high
        inside_z = z_low <= point[2] <= z_high
        inside = bool(inside_x and inside_z)
        if inside:
            category = "passage"
        else:
            parts = []
            if point[2] > z_high:
                parts.append("upper")
            elif point[2] < z_low:
                parts.append("lower")
            if point[0] < x_low:
                parts.append("lateral_left")
            elif point[0] > x_high:
                parts.append("lateral_right")
            category = "_and_".join(parts) + "_bypass" if parts else "bypass_unknown_side"
    return {
        "passage_used": inside, "bypass_detected": not inside,
        "gate_plane_crossing_exists": crossing is not None,
        "crossing_index": crossing[0] if crossing else None,
        "crossing_position_m": point.tolist() if point is not None else None,
        "crossing_inside_nominal_opening": inside,
        "crossing_distance_to_gate_center_m": float(np.linalg.norm(point - center)) if point is not None else None,
        "route_category": category,
        "opening_x_min_m": x_low, "opening_x_max_m": x_high,
        "active_gate_z_min_m": z_low, "active_gate_z_max_m": z_high,
    }


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    old = json.loads((args.original_results / "trajectories.json").read_text(encoding="utf-8"))["runs"]
    new = json.loads((args.diagnostic_results / "wide_extension_trajectories.json").read_text(encoding="utf-8"))["runs"]
    canonical = [row for row in [*old, *new] if row["canonical_run"]]
    if len(canonical) != 55 or len({row["condition_id"] for row in canonical}) != 55:
        raise ValueError("expected 55 unique canonical trajectories")
    planner_cfg = config["planner"]
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=config["robot_config"], scene_model=make_scene(gate_boxes(config, 100, 0), False),
        random_seed=planner_cfg["random_seed"], optimizer_collision_activation_distance=planner_cfg["optimizer_collision_activation_distance_m"],
    ))
    link_map = sphere_link_map(planner)
    passage_rows, clearance_rows, nominal_rows = [], [], []
    for row in canonical:
        if not row["planning_success"]:
            continue
        width = int(row["corridor_width_mm"])
        trajectory = torch.tensor(row["trajectory_positions"], device="cuda")
        state = trajectory_state(planner, trajectory)
        spheres = state.robot_spheres.squeeze(1)
        positions = end_effector_positions(planner, state)
        boxes = gate_boxes(config, width, 0)
        matrix = clearance_matrix(spheres, boxes)
        all_indices = list(range(len(positions)))
        local_indices = [index for index, point in enumerate(positions) if abs(float(point[1]) - config["gate"]["plane_position_m"]) <= GATE_LOCAL_HALF_THICKNESS_M]
        global_record = clearance_record(matrix, spheres, boxes, all_indices, link_map)
        local_record = clearance_record(matrix, spheres, boxes, local_indices, link_map)
        route = strengthened_passage(positions, config, width)
        global_inside = global_record["trajectory_index"] in local_indices
        common = {"condition_id": row["condition_id"], "corridor_width_mm": width, "inflation_mm": int(row["inflation_mm"]), "effective_planning_gap_mm": int(row["effective_planning_gap_mm"])}
        passage_rows.append({**common, **route, "gate_local_state_indices": local_indices, "gate_local_slab_half_thickness_m": GATE_LOCAL_HALF_THICKNESS_M})
        clearance_rows.append({
            **common, "passage_used": route["passage_used"],
            "global_curobo_clearance_mm": global_record["clearance_m"] * 1000.0,
            "global_curobo_minimum_index": global_record["trajectory_index"],
            "global_curobo_link": global_record["link"], "global_curobo_obstacle": global_record["obstacle"],
            "global_bottleneck_inside_gate_region": global_inside,
            "gate_local_curobo_clearance_mm": local_record["clearance_m"] * 1000.0,
            "gate_local_minimum_index": local_record["trajectory_index"],
            "gate_local_sphere_index": local_record["sphere_index"], "gate_local_link": local_record["link"],
            "gate_local_obstacle": local_record["obstacle"], "gate_local_obstacle_surface": local_record["obstacle_surface"],
            "gate_local_sphere_center_m": local_record["sphere_center_m"], "gate_local_obstacle_nearest_surface_point_m": local_record["obstacle_nearest_surface_point_m"],
            "gate_local_measure_type": "curobo_collision_sphere_to_original_box_aabb_signed_clearance",
        })
        nominal_rows.append({
            **common, "planning_success": True, "nominal_curobo_collision": global_record["clearance_m"] <= 0.0,
            "nominal_curobo_safe": global_record["clearance_m"] > 0.0,
            "global_curobo_clearance_m": global_record["clearance_m"], "global_minimum_index": global_record["trajectory_index"],
            "global_bottleneck_sphere": global_record["sphere_index"], "global_bottleneck_link": global_record["link"], "global_bottleneck_obstacle": global_record["obstacle"],
        })
        print("GATE_LOCAL_AUDIT " + json.dumps({"condition_id": row["condition_id"], "route": route["route_category"], "global_mm": global_record["clearance_m"] * 1000, "local_mm": local_record["clearance_m"] * 1000, "global_inside": global_inside}, sort_keys=True), flush=True)
    args.diagnostic_results.mkdir(parents=True, exist_ok=True)
    write_csv(args.diagnostic_results / "passage_usage_audit.csv", passage_rows)
    write_csv(args.diagnostic_results / "gate_local_clearance.csv", clearance_rows)
    write_csv(args.diagnostic_results / "nominal_curobo_validation.csv", nominal_rows)
    metadata = {
        "gate_region": {
            "center_m": [config["gate"]["opening_center_m"], config["gate"]["plane_position_m"], config["gate"]["box_center_z_m"]],
            "plane": f"y={config['gate']['plane_position_m']} m", "normal_world": [0.0, 1.0, 0.0],
            "opening_bounds_rule": "x=center_x +/- corridor_width/2; active z span equals nominal post z span",
            "local_evaluation_rule": f"trajectory states whose panda_hand y is within +/- {GATE_LOCAL_HALF_THICKNESS_M} m of the fixed gate plane",
            "slab_total_thickness_m": 2 * GATE_LOCAL_HALF_THICKNESS_M,
            "condition_specific_tuning": False,
        },
        "conditions_audited": 55, "passage_definition": "interpolated panda_hand crossing lies inside nominal x opening and active nominal z span",
        "gate_local_curobo_measure": "signed cuRobo collision-sphere to original two-box AABB clearance over fixed gate-local trajectory states",
    }
    (args.diagnostic_results / "gate_region_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

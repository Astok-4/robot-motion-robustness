#!/usr/bin/env python3
"""Discover geometrically diverse Franka single-box nominal scenes.

This search keeps the established calibration start and goal fixed.  Candidate
boxes are generated around internal points of the obstacle-free reference path,
then every retained scene is independently replanned and measured with cuRobo's
Franka collision spheres.  It does not freeze or select a new formal baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from calibrate_nominal_scenarios import (
    ARM_JOINT_NAMES,
    BOX_DIMS,
    FAR_BOX_CENTER,
    GOAL_Q,
    START_Q,
    clearance_profile,
    make_scene,
    plan_trajectory,
    sphere_box_clearance_matrix,
    trajectory_length,
    trajectory_spheres,
)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import ContentPath, JointState


AXES = {
    "+X": np.asarray([1.0, 0.0, 0.0]),
    "-X": np.asarray([-1.0, 0.0, 0.0]),
    "+Y": np.asarray([0.0, 1.0, 0.0]),
    "-Y": np.asarray([0.0, -1.0, 0.0]),
    "+Z": np.asarray([0.0, 0.0, 1.0]),
    "-Z": np.asarray([0.0, 0.0, -1.0]),
}
TARGET_CLEARANCES_M = [0.008, 0.012, 0.018]
MIN_FRACTION = 0.20
MAX_FRACTION = 0.80
MIN_ENDPOINT_CLEARANCE_M = 0.03
MIN_NOMINAL_CLEARANCE_M = 0.004
MAX_NOMINAL_CLEARANCE_M = 0.025
BASELINE_V1_DANGER = np.asarray([0.0, 0.0, -1.0])


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results_calibration" / "multiscene_candidates",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=project_dir / "configs" / "experiments" / "formal_baseline_v1.json",
    )
    parser.add_argument("--max-evaluations", type=int, default=48)
    parser.add_argument("--shortlist-count", type=int, default=5)
    parser.add_argument("--gui", metavar="CANDIDATE_ID")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sphere_link_map(planner: MotionPlanner) -> dict[int, str]:
    """Build the sphere-to-link map from cuRobo's active kinematics config."""
    kinematics = planner.kinematics.config.kinematics_config
    mapping: dict[int, str] = {}
    if kinematics.link_name_to_idx_map is None:
        raise RuntimeError("cuRobo kinematics config has no link-name map")
    for link_name in kinematics.link_name_to_idx_map:
        indices = kinematics.get_sphere_index_from_link_name(link_name)
        for sphere_index in indices.detach().cpu().tolist():
            index = int(sphere_index)
            if index in mapping:
                raise RuntimeError(f"collision sphere {index} maps to more than one link")
            mapping[index] = link_name
    expected = int(kinematics.link_spheres.shape[-2])
    if len(mapping) != expected:
        raise RuntimeError(f"sphere/link mapping incomplete: {len(mapping)} of {expected}")
    return mapping


def local_geometry(
    spheres: torch.Tensor,
    center: list[float],
    trajectory_index: int,
    sphere_index: int,
    link_map: dict[int, str],
) -> dict[str, Any]:
    sphere = spheres[trajectory_index, sphere_index].detach().cpu().numpy().astype(float)
    center_array = np.asarray(center, dtype=float)
    half = np.asarray(BOX_DIMS, dtype=float) / 2.0
    nearest = np.clip(sphere[:3], center_array - half, center_array + half)
    surface_to_sphere = sphere[:3] - nearest
    distance = float(np.linalg.norm(surface_to_sphere))
    if distance <= 1.0e-12:
        raise RuntimeError("cannot define local danger direction for sphere center inside box")
    direction = surface_to_sphere / distance
    return {
        "minimum_sphere_index": sphere_index,
        "minimum_sphere_link": link_map[sphere_index],
        "minimum_sphere_world_position_m": sphere[:3].tolist(),
        "minimum_sphere_radius_m": float(sphere[3]),
        "obstacle_nearest_surface_point_m": nearest.tolist(),
        "surface_to_sphere_vector_m": surface_to_sphere.tolist(),
        "surface_to_sphere_center_distance_m": distance,
        "danger_direction": direction.tolist(),
        "danger_direction_x": float(direction[0]),
        "danger_direction_y": float(direction[1]),
        "danger_direction_z": float(direction[2]),
        "angle_from_formal_baseline_v1_deg": float(
            math.degrees(math.acos(float(np.clip(np.dot(direction, BASELINE_V1_DANGER), -1.0, 1.0))))
        ),
    }


def dominant_direction(direction: list[float]) -> str:
    values = np.asarray(direction)
    index = int(np.argmax(np.abs(values)))
    return ("+" if values[index] >= 0.0 else "-") + "XYZ"[index]


def generate_centers(
    baseline_spheres: torch.Tensor,
    link_map: dict[int, str],
    baseline_v1_center: list[float],
) -> list[dict[str, Any]]:
    """Generate and round-robin diverse centers before expensive replanning."""
    step_count, sphere_count, _ = baseline_spheres.shape
    raw: list[dict[str, Any]] = []
    target_indices = range(
        int(math.ceil(MIN_FRACTION * (step_count - 1))),
        int(math.floor(MAX_FRACTION * (step_count - 1))) + 1,
        2,
    )
    for target_index in target_indices:
        for sphere_index in range(sphere_count):
            sphere = baseline_spheres[target_index, sphere_index]
            radius = float(sphere[3].item())
            if radius <= 0.0:
                continue
            sphere_position = sphere[:3].detach().cpu().numpy()
            for source_axis_name, axis in AXES.items():
                for target_clearance in TARGET_CLEARANCES_M:
                    center = sphere_position + axis * (radius + BOX_DIMS[0] / 2.0 + target_clearance)
                    if not (-0.30 <= center[0] <= 0.95 and -0.85 <= center[1] <= 0.85 and 0.10 <= center[2] <= 1.25):
                        continue
                    if np.linalg.norm(center - np.asarray(baseline_v1_center)) < 0.06:
                        continue
                    center_list = [float(value) for value in center]
                    profile, minimum_index, minimum_sphere = clearance_profile(baseline_spheres, center_list)
                    minimum = min(profile)
                    fraction = minimum_index / (step_count - 1)
                    endpoint_minimum = min(profile[0], profile[-1])
                    if not (MIN_NOMINAL_CLEARANCE_M <= minimum <= MAX_NOMINAL_CLEARANCE_M):
                        continue
                    if endpoint_minimum < MIN_ENDPOINT_CLEARANCE_M:
                        continue
                    if not MIN_FRACTION <= fraction <= MAX_FRACTION:
                        continue
                    geometry = local_geometry(
                        baseline_spheres, center_list, minimum_index, minimum_sphere, link_map
                    )
                    raw.append(
                        {
                            "box_center_m": center_list,
                            "source_target_index": target_index,
                            "source_sphere_index": sphere_index,
                            "source_axis": source_axis_name,
                            "target_clearance_m": target_clearance,
                            "reference_minimum_clearance_m": minimum,
                            "reference_minimum_index": minimum_index,
                            "reference_start_clearance_m": profile[0],
                            "reference_goal_clearance_m": profile[-1],
                            "reference_geometry": geometry,
                            "generation_score": (
                                endpoint_minimum
                                - 6.0 * abs(minimum - 0.012)
                                - 0.01 * abs(fraction - 0.5)
                            ),
                        }
                    )

    raw.sort(key=lambda item: item["generation_score"], reverse=True)
    deduplicated: list[dict[str, Any]] = []
    for item in raw:
        center = np.asarray(item["box_center_m"])
        if any(np.linalg.norm(center - np.asarray(old["box_center_m"])) < 0.035 for old in deduplicated):
            continue
        deduplicated.append(item)

    groups: dict[tuple[str, str], deque[dict[str, Any]]] = defaultdict(deque)
    for item in deduplicated:
        geometry = item["reference_geometry"]
        key = (geometry["minimum_sphere_link"], dominant_direction(geometry["danger_direction"]))
        groups[key].append(item)
    ordered_groups = sorted(groups, key=lambda key: (key[1], key[0]))
    balanced: list[dict[str, Any]] = []
    while ordered_groups:
        next_groups: list[tuple[str, str]] = []
        for key in ordered_groups:
            if groups[key]:
                balanced.append(groups[key].popleft())
            if groups[key]:
                next_groups.append(key)
        ordered_groups = next_groups
    return balanced


def evaluate(
    planner: MotionPlanner,
    generated: list[dict[str, Any]],
    link_map: dict[int, str],
    max_evaluations: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for number, source in enumerate(generated[:max_evaluations], start=1):
        candidate_id = f"multiscene_candidate_{number:03d}"
        center = source["box_center_m"]
        planner.clear_scene_cache()
        planner.update_world(make_scene(center))
        planner.reset_seed()
        success, trajectory, timing = plan_trajectory(planner)
        record: dict[str, Any] = {
            "candidate_id": candidate_id,
            "planning_success": success,
            "obstacle_position_m": center,
            "obstacle_size_m": BOX_DIMS,
            "start_joint_state": START_Q,
            "goal_joint_state": GOAL_Q,
            "generation": source,
            **timing,
            "status": "planned" if success else "planning_failed",
            "error": None,
        }
        if success and trajectory is not None:
            spheres = trajectory_spheres(planner, trajectory)
            profile, minimum_index, sphere_index = clearance_profile(spheres, center)
            steps = int(trajectory.shape[0])
            fraction = minimum_index / max(1, steps - 1)
            minimum = profile[minimum_index]
            geometry = local_geometry(spheres, center, minimum_index, sphere_index, link_map)
            eligible = (
                minimum > 0.0
                and MIN_NOMINAL_CLEARANCE_M <= minimum <= MAX_NOMINAL_CLEARANCE_M
                and MIN_FRACTION <= fraction <= MAX_FRACTION
                and min(profile[0], profile[-1]) >= MIN_ENDPOINT_CLEARANCE_M
            )
            record.update(
                {
                    "trajectory_num_steps": steps,
                    "trajectory_length_rad": trajectory_length(trajectory),
                    "trajectory_positions": trajectory.detach().cpu().tolist(),
                    "clearance_profile_m": profile,
                    "start_clearance_m": profile[0],
                    "goal_clearance_m": profile[-1],
                    "minimum_clearance_m": minimum,
                    "minimum_clearance_trajectory_index": minimum_index,
                    "minimum_clearance_fraction": fraction,
                    "nominal_collision": minimum <= 0.0,
                    "minimum_is_internal_20_80": MIN_FRACTION <= fraction <= MAX_FRACTION,
                    "endpoints_clearly_safe": min(profile[0], profile[-1]) >= MIN_ENDPOINT_CLEARANCE_M,
                    "shortlist_eligible": eligible,
                    **geometry,
                }
            )
        else:
            for key in (
                "trajectory_num_steps", "trajectory_length_rad", "trajectory_positions",
                "clearance_profile_m", "start_clearance_m", "goal_clearance_m",
                "minimum_clearance_m", "minimum_clearance_trajectory_index",
                "minimum_clearance_fraction", "nominal_collision", "minimum_sphere_index",
                "minimum_sphere_link", "minimum_sphere_world_position_m",
                "minimum_sphere_radius_m", "obstacle_nearest_surface_point_m",
                "surface_to_sphere_vector_m", "surface_to_sphere_center_distance_m",
                "danger_direction", "danger_direction_x", "danger_direction_y",
                "danger_direction_z", "angle_from_formal_baseline_v1_deg",
            ):
                record[key] = None
            record.update(
                {
                    "minimum_is_internal_20_80": False,
                    "endpoints_clearly_safe": False,
                    "shortlist_eligible": False,
                }
            )
        candidates.append(record)
        print(
            "MULTISCENE_CANDIDATE "
            + json.dumps(
                {
                    "candidate_id": candidate_id,
                    "planning_success": success,
                    "eligible": record["shortlist_eligible"],
                    "minimum_clearance_m": record["minimum_clearance_m"],
                    "minimum_index": record["minimum_clearance_trajectory_index"],
                    "link": record["minimum_sphere_link"],
                    "danger_direction": record["danger_direction"],
                },
                sort_keys=True,
            )
        )
    return candidates


def select_shortlist(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    eligible = [item for item in candidates if item["shortlist_eligible"]]
    eligible.sort(
        key=lambda item: (
            item["minimum_sphere_link"] != "panda_link6",
            item["angle_from_formal_baseline_v1_deg"],
            min(item["start_clearance_m"], item["goal_clearance_m"]),
            -abs(item["minimum_clearance_m"] - 0.012),
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    used_signatures: set[tuple[str, str]] = set()
    for candidate in eligible:
        signature = (
            candidate["minimum_sphere_link"],
            dominant_direction(candidate["danger_direction"]),
        )
        if signature in used_signatures:
            continue
        selected.append(candidate)
        used_signatures.add(signature)
        if len(selected) == count:
            return selected
    for candidate in eligible:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def write_outputs(
    output_dir: Path,
    baseline_path: Path,
    candidates: list[dict[str, Any]],
    shortlist: list[dict[str, Any]],
    generated_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [item for item in candidates if item["shortlist_eligible"]]
    metadata = {
        "schema_version": 1,
        "status": "completed",
        "automatic_final_selection": False,
        "formal_baseline_v1_modified": False,
        "search_scope": "fixed start/goal, public Franka Panda, one axis-aligned box",
        "planner": "cuRobo V2 MotionPlanner.plan_cspace",
        "candidate_generation": "deterministic offsets around obstacle-free reference-path collision spheres",
        "generated_center_count": generated_count,
        "evaluated_candidate_count": len(candidates),
        "valid_candidate_count": len(eligible),
        "shortlist_candidate_ids": [item["candidate_id"] for item in shortlist],
        "fixed_parameters": {
            "start_joint_state": START_Q,
            "goal_joint_state": GOAL_Q,
            "obstacle_size_m": BOX_DIMS,
        },
        "formal_baseline_v1": {
            "config_file": "configs/experiments/formal_baseline_v1.json",
            "sha256": sha256_file(baseline_path),
            "danger_direction": BASELINE_V1_DANGER.tolist(),
            "minimum_sphere_link": "panda_link6",
            "minimum_clearance_trajectory_index": 19,
        },
    }
    with (output_dir / "scenario_candidates.json").open("w", encoding="utf-8") as stream:
        json.dump({**metadata, "candidates": candidates}, stream, indent=2, allow_nan=False)
    with (output_dir / "shortlist.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "automatic_final_selection": False,
                "purpose": "candidates for later human review and Isaac Sim validation",
                "selection_dimensions": ["nominal safety", "internal minimum", "safe endpoints", "danger direction", "robot link"],
                "candidates": shortlist,
            },
            stream,
            indent=2,
            allow_nan=False,
        )

    summary_fields = [
        "candidate_id", "obstacle_position_m", "obstacle_size_m", "planning_success",
        "trajectory_num_steps", "trajectory_length_rad", "start_clearance_m",
        "goal_clearance_m", "minimum_clearance_m", "minimum_clearance_trajectory_index",
        "minimum_clearance_fraction", "minimum_sphere_index", "minimum_sphere_link",
        "danger_direction_x", "danger_direction_y", "danger_direction_z",
        "angle_from_formal_baseline_v1_deg", "nominal_collision", "shortlist_eligible", "status",
    ]
    with (output_dir / "scenario_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({key: candidate.get(key) for key in summary_fields})

    danger_fields = [
        "candidate_id", "minimum_clearance_trajectory_index", "minimum_sphere_index",
        "minimum_sphere_link", "minimum_sphere_world_position_m",
        "obstacle_nearest_surface_point_m", "surface_to_sphere_vector_m",
        "danger_direction_x", "danger_direction_y", "danger_direction_z",
        "angle_from_formal_baseline_v1_deg",
    ]
    with (output_dir / "danger_directions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=danger_fields, lineterminator="\n")
        writer.writeheader()
        for candidate in candidates:
            if candidate["planning_success"]:
                writer.writerow({key: candidate.get(key) for key in danger_fields})

    with (output_dir / "clearance_profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["candidate_id", "trajectory_index", "trajectory_fraction", "clearance_m"],
            lineterminator="\n",
        )
        writer.writeheader()
        for candidate in candidates:
            profile = candidate["clearance_profile_m"]
            if profile is None:
                continue
            denominator = max(1, len(profile) - 1)
            for index, clearance in enumerate(profile):
                writer.writerow(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "trajectory_index": index,
                        "trajectory_fraction": index / denominator,
                        "clearance_m": clearance,
                    }
                )
    print("MULTISCENE_SHORTLIST " + json.dumps([item["candidate_id"] for item in shortlist]))
    print(f"MULTISCENE_RESULTS {output_dir}")


def run_search(args: argparse.Namespace) -> None:
    if not args.baseline_config.is_file():
        raise FileNotFoundError(args.baseline_config)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multiscene calibration")
    with args.baseline_config.open(encoding="utf-8") as stream:
        baseline_v1 = json.load(stream)
    baseline_v1_center = baseline_v1["obstacle"]["nominal_position_m"]
    baseline_hash_before = sha256_file(args.baseline_config)
    torch.manual_seed(0)
    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot="franka.yml",
            scene_model=make_scene(FAR_BOX_CENTER),
            random_seed=0,
            optimizer_collision_activation_distance=0.01,
        )
    )
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    success, reference_trajectory, _ = plan_trajectory(planner)
    if not success or reference_trajectory is None:
        raise RuntimeError("obstacle-free reference plan failed")
    spheres = trajectory_spheres(planner, reference_trajectory)
    link_map = sphere_link_map(planner)
    generated = generate_centers(spheres, link_map, baseline_v1_center)
    if not generated:
        raise RuntimeError("no candidate centers passed reference-path prefilter")
    print(f"MULTISCENE_GENERATED {len(generated)}")
    candidates = evaluate(planner, generated, link_map, args.max_evaluations)
    shortlist = select_shortlist(candidates, args.shortlist_count)
    if len([item for item in candidates if item["shortlist_eligible"]]) < 5:
        raise RuntimeError("fewer than five valid candidates were found")
    if len(shortlist) < 3:
        raise RuntimeError("fewer than three candidates were available for shortlist")
    write_outputs(args.output_dir, args.baseline_config, candidates, shortlist, len(generated))
    if sha256_file(args.baseline_config) != baseline_hash_before:
        raise RuntimeError("Formal Baseline V1 changed during search")


def run_gui(output_dir: Path, candidate_id: str, port: int) -> None:
    with (output_dir / "scenario_candidates.json").open(encoding="utf-8") as stream:
        payload = json.load(stream)
    matches = [item for item in payload["candidates"] if item["candidate_id"] == candidate_id]
    if len(matches) != 1 or not matches[0]["planning_success"]:
        raise ValueError(f"unknown or unsuccessful candidate: {candidate_id}")
    candidate = matches[0]
    from curobo.viewer import ViserVisualizer

    visualizer = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=False,
        visualize_robot_spheres=True,
    )
    visualizer.add_scene(make_scene(candidate["obstacle_position_m"]))
    index = candidate["minimum_clearance_trajectory_index"]
    q = torch.tensor(candidate["trajectory_positions"][index], device="cuda", dtype=torch.float32)
    visualizer.set_joint_state(JointState.from_position(q, ARM_JOINT_NAMES))
    print(f"MULTISCENE_GUI_READY http://localhost:{port} candidate={candidate_id} index={index}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


def main() -> None:
    args = parse_args()
    if args.gui:
        run_gui(args.output_dir, args.gui, args.port)
    else:
        run_search(args)


if __name__ == "__main__":
    main()

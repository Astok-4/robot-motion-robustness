#!/usr/bin/env python3
"""Generate and evaluate deterministic Franka single-box calibration scenes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Cuboid, Scene
from curobo.types import ContentPath, JointState


START_Q = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
GOAL_Q = [0.785, -0.785, 0.524, -2.356, 0.0, 1.571, 0.785]
ARM_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]
BOX_DIMS = [0.1, 0.1, 0.1]
FAR_BOX_CENTER = [2.0, 2.0, 2.0]
TARGET_FRACTIONS = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
AXES = [
    [1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, 1.0],
]
TARGET_BASELINE_CLEARANCES_M = [0.005, 0.01, 0.02]
MIN_ENDPOINT_PREFILTER_M = 0.04
MIN_ENDPOINT_SHORTLIST_M = 0.03
MIN_FRACTION = 0.20
MAX_FRACTION = 0.80


def parse_args() -> argparse.Namespace:
    """Parse search and optional Viser display arguments."""
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=project_dir / "results_calibration")
    parser.add_argument("--max-evaluations", type=int, default=30)
    parser.add_argument("--shortlist-count", type=int, default=5)
    parser.add_argument("--gui", metavar="CANDIDATE_ID")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def make_scene(center: list[float]) -> Scene:
    """Create the calibration's single axis-aligned box scene."""
    return Scene(
        cuboid=[
            Cuboid(
                name="calibration_box",
                dims=BOX_DIMS,
                pose=[*center, 1.0, 0.0, 0.0, 0.0],
            )
        ]
    )


def trajectory_spheres(planner: MotionPlanner, q: torch.Tensor) -> torch.Tensor:
    """Compute cuRobo world-space collision spheres for a ``(T, 7)`` path."""
    names = planner.kinematics.config.kinematics_config.joint_names
    state = planner.kinematics.compute_kinematics(JointState.from_position(q, names))
    return state.robot_spheres.squeeze(1)


def sphere_box_clearance_matrix(spheres: torch.Tensor, center: list[float]) -> torch.Tensor:
    """Return signed sphere-to-AABB clearances with shape ``(T, spheres)``."""
    center_t = torch.tensor(center, device=spheres.device, dtype=spheres.dtype)
    half_t = torch.tensor(BOX_DIMS, device=spheres.device, dtype=spheres.dtype) / 2.0
    delta = torch.abs(spheres[..., :3] - center_t) - half_t
    outside = torch.linalg.vector_norm(torch.clamp(delta, min=0.0), dim=-1)
    inside = torch.clamp(torch.max(delta, dim=-1).values, max=0.0)
    return outside + inside - spheres[..., 3]


def clearance_profile(spheres: torch.Tensor, center: list[float]) -> tuple[list[float], int, int]:
    """Return per-waypoint minima and the global waypoint/sphere indices."""
    matrix = sphere_box_clearance_matrix(spheres, center)
    values, sphere_indices = torch.min(matrix, dim=-1)
    minimum_value, minimum_index = torch.min(values, dim=0)
    index = int(minimum_index.item())
    assert math.isclose(float(values[index].item()), float(minimum_value.item()), abs_tol=1e-9)
    return (
        [float(value) for value in values.detach().cpu().tolist()],
        index,
        int(sphere_indices[index].item()),
    )


def plan_trajectory(planner: MotionPlanner) -> tuple[bool, torch.Tensor | None, dict[str, float | None]]:
    """Plan once between the fixed calibration endpoints."""
    names = planner.joint_names
    start = JointState.from_position(torch.tensor([START_Q], device="cuda"), names)
    goal = JointState.from_position(torch.tensor([GOAL_Q], device="cuda"), names)
    wall_start = time.perf_counter()
    result = planner.plan_cspace(goal, start, max_attempts=5)
    wall_time = time.perf_counter() - wall_start
    success = bool(result is not None and result.success.any().item())
    timing: dict[str, float | None] = {
        "solve_time_s": float(result.solve_time) if result is not None else None,
        "total_planner_time_s": float(result.total_time) if result is not None else None,
        "wall_time_s": wall_time,
    }
    if not success:
        return False, None, timing
    active = planner.kinematics.get_active_js(result.get_interpolated_plan())
    return True, active.position.reshape(-1, 7).contiguous(), timing


def trajectory_length(q: torch.Tensor) -> float:
    """Return accumulated Euclidean joint-space length in radians."""
    return float(torch.linalg.vector_norm(q[1:] - q[:-1], dim=-1).sum().item())


def generate_candidate_centers(baseline_spheres: torch.Tensor) -> list[dict[str, Any]]:
    """Generate deterministic box centers near internal moving robot spheres."""
    step_count, sphere_count, _ = baseline_spheres.shape
    raw: list[dict[str, Any]] = []
    for fraction in TARGET_FRACTIONS:
        target_index = int(round(fraction * (step_count - 1)))
        for sphere_index in range(sphere_count):
            sphere = baseline_spheres[target_index, sphere_index]
            radius = float(sphere[3].item())
            if radius <= 0.0:
                continue
            point = sphere[:3].detach().cpu().numpy()
            for axis in AXES:
                for target_clearance in TARGET_BASELINE_CLEARANCES_M:
                    offset_distance = radius + BOX_DIMS[0] / 2.0 + target_clearance
                    center = point + offset_distance * np.asarray(axis)
                    if not (-0.25 <= center[0] <= 0.9 and -0.8 <= center[1] <= 0.8 and 0.12 <= center[2] <= 1.2):
                        continue
                    center_list = [float(value) for value in center]
                    profile, baseline_min_index, _ = clearance_profile(baseline_spheres, center_list)
                    endpoint_min = min(profile[0], profile[-1])
                    baseline_min = min(profile)
                    if endpoint_min < MIN_ENDPOINT_PREFILTER_M or not 0.001 <= baseline_min <= 0.03:
                        continue
                    raw.append(
                        {
                            "box_center_m": center_list,
                            "source_target_fraction": fraction,
                            "source_target_index": target_index,
                            "source_sphere_index": sphere_index,
                            "source_axis": axis,
                            "target_baseline_clearance_m": target_clearance,
                            "baseline_start_clearance_m": profile[0],
                            "baseline_goal_clearance_m": profile[-1],
                            "baseline_minimum_clearance_m": baseline_min,
                            "baseline_minimum_index": baseline_min_index,
                            "generation_score": (
                                endpoint_min
                                - abs(baseline_min - 0.01)
                                - 0.002 * abs(baseline_min_index / (step_count - 1) - 0.5)
                            ),
                        }
                    )

    raw.sort(key=lambda item: item["generation_score"], reverse=True)
    deduplicated: list[dict[str, Any]] = []
    for item in raw:
        center = np.asarray(item["box_center_m"])
        if any(np.linalg.norm(center - np.asarray(existing["box_center_m"])) < 0.055 for existing in deduplicated):
            continue
        deduplicated.append(item)
    return deduplicated


def evaluate_candidates(
    planner: MotionPlanner,
    generated: list[dict[str, Any]],
    max_evaluations: int,
) -> list[dict[str, Any]]:
    """Replan and compute the full clearance profile for each candidate scene."""
    evaluated: list[dict[str, Any]] = []
    for candidate_number, generated_item in enumerate(generated[:max_evaluations], start=1):
        candidate_id = f"candidate_{candidate_number:03d}"
        center = generated_item["box_center_m"]
        planner.clear_scene_cache()
        planner.update_world(make_scene(center))
        planner.reset_seed()
        success, q, timing = plan_trajectory(planner)
        record: dict[str, Any] = {
            "candidate_id": candidate_id,
            "planning_success": success,
            "box_center_m": center,
            "box_dims_m": BOX_DIMS,
            "start_q": START_Q,
            "goal_q": GOAL_Q,
            "generation": generated_item,
            **timing,
            "status": "planned" if success else "planning_failed",
            "error": None,
        }
        if success and q is not None:
            spheres = trajectory_spheres(planner, q)
            profile, minimum_index, sphere_index = clearance_profile(spheres, center)
            step_count = int(q.shape[0])
            minimum_fraction = minimum_index / (step_count - 1) if step_count > 1 else 0.0
            start_clearance = profile[0]
            goal_clearance = profile[-1]
            minimum_clearance = profile[minimum_index]
            nominal_collision = minimum_clearance <= 0.0
            interior = MIN_FRACTION <= minimum_fraction <= MAX_FRACTION
            endpoints_safe = (
                start_clearance >= MIN_ENDPOINT_SHORTLIST_M
                and goal_clearance >= MIN_ENDPOINT_SHORTLIST_M
            )
            shortlist_eligible = not nominal_collision and interior and endpoints_safe
            score = (
                (1000.0 if shortlist_eligible else 0.0)
                + 10.0 * min(start_clearance, goal_clearance)
                - abs(minimum_fraction - 0.5)
                - 8.0 * abs(minimum_clearance - 0.01)
            )
            record.update(
                {
                    "trajectory_num_steps": step_count,
                    "trajectory_length_rad": trajectory_length(q),
                    "trajectory_positions": q.detach().cpu().tolist(),
                    "clearance_profile_m": profile,
                    "start_clearance_m": start_clearance,
                    "goal_clearance_m": goal_clearance,
                    "minimum_clearance_m": minimum_clearance,
                    "minimum_clearance_trajectory_index": minimum_index,
                    "minimum_clearance_fraction": minimum_fraction,
                    "minimum_clearance_sphere_index": sphere_index,
                    "nominal_collision": nominal_collision,
                    "minimum_is_internal_20_80": interior,
                    "endpoints_clearly_safe": endpoints_safe,
                    "shortlist_eligible": shortlist_eligible,
                    "ranking_score": score,
                }
            )
        else:
            record.update(
                {
                    "trajectory_num_steps": None,
                    "trajectory_length_rad": None,
                    "trajectory_positions": None,
                    "clearance_profile_m": None,
                    "start_clearance_m": None,
                    "goal_clearance_m": None,
                    "minimum_clearance_m": None,
                    "minimum_clearance_trajectory_index": None,
                    "minimum_clearance_fraction": None,
                    "minimum_clearance_sphere_index": None,
                    "nominal_collision": None,
                    "minimum_is_internal_20_80": False,
                    "endpoints_clearly_safe": False,
                    "shortlist_eligible": False,
                    "ranking_score": -1.0e9,
                }
            )
        evaluated.append(record)
        print(
            "CALIBRATION_CANDIDATE "
            + json.dumps(
                {
                    "candidate_id": candidate_id,
                    "planning_success": success,
                    "minimum_clearance_m": record["minimum_clearance_m"],
                    "minimum_index": record["minimum_clearance_trajectory_index"],
                    "minimum_fraction": record["minimum_clearance_fraction"],
                    "start_clearance_m": record["start_clearance_m"],
                    "goal_clearance_m": record["goal_clearance_m"],
                    "shortlist_eligible": record["shortlist_eligible"],
                },
                sort_keys=True,
            )
        )
    return evaluated


def write_outputs(output_dir: Path, candidates: list[dict[str, Any]], shortlist_count: int) -> None:
    """Write all candidate parameters, summaries, and per-waypoint profiles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eligible = [item for item in candidates if item["shortlist_eligible"]]
    eligible.sort(key=lambda item: item["ranking_score"], reverse=True)
    shortlist = eligible[:shortlist_count]
    payload = {
        "schema_version": 1,
        "status": "completed",
        "error": None,
        "method": {
            "robot": "Franka Panda",
            "planner": "cuRobo V2 MotionPlanner.plan_cspace",
            "scene": "single axis-aligned box",
            "candidate_generation": "deterministic positive-clearance offsets from internal baseline collision spheres",
            "random_seed": 0,
            "minimum_clearance_fraction_preference": [MIN_FRACTION, MAX_FRACTION],
            "minimum_endpoint_clearance_for_shortlist_m": MIN_ENDPOINT_SHORTLIST_M,
            "automatic_final_selection": False,
        },
        "fixed_parameters": {
            "start_q": START_Q,
            "goal_q": GOAL_Q,
            "box_dims_m": BOX_DIMS,
        },
        "candidate_count": len(candidates),
        "shortlist_candidate_ids": [item["candidate_id"] for item in shortlist],
        "candidates": candidates,
    }
    with (output_dir / "scenario_candidates.json").open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)

    summary_fields = [
        "candidate_id",
        "planning_success",
        "box_center_m",
        "box_dims_m",
        "trajectory_num_steps",
        "trajectory_length_rad",
        "start_clearance_m",
        "goal_clearance_m",
        "minimum_clearance_m",
        "minimum_clearance_trajectory_index",
        "minimum_clearance_fraction",
        "minimum_clearance_sphere_index",
        "nominal_collision",
        "minimum_is_internal_20_80",
        "endpoints_clearly_safe",
        "shortlist_eligible",
        "ranking_score",
        "solve_time_s",
        "wall_time_s",
        "status",
    ]
    with (output_dir / "scenario_candidates.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow({field: candidate.get(field) for field in summary_fields})

    with (output_dir / "clearance_profiles.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["candidate_id", "trajectory_index", "trajectory_fraction", "clearance_m"]
        writer = csv.DictWriter(stream, fieldnames=fields)
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

    with (output_dir / "shortlist.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "automatic_final_selection": False,
                "selection_rule": "safe nominal path, >=30 mm endpoint clearance, minimum index in 20%-80%",
                "candidates": shortlist,
            },
            stream,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    print("CALIBRATION_SHORTLIST " + json.dumps([item["candidate_id"] for item in shortlist]))
    print(f"CALIBRATION_RESULTS {output_dir}")


def run_search(args: argparse.Namespace) -> None:
    """Generate, evaluate, rank, and save deterministic calibration candidates."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for scenario calibration")
    torch.manual_seed(0)
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=make_scene(FAR_BOX_CENTER),
        random_seed=0,
        optimizer_collision_activation_distance=0.01,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=5)
    baseline_success, baseline_q, _ = plan_trajectory(planner)
    if not baseline_success or baseline_q is None:
        raise RuntimeError("baseline calibration plan failed")
    baseline_spheres = trajectory_spheres(planner, baseline_q)
    generated = generate_candidate_centers(baseline_spheres)
    if not generated:
        raise RuntimeError("candidate generator found no centers with safe baseline endpoints")
    print(f"CALIBRATION_GENERATED {len(generated)}")
    candidates = evaluate_candidates(planner, generated, args.max_evaluations)
    write_outputs(args.output_dir, candidates, args.shortlist_count)


def run_gui(output_dir: Path, candidate_id: str, port: int) -> None:
    """Show one saved candidate at its minimum-clearance waypoint in Viser."""
    result_path = output_dir / "scenario_candidates.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"calibration results do not exist: {result_path}")
    with result_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    matches = [item for item in payload["candidates"] if item["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError(f"unknown candidate id: {candidate_id}")
    candidate = matches[0]
    if not candidate["planning_success"]:
        raise ValueError(f"candidate did not plan successfully: {candidate_id}")

    from curobo.viewer import ViserVisualizer

    visualizer = ViserVisualizer(
        content_path=ContentPath(robot_config_file="franka.yml"),
        connect_ip="0.0.0.0",
        connect_port=port,
        add_control_frames=False,
        visualize_robot_spheres=True,
    )
    visualizer.add_scene(make_scene(candidate["box_center_m"]))
    index = candidate["minimum_clearance_trajectory_index"]
    q = torch.tensor(candidate["trajectory_positions"][index], device="cuda", dtype=torch.float32)
    visualizer.set_joint_state(JointState.from_position(q, ARM_JOINT_NAMES))
    print(f"CALIBRATION_GUI_READY http://localhost:{port} candidate={candidate_id} index={index}")
    print("Press Ctrl-C to close the viewer.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        return


def main() -> None:
    """Run deterministic search or display one saved candidate."""
    args = parse_args()
    if args.gui:
        run_gui(args.output_dir, args.gui, args.port)
    else:
        run_search(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run/resume the frozen Formal RQ2 Planning-Time Benchmark V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from calibrate_planning_obstacle_inflation import box_scene, plan_once, trajectory_length
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_formal/rq2_planning_time_v1"
BASELINE = ROOT / "configs/experiments/formal_baseline_set_v1.json"
MATRIX = ROOT / "configs/experiments/formal_safety_margin_matrix_v1.json"
PROTOCOL = ROOT / "configs/experiments/formal_rq1_rq2_protocol_v1.json"
PLANNER_CONFIG = ROOT / "configs/experiments/planning_obstacle_inflation_calibration_v1.json"
EXPECTED = {
    BASELINE: "3a3b4806e2a407253006be509063ccf4b584c873bc5e8e8f9110949de94e8f6b",
    MATRIX: "8869b9709cf6c84274d77e2d28a2bf8354eb15f3e3740362da0b2d8871880d04",
    PROTOCOL: "eb270c2219460b44a1e9ff445b66dc40b18d1f4690383c66d6478419dba3022f",
}
FIELDS = [
    "attempt_id", "phase", "round_id", "execution_position", "position_within_round",
    "condition_id", "scene_id", "inflation_mm", "planning_success", "failure_reason",
    "planning_reported_solve_time_s", "planning_reported_total_time_s", "measured_wall_time_s",
    "trajectory_shape", "trajectory_finite", "maximum_absolute_joint_difference_rad",
    "mean_absolute_joint_difference_rad", "trajectory_length_rad",
    "frozen_trajectory_length_rad", "trajectory_length_difference_rad",
    "exact_frozen_trajectory_match", "tolerance_frozen_trajectory_match_1e_7_rad",
    "status", "error",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_assets() -> dict[str, str]:
    actual = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        if actual[str(path.relative_to(ROOT))] != expected:
            raise RuntimeError(f"frozen asset SHA mismatch: {path}")
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-measured", type=int)
    return parser.parse_args()


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        values = [row["attempt_id"] for row in csv.DictReader(stream)]
    if len(values) != len(set(values)):
        raise RuntimeError(f"duplicate attempt ID in {path}")
    return set(values)


def serialize(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: json.dumps(row.get(key), separators=(",", ":"))
        if isinstance(row.get(key), (list, dict)) else row.get(key)
        for key in FIELDS
    }


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(serialize(row))
        stream.flush()
        os.fsync(stream.fileno())


def run_attempt(
    planner: MotionPlanner, condition: dict[str, Any], planner_cfg: dict[str, Any],
    *, attempt_id: str, phase: str, round_id: int, execution_position: int,
    position_within_round: int,
) -> dict[str, Any]:
    planner.clear_scene_cache()
    planner.update_world(box_scene(
        condition["nominal_obstacle_center_m"], condition["planning_obstacle_size_m"]
    ))
    planner.reset_seed()
    torch.manual_seed(int(planner_cfg["random_seed"]))
    success, trajectory, timing = plan_once(
        planner, condition["start_configuration"], condition["goal_configuration"],
        int(planner_cfg["max_attempts"]),
    )
    frozen = np.asarray(condition["trajectory"]["position"], dtype=np.float64)
    record: dict[str, Any] = {
        "attempt_id": attempt_id, "phase": phase, "round_id": round_id,
        "execution_position": execution_position,
        "position_within_round": position_within_round,
        "condition_id": condition["condition_id"], "scene_id": condition["scene_id"],
        "inflation_mm": condition["inflation_mm"], "planning_success": success,
        **timing, "frozen_trajectory_length_rad": condition["trajectory_length_rad"],
        "status": "completed", "error": None,
    }
    if success and trajectory is not None:
        current = trajectory.detach().cpu().numpy().astype(np.float64)
        finite = bool(np.isfinite(current).all())
        shapes_equal = current.shape == frozen.shape
        max_diff = float(np.max(np.abs(current - frozen))) if shapes_equal else None
        mean_diff = float(np.mean(np.abs(current - frozen))) if shapes_equal else None
        current_length = trajectory_length(trajectory)
        record.update({
            "trajectory_shape": list(current.shape), "trajectory_finite": finite,
            "maximum_absolute_joint_difference_rad": max_diff,
            "mean_absolute_joint_difference_rad": mean_diff,
            "trajectory_length_rad": current_length,
            "trajectory_length_difference_rad": current_length - condition["trajectory_length_rad"],
            "exact_frozen_trajectory_match": bool(shapes_equal and np.array_equal(current, frozen)),
            "tolerance_frozen_trajectory_match_1e_7_rad": bool(
                finite and shapes_equal and max_diff is not None and max_diff <= 1e-7
            ),
        })
    else:
        record.update({key: None for key in [
            "trajectory_shape", "trajectory_finite", "maximum_absolute_joint_difference_rad",
            "mean_absolute_joint_difference_rad", "trajectory_length_rad",
            "trajectory_length_difference_rad", "exact_frozen_trajectory_match",
            "tolerance_frozen_trajectory_match_1e_7_rad",
        ]})
    return record


def main() -> None:
    args = parse_args()
    hashes = verify_assets()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    planner_design = json.loads(PLANNER_CONFIG.read_text(encoding="utf-8"))
    conditions = {row["condition_id"]: row for row in matrix["conditions"]}
    timing = protocol["planning_timing_protocol"]
    warmup_schedule = timing["warmup_schedule"]
    measured_schedule = timing["execution_schedule"]
    if len(conditions) != 20 or len(warmup_schedule) != 20 or len(measured_schedule) != 30:
        raise RuntimeError("invalid frozen timing population")
    if set(warmup_schedule) != set(conditions):
        raise RuntimeError("warmup schedule does not cover formal conditions")
    for schedule in measured_schedule:
        if len(schedule["condition_order"]) != 20 or set(schedule["condition_order"]) != set(conditions):
            raise RuntimeError("invalid frozen measured order")
        expected_offset = (7 * (int(schedule["round_index"]) - 1)) % 20
        if int(schedule["cyclic_offset"]) != expected_offset:
            raise RuntimeError("frozen cyclic offset audit failed")

    planner_cfg = planner_design["planner"]
    first = conditions[warmup_schedule[0]]
    torch.manual_seed(int(planner_cfg["random_seed"]))
    config = MotionPlannerCfg.create(
        robot=planner_cfg["robot_config"],
        scene_model=box_scene(first["nominal_obstacle_center_m"], first["planning_obstacle_size_m"]),
        random_seed=int(planner_cfg["random_seed"]),
        optimizer_collision_activation_distance=float(planner_cfg["optimizer_collision_activation_distance_m"]),
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=int(planner_cfg["warmup_iterations"]))
    OUT.mkdir(parents=True, exist_ok=True)
    warmup_path, measured_path = OUT / "warmup.csv", OUT / "measured_runs.csv"
    warmup_done, measured_done = existing_ids(warmup_path), existing_ids(measured_path)
    started = time.perf_counter()
    run_started_utc = datetime.now(timezone.utc).isoformat()

    for index, condition_id in enumerate(warmup_schedule, start=1):
        attempt_id = f"warmup_{index:02d}__{condition_id}"
        if attempt_id in warmup_done:
            continue
        row = run_attempt(
            planner, conditions[condition_id], planner_cfg, attempt_id=attempt_id,
            phase="warmup", round_id=0, execution_position=index,
            position_within_round=index,
        )
        append_rows(warmup_path, [row])
        print(f"FORMAL_TIMING_WARMUP {index}/20 success={row['planning_success']} condition={condition_id}", flush=True)
    if len(existing_ids(warmup_path)) != 20:
        raise RuntimeError("formal warmup incomplete")

    scheduled = []
    global_position = 0
    for schedule in measured_schedule:
        round_id = int(schedule["round_index"])
        for position, condition_id in enumerate(schedule["condition_order"], start=1):
            global_position += 1
            scheduled.append((
                f"round_{round_id:02d}__position_{position:02d}__{condition_id}",
                round_id, global_position, position, condition_id,
            ))
    pending = [item for item in scheduled if item[0] not in measured_done]
    if args.max_new_measured is not None:
        pending = pending[:args.max_new_measured]
    for completed_this_run, (attempt_id, round_id, execution_position, position, condition_id) in enumerate(pending, start=1):
        row = run_attempt(
            planner, conditions[condition_id], planner_cfg, attempt_id=attempt_id,
            phase="measured", round_id=round_id, execution_position=execution_position,
            position_within_round=position,
        )
        append_rows(measured_path, [row])
        if completed_this_run % 20 == 0 or completed_this_run == len(pending):
            print(
                f"FORMAL_TIMING_PROGRESS new={completed_this_run}/{len(pending)} "
                f"total={len(measured_done)+completed_this_run}/600 round={round_id}", flush=True
            )

    metadata = {
        "experiment_id": "formal_rq2_planning_time_v1", "status": "execution_complete_pending_audit",
        "execution_started_utc": run_started_utc,
        "execution_finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_time_this_process_s": time.perf_counter() - started,
        "frozen_sha256": hashes,
        "planner_configuration_file_sha256": sha256(PLANNER_CONFIG),
        "planner_configuration": planner_cfg,
        "curobo_commit": "8e734f3", "python_version": platform.python_version(),
        "pytorch_version": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0),
        "warmup_attempt_count": len(existing_ids(warmup_path)),
        "measured_attempt_count": len(existing_ids(measured_path)),
        "resume_supported_by_attempt_id": True,
        "crash_resume_occurred": False,
        "process_resume_count": 0,
        "robustness_experiment_run": False,
    }
    (OUT / "execution_status.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"FORMAL_TIMING_EXECUTION_RETURN warmup={metadata['warmup_attempt_count']} measured={metadata['measured_attempt_count']}")


if __name__ == "__main__":
    main()

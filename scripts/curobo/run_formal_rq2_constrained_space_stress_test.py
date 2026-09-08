#!/usr/bin/env python3
"""Run the frozen supplementary constrained-space RQ2 timing benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from audit_rq2_constrained_space_gate_local import strengthened_passage  # noqa: E402
from calibrate_rq2_constrained_space import (  # noqa: E402
    end_effector_positions, make_scene, plan_once, trajectory_length, trajectory_state,
)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402

PROTOCOL = ROOT / "configs/experiments/formal_rq2_constrained_space_stress_test_v1.json"
CALIBRATION_CONFIG = ROOT / "configs/experiments/rq2_constrained_space_calibration_v1.json"
OUT = ROOT / "results_formal/rq2_constrained_space_stress_test_v1"
FIELDS = [
    "attempt_id", "phase", "round_id", "execution_position", "condition_id",
    "corridor_width_mm", "inflation_mm", "effective_planning_gap_mm",
    "planning_success", "failure_reason", "planning_reported_solve_time_s",
    "planning_reported_total_time_s", "measured_wall_time_s", "trajectory_shape",
    "trajectory_finite", "trajectory_length_rad", "maximum_absolute_joint_difference_rad",
    "exact_canonical_trajectory_match", "route_category", "passage_used", "upper_bypass",
    "crossing_index", "crossing_position_m", "status",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def append(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow({k: json.dumps(row.get(k), separators=(",", ":")) if isinstance(row.get(k), (list, dict)) else row.get(k) for k in FIELDS})
        stream.flush(); os.fsync(stream.fileno())


def existing(path: Path) -> set[str]:
    if not path.exists(): return set()
    with path.open(newline="", encoding="utf-8") as stream: ids = [r["attempt_id"] for r in csv.DictReader(stream)]
    if len(ids) != len(set(ids)): raise RuntimeError(f"duplicate attempt in {path}")
    return set(ids)


def main() -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    config = json.loads(CALIBRATION_CONFIG.read_text(encoding="utf-8"))
    for rel, expected in protocol["source_sha256"].items():
        if rel.startswith("configs/") and sha256(ROOT / rel) != expected:
            raise RuntimeError(f"frozen asset SHA mismatch: {rel}")
    protocol_sha = sha256(PROTOCOL)
    conditions = {c["condition_id"]: c for c in protocol["conditions"]}
    first = next(iter(conditions.values()))
    planner_cfg = protocol["planner"]
    planner = MotionPlanner(MotionPlannerCfg.create(
        robot=planner_cfg["robot_config"], scene_model=make_scene(first["boxes"], True),
        random_seed=planner_cfg["random_seed"],
        optimizer_collision_activation_distance=planner_cfg["optimizer_collision_activation_distance_m"],
    ))
    planner.warmup(enable_graph=True, num_warmup_iterations=10)
    OUT.mkdir(parents=True, exist_ok=True)
    warm_path, measured_path = OUT / "warmup.csv", OUT / "measured_runs.csv"

    def execute(condition: dict[str, Any], attempt_id: str, phase: str, round_id: int, position: int) -> dict[str, Any]:
        planner.clear_scene_cache(); planner.update_world(make_scene(condition["boxes"], True))
        planner.reset_seed(); torch.manual_seed(int(planner_cfg["random_seed"]))
        success, trajectory, timing = plan_once(planner, condition["start_configuration"], condition["goal_configuration"], int(planner_cfg["max_attempts"]))
        row: dict[str, Any] = {
            "attempt_id": attempt_id, "phase": phase, "round_id": round_id,
            "execution_position": position, "condition_id": condition["condition_id"],
            "corridor_width_mm": condition["corridor_width_mm"], "inflation_mm": condition["inflation_mm"],
            "effective_planning_gap_mm": condition["effective_planning_gap_mm"],
            "planning_success": success, "status": "completed", **timing,
        }
        if success and trajectory is not None:
            current = trajectory.detach().cpu().numpy().astype(np.float64)
            canonical = np.asarray(condition["canonical_trajectory_position"], dtype=np.float64)
            same = current.shape == canonical.shape
            route = strengthened_passage(end_effector_positions(planner, trajectory_state(planner, trajectory)), config, int(condition["corridor_width_mm"]))
            row.update({
                "trajectory_shape": list(current.shape), "trajectory_finite": bool(np.isfinite(current).all()),
                "trajectory_length_rad": trajectory_length(trajectory),
                "maximum_absolute_joint_difference_rad": float(np.max(np.abs(current-canonical))) if same else None,
                "exact_canonical_trajectory_match": bool(same and np.array_equal(current, canonical)),
                "route_category": route["route_category"], "passage_used": route["passage_used"],
                "upper_bypass": route["route_category"] == "upper_bypass",
                "crossing_index": route["crossing_index"], "crossing_position_m": route["crossing_position_m"],
            })
        return row

    done = existing(warm_path)
    for position, condition_id in enumerate(protocol["warmup"]["condition_order"]):
        attempt_id = f"warmup_{position:03d}"
        if attempt_id not in done: append(warm_path, execute(conditions[condition_id], attempt_id, "warmup", -1, position))
    done = existing(measured_path)
    for item in protocol["timing"]["measured_schedule"]:
        attempt_id = f"round_{int(item['round_id']):02d}__position_{int(item['execution_position']):02d}"
        if attempt_id not in done:
            append(measured_path, execute(conditions[item["condition_id"]], attempt_id, "measured", int(item["round_id"]), int(item["execution_position"])))
    metadata = {
        "experiment_id": "formal_rq2_constrained_space_stress_test_v1", "experiment_type": "supplementary_post_hoc_stress_test",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_sha,
        "warmup_calls": 20, "measured_calls": 600, "robustness_experiment_run": False,
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0), "source_sha256": protocol["source_sha256"],
    }
    (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("FORMAL_RQ2_CONSTRAINED_SPACE_STRESS_TEST_EXECUTION_COMPLETE")


if __name__ == "__main__": main()

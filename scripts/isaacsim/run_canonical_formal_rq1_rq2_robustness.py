#!/usr/bin/env python3
"""Run/resume the canonical formal sequential PhysX robustness dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
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
ARM_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
FRANKA_ASSET = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
FRANKA_PATH = "/World/Franka"
OBSTACLE_PATH = "/World/FormalObstacle"
FIELDS = [
    "trial_id", "condition_id", "scene_id", "inflation_mm", "direction_id", "magnitude_mm",
    "direction_x", "direction_y", "direction_z", "delta_x_m", "delta_y_m", "delta_z_m",
    "actual_obstacle_position_m", "curobo_minimum_clearance_m", "curobo_minimum_clearance_index",
    "curobo_collision", "physx_collision", "physx_minimum_separation_m",
    "physx_minimum_separation_index", "physx_first_collision_index",
    "physx_minimum_contact", "physx_first_collision_contact", "trajectory_states_replayed",
    "trajectory_frozen", "replanned_after_perturbation", "status", "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=["nominal", "coarse", "refinement", "reproducibility"], required=True
    )
    parser.add_argument("--max-new-trials", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_assets() -> dict[str, str]:
    actual = {str(path.relative_to(ROOT)): sha256(path) for path in EXPECTED}
    for path, expected in EXPECTED.items():
        if actual[str(path.relative_to(ROOT))] != expected:
            raise RuntimeError(f"frozen asset SHA mismatch: {path}")
    return actual


def parse_bool(value: str) -> bool:
    return value.lower() == "true"


def load_prepared(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    parsed = []
    for row in rows:
        parsed.append({
            **row,
            "inflation_mm": int(row["inflation_mm"]), "magnitude_mm": float(row["magnitude_mm"]),
            "direction_x": float(row["direction_x"]), "direction_y": float(row["direction_y"]), "direction_z": float(row["direction_z"]),
            "delta_x_m": float(row["delta_x_m"]), "delta_y_m": float(row["delta_y_m"]), "delta_z_m": float(row["delta_z_m"]),
            "actual_obstacle_position": np.asarray(json.loads(row["actual_obstacle_position_m"]), dtype=np.float32),
            "curobo_minimum_clearance_m": float(row["curobo_minimum_clearance_m"]),
            "curobo_minimum_clearance_index": int(row["curobo_minimum_clearance_index"]),
            "curobo_collision": parse_bool(row["curobo_collision"]),
        })
    return parsed


def load_conditions() -> dict[str, dict[str, Any]]:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    result = {}
    for condition in matrix["conditions"]:
        trajectory = np.asarray(condition["trajectory"]["position"], dtype=np.float32)
        if trajectory.shape != (41, 7) or condition["joint_names"] != ARM_NAMES:
            raise ValueError(f"invalid frozen trajectory {condition['condition_id']}")
        result[condition["condition_id"]] = {**condition, "trajectory_array": trajectory}
    if len(result) != 20:
        raise ValueError("expected 20 conditions")
    return result


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    ids = [row["trial_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"duplicate checkpoint trial ID in {path}")
    return set(ids)


def path_string(encoded: int, converter: Any) -> str:
    return str(converter(encoded))


def contacts(simulation_interface: Any, converter: Any) -> list[dict[str, Any]]:
    headers, data = simulation_interface.get_contact_report()
    records = []
    for header in headers:
        paths = {
            "actor0": path_string(header.actor0, converter), "actor1": path_string(header.actor1, converter),
            "collider0": path_string(header.collider0, converter), "collider1": path_string(header.collider1, converter),
        }
        values = list(paths.values())
        robot = any(p == FRANKA_PATH or p.startswith(FRANKA_PATH + "/") for p in values)
        obstacle = any(p == OBSTACLE_PATH or p.startswith(OBSTACLE_PATH + "/") for p in values)
        if not robot or not obstacle or header.num_contact_data <= 0:
            continue
        separations = [float(data[i].separation) for i in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data)]
        records.append({**paths, "contact_count": int(header.num_contact_data), "minimum_separation_m": min(separations)})
    return records


def replay(
    trial: dict[str, Any], trajectory: np.ndarray, franka: Any, obstacle: Any, world: Any,
    arm_indices: list[int], finger_indices: list[int], simulation_interface: Any, converter: Any,
) -> dict[str, Any]:
    obstacle.set_world_pose(position=trial["actual_obstacle_position"])
    simulation_interface.get_contact_report()
    reports = []
    collision_indices = []
    first_contact = None
    for index, arm in enumerate(trajectory):
        q = np.asarray(franka.get_joint_positions(), dtype=np.float32)
        q[arm_indices] = arm
        q[finger_indices] = 0.04
        franka.set_joint_positions(q)
        franka.set_joint_velocities(np.zeros(9, dtype=np.float32))
        world.step(render=False)
        current = contacts(simulation_interface, converter)
        reports.extend((index, record) for record in current)
        penetrating = [record for record in current if record["minimum_separation_m"] <= 0.0]
        if penetrating:
            collision_indices.append(index)
            if first_contact is None:
                first_contact = penetrating[0]
    minimum = min(reports, key=lambda item: item[1]["minimum_separation_m"], default=None)
    return {
        "physx_collision": bool(collision_indices),
        "physx_minimum_separation_m": minimum[1]["minimum_separation_m"] if minimum else None,
        "physx_minimum_separation_index": minimum[0] if minimum else None,
        "physx_first_collision_index": collision_indices[0] if collision_indices else None,
        "physx_minimum_contact": minimum[1] if minimum else None,
        "physx_first_collision_contact": first_contact,
    }


def serialized(row: dict[str, Any]) -> dict[str, Any]:
    return {key: json.dumps(row.get(key), separators=(",", ":")) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in FIELDS}


def main() -> None:
    args = parse_args()
    hashes = verify_assets()
    conditions = load_conditions()
    phase_files = {
        "nominal": ("prepared_nominal_trials.csv", "nominal_validation.csv"),
        "coarse": ("prepared_coarse_trials.csv", "coarse_trials.csv"),
        "refinement": ("prepared_boundary_refinement_trials.csv", "boundary_refinement_trials.csv"),
        "reproducibility": ("prepared_reproducibility_subset.csv", "reproducibility_subset.csv"),
    }
    prepared_name, output_name = phase_files[args.phase]
    prepared_path = OUT / prepared_name
    output_path = OUT / output_name
    trials = load_prepared(prepared_path)
    expected = {"nominal": 20, "coarse": 76800}.get(args.phase, len(trials))
    if len(trials) != expected or len({row["trial_id"] for row in trials}) != expected:
        raise RuntimeError(f"invalid prepared {args.phase} population")
    if args.phase == "coarse":
        nominal_path = OUT / "nominal_validation.csv"
        if len(read_completed(nominal_path)) != 20:
            raise RuntimeError("all 20 nominal validations must complete before coarse phase")
        with nominal_path.open(newline="", encoding="utf-8") as stream:
            unsafe = [row["condition_id"] for row in csv.DictReader(stream) if parse_bool(row["physx_collision"])]
        if unsafe:
            raise RuntimeError(f"nominal PhysX gate failed: {unsafe}")
    completed = read_completed(output_path)
    pending = [row for row in trials if row["trial_id"] not in completed]
    if args.max_new_trials is not None:
        pending = pending[: args.max_new_trials]
    print(f"CANONICAL_RESUME phase={args.phase} completed={len(completed)} pending_this_run={len(pending)} expected={expected}", flush=True)
    if not pending:
        print(f"CANONICAL_PHASE_ALREADY_COMPLETE phase={args.phase}")
        return

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.stage import add_reference_to_stage
        from isaacsim.storage.native import get_assets_root_path
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, PhysicsSchemaTools

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        assets = get_assets_root_path()
        if assets is None:
            raise RuntimeError("Isaac asset root unavailable")
        add_reference_to_stage(assets + FRANKA_ASSET, FRANKA_PATH)
        franka = world.scene.add(SingleArticulation(prim_path=FRANKA_PATH, name="formal_franka"))
        obstacle = world.scene.add(FixedCuboid(prim_path=OBSTACLE_PATH, name="formal_obstacle", position=np.zeros(3), scale=np.ones(3) * 0.1))
        PhysxSchema.PhysxContactReportAPI.Apply(world.stage.GetPrimAtPath(OBSTACLE_PATH)).CreateThresholdAttr().Set(0.0)
        world.reset()
        names = list(franka.dof_names)
        if len(names) != 9 or set(names) != set(ARM_NAMES + FINGER_NAMES):
            raise RuntimeError(f"unexpected Franka DOFs: {names}")
        arm_indices = [int(franka.get_dof_index(name)) for name in ARM_NAMES]
        finger_indices = [int(franka.get_dof_index(name)) for name in FINGER_NAMES]
        interface = get_physx_simulation_interface()
        converter = PhysicsSchemaTools.intToSdfPath
        interface.get_contact_report()
        for _ in range(3):
            world.step(render=False)
            interface.get_contact_report()

        output_path.parent.mkdir(parents=True, exist_ok=True)
        exists = output_path.exists() and output_path.stat().st_size > 0
        stream = output_path.open("a", newline="", encoding="utf-8")
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        if not exists:
            writer.writeheader()
            stream.flush()
            os.fsync(stream.fileno())
        started = time.perf_counter()
        run_count = 0
        try:
            for trial in pending:
                result = replay(trial, conditions[trial["condition_id"]]["trajectory_array"], franka, obstacle, world, arm_indices, finger_indices, interface, converter)
                row = {
                    **trial, **result,
                    "actual_obstacle_position_m": json.loads(trial["actual_obstacle_position_m"]),
                    "trajectory_states_replayed": 41, "trajectory_frozen": True,
                    "replanned_after_perturbation": False, "status": "completed", "error": None,
                }
                writer.writerow(serialized(row))
                run_count += 1
                if run_count % 100 == 0 or run_count == len(pending):
                    stream.flush()
                    os.fsync(stream.fileno())
                if run_count % 1000 == 0 or run_count == len(pending):
                    elapsed = time.perf_counter() - started
                    print(f"CANONICAL_PROGRESS phase={args.phase} new={run_count}/{len(pending)} total={len(completed)+run_count}/{expected} wall_s={elapsed:.3f}", flush=True)
        finally:
            stream.close()
        total = len(read_completed(output_path))
        run_meta = {
            "experiment_id": "canonical_formal_rq1_rq2_robustness_v1",
            "phase": args.phase, "execution_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "isaac_sim_version": importlib.metadata.version("isaacsim"), "curobo_commit": "8e734f3",
            "backend": "sequential PhysX", "collision_criterion": "separation <= 0",
            "trajectory_frozen": True, "replanned_after_perturbation": False,
            "frozen_sha256": hashes, "expected_trial_count": expected,
            "completed_trial_count": total, "new_trial_count_this_run": run_count,
            "wall_time_this_run_s": time.perf_counter() - started,
            "checkpoint_flush_interval_trials": 100,
        }
        (OUT / f"{args.phase}_run_status.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
        print(f"CANONICAL_PHASE_RUN_COMPLETE phase={args.phase} total={total}/{expected}", flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()

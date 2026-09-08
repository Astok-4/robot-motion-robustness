#!/usr/bin/env python3
"""Independent PhysX nominal validation for constrained-space calibration paths."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from isaacsim_collision_validation import (
    ARM_JOINT_NAMES,
    COLLISION_SEPARATION_THRESHOLD_M,
    FINGER_JOINT_NAMES,
    FINGER_POSITION_M,
    FRANKA_ASSET,
    FRANKA_PRIM,
    contact_records,
)


os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
LEFT_PRIM = "/World/CorridorLeftBox"
RIGHT_PRIM = "/World/CorridorRightBox"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=root / "results_calibration/rq2_constrained_space_v1")
    parser.add_argument("--config", type=Path, default=root / "configs/experiments/rq2_constrained_space_calibration_v1.json")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    payload = json.loads((args.results_dir / "trajectories.json").read_text(encoding="utf-8"))
    conditions = [row for row in payload["runs"] if row["canonical_run"]]
    if not conditions or len({row["condition_id"] for row in conditions}) != len(conditions):
        raise ValueError("invalid canonical constrained-space trajectories")

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
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Isaac asset root unavailable")
        add_reference_to_stage(assets_root + FRANKA_ASSET, FRANKA_PRIM)
        franka = world.scene.add(SingleArticulation(prim_path=FRANKA_PRIM, name="franka"))
        first_boxes = conditions[0]["boxes"]
        obstacles = []
        for prim_path, name, box in zip((LEFT_PRIM, RIGHT_PRIM), ("corridor_left_box", "corridor_right_box"), first_boxes):
            obstacle = world.scene.add(FixedCuboid(
                prim_path=prim_path, name=name,
                position=np.asarray(box["center_m"], dtype=np.float32),
                scale=np.asarray(box["nominal_size_m"], dtype=np.float32),
                color=np.asarray([0.8, 0.15, 0.15], dtype=np.float32),
            ))
            PhysxSchema.PhysxContactReportAPI.Apply(world.stage.GetPrimAtPath(prim_path)).CreateThresholdAttr().Set(0.0)
            obstacles.append(obstacle)
        world.reset()
        expected = ARM_JOINT_NAMES + FINGER_JOINT_NAMES
        if len(franka.dof_names) != 9 or set(franka.dof_names) != set(expected):
            raise RuntimeError(f"unexpected Franka DOFs: {franka.dof_names}")
        arm_indices = [int(franka.get_dof_index(name)) for name in ARM_JOINT_NAMES]
        finger_indices = [int(franka.get_dof_index(name)) for name in FINGER_JOINT_NAMES]
        interface = get_physx_simulation_interface()
        frame = 0
        results: list[dict[str, Any]] = []
        for condition in conditions:
            if not condition["planning_success"]:
                results.append({
                    "condition_id": condition["condition_id"], "corridor_width_mm": condition["corridor_width_mm"],
                    "inflation_mm": condition["inflation_mm"], "planning_success": False,
                    "nominal_physx_collision": None, "nominal_physx_safe": None,
                    "c_physx_m": None, "minimum_index": None, "first_collision_index": None,
                    "minimum_actor0": None, "minimum_actor1": None, "minimum_collider0": None, "minimum_collider1": None,
                    "trajectory_points_replayed": 0, "status": "skipped_planning_failure", "error": condition["failure_reason"],
                })
                continue
            trajectory = np.asarray(condition["trajectory_positions"], dtype=np.float32)
            if trajectory.shape != (41, 7) or not np.isfinite(trajectory).all():
                raise ValueError(f"invalid trajectory {condition['condition_id']}")
            for obstacle, box in zip(obstacles, condition["boxes"]):
                obstacle.set_world_pose(position=np.asarray(box["center_m"], dtype=np.float32))
            interface.get_contact_report()
            contacts_all: list[tuple[int, int, dict[str, Any]]] = []
            collision_indices: list[int] = []
            for trajectory_index, arm_position in enumerate(trajectory):
                full = np.asarray(franka.get_joint_positions(), dtype=np.float32)
                full[arm_indices] = arm_position
                full[finger_indices] = FINGER_POSITION_M
                franka.set_joint_positions(full)
                franka.set_joint_velocities(np.zeros(9, dtype=np.float32))
                world.step(render=False)
                frame += 1
                contacts = []
                for prim_path in (LEFT_PRIM, RIGHT_PRIM):
                    contacts.extend(contact_records(prim_path, FRANKA_PRIM, interface, PhysicsSchemaTools.intToSdfPath))
                contacts_all.extend((trajectory_index, frame, item) for item in contacts)
                if any(item["minimum_separation_m"] <= COLLISION_SEPARATION_THRESHOLD_M for item in contacts):
                    collision_indices.append(trajectory_index)
            minimum = min(contacts_all, key=lambda item: item[2]["minimum_separation_m"], default=None)
            contact = minimum[2] if minimum else None
            results.append({
                "condition_id": condition["condition_id"], "corridor_width_mm": condition["corridor_width_mm"],
                "inflation_mm": condition["inflation_mm"], "planning_success": True,
                "nominal_physx_collision": bool(collision_indices), "nominal_physx_safe": not bool(collision_indices),
                "c_physx_m": contact["minimum_separation_m"] if contact else None,
                "minimum_index": minimum[0] if minimum else None,
                "first_collision_index": collision_indices[0] if collision_indices else None,
                "minimum_actor0": contact.get("actor0") if contact else None,
                "minimum_actor1": contact.get("actor1") if contact else None,
                "minimum_collider0": contact.get("collider0") if contact else None,
                "minimum_collider1": contact.get("collider1") if contact else None,
                "trajectory_points_replayed": 41, "status": "completed", "error": None,
            })
            print("CONSTRAINED_PHYSX_NOMINAL " + json.dumps({"condition_id": condition["condition_id"], "collision": bool(collision_indices), "separation_m": contact["minimum_separation_m"] if contact else None, "index": minimum[0] if minimum else None}, sort_keys=True), flush=True)
        write_csv(args.results_dir / "physx_nominal_validation.csv", results)
        metadata = {
            "validation_id": "rq2_constrained_space_nominal_physx_v1",
            "conditions_total": len(conditions), "conditions_replayed": sum(r["status"] == "completed" for r in results),
            "trajectory_states_replayed": frame, "collision_criterion": "PhysX contact separation <= 0",
            "evaluation_geometry": "original_uninflated_two_box_geometry",
            "isaac_sim_version": importlib.metadata.version("isaacsim"),
            "input_sha256": {"trajectories.json": sha256(args.results_dir / "trajectories.json"), "config": sha256(args.config)},
            "robustness_perturbation_run": False,
        }
        (args.results_dir / "physx_experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        app.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay prepared constant-bias joint trajectories in PhysX; no planning."""

from __future__ import annotations

import csv
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_pilot/joint_state_perturbation_v0"
FRANKA_ASSET = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
FRANKA_PATH = "/World/Franka"
OBSTACLE_PATH = "/World/PilotObstacle"
ARM_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]


def truth(value: str) -> bool:
    return value.lower() == "true"


def contacts(interface: Any, converter: Any) -> list[dict[str, Any]]:
    headers, data = interface.get_contact_report()
    records = []
    for header in headers:
        paths = {
            "actor0": str(converter(header.actor0)), "actor1": str(converter(header.actor1)),
            "collider0": str(converter(header.collider0)), "collider1": str(converter(header.collider1)),
        }
        values = list(paths.values())
        robot = any(p == FRANKA_PATH or p.startswith(FRANKA_PATH + "/") for p in values)
        obstacle = any(p == OBSTACLE_PATH or p.startswith(OBSTACLE_PATH + "/") for p in values)
        if not robot or not obstacle or header.num_contact_data <= 0:
            continue
        separations = [
            float(data[i].separation)
            for i in range(header.contact_data_offset, header.contact_data_offset + header.num_contact_data)
        ]
        records.append({**paths, "minimum_separation_m": min(separations)})
    return records


def replay(
    trajectory: np.ndarray, franka: Any, obstacle: Any, world: Any,
    arm_indices: list[int], finger_indices: list[int], interface: Any, converter: Any,
) -> dict[str, Any]:
    reports: list[tuple[int, dict[str, Any]]] = []
    collision_indices: list[int] = []
    interface.get_contact_report()
    for state_index, arm in enumerate(trajectory):
        q = np.asarray(franka.get_joint_positions(), dtype=np.float32)
        q[arm_indices] = arm
        q[finger_indices] = 0.04
        franka.set_joint_positions(q)
        franka.set_joint_velocities(np.zeros(9, dtype=np.float32))
        world.step(render=False)
        current = contacts(interface, converter)
        reports.extend((state_index, record) for record in current)
        if any(record["minimum_separation_m"] <= 0.0 for record in current):
            collision_indices.append(state_index)
    minimum = min(reports, key=lambda item: item[1]["minimum_separation_m"], default=None)
    return {
        "physx_collision": bool(collision_indices),
        "physx_min_separation_m": minimum[1]["minimum_separation_m"] if minimum else None,
        "physx_min_separation_state_index": minimum[0] if minimum else None,
        "physx_first_collision_state_index": collision_indices[0] if collision_indices else None,
    }


def main() -> None:
    metadata = json.loads((OUT / "experiment.json").read_text(encoding="utf-8"))
    payload = json.loads((OUT / "prepared_trajectories.json").read_text(encoding="utf-8"))
    with (OUT / "prepared_joint_state_pilot_v0.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if metadata["status"] != "prepared_sanity_pass" or len(rows) != 71:
        raise RuntimeError("prepared sanity gate failed")
    if metadata["motion_planner_invoked"] or metadata["replanned_after_perturbation"]:
        raise RuntimeError("replanning is forbidden")
    obstacle_position = np.asarray(payload["obstacle"]["nominal_position_m"], dtype=np.float32)
    if payload["obstacle"]["size_m"] != [0.1, 0.1, 0.1]:
        raise RuntimeError("nominal obstacle changed")

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
        franka = world.scene.add(SingleArticulation(prim_path=FRANKA_PATH, name="pilot_franka"))
        obstacle = world.scene.add(FixedCuboid(
            prim_path=OBSTACLE_PATH, name="pilot_obstacle",
            position=obstacle_position, scale=np.ones(3, dtype=np.float32) * 0.1,
        ))
        PhysxSchema.PhysxContactReportAPI.Apply(
            world.stage.GetPrimAtPath(OBSTACLE_PATH)
        ).CreateThresholdAttr().Set(0.0)
        world.reset()
        names = list(franka.dof_names)
        if len(names) != 9 or set(names) != set(ARM_NAMES + FINGER_NAMES):
            raise RuntimeError(f"unexpected Franka DOFs: {names}")
        arm_indices = [int(franka.get_dof_index(name)) for name in ARM_NAMES]
        finger_indices = [int(franka.get_dof_index(name)) for name in FINGER_NAMES]
        interface = get_physx_simulation_interface()
        converter = PhysicsSchemaTools.intToSdfPath
        for _ in range(3):
            world.step(render=False)
            interface.get_contact_report()

        final_rows = []
        for row in rows:
            trajectory = np.asarray(payload["trajectories"][row["trial_id"]], dtype=np.float32)
            if trajectory.shape != (41, 7):
                raise RuntimeError(f"bad trajectory: {row['trial_id']}")
            valid = truth(row["valid_joint_limits"])
            physx = replay(
                trajectory, franka, obstacle, world, arm_indices, finger_indices, interface, converter
            ) if valid else {
                "physx_collision": None, "physx_min_separation_m": None,
                "physx_min_separation_state_index": None,
                "physx_first_collision_state_index": None,
            }
            cu_collision = None if row["curobo_collision"] == "" else truth(row["curobo_collision"])
            agreement = (cu_collision == physx["physx_collision"]) if valid else None
            final_rows.append({
                **row, **physx, "engine_agreement": agreement,
                "obstacle_position_unchanged": True,
                "trajectory_frozen_before_bias": True,
                "replanned_after_perturbation": False,
            })
        with (OUT / "joint_state_pilot_v0.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(final_rows[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(final_rows)
        nominal = final_rows[0]
        if nominal["physx_collision"] is not False:
            raise RuntimeError("nominal PhysX sanity check failed")
        metadata.update({
            "status": "execution_complete_pending_summary",
            "isaac_sim_version": importlib.metadata.version("isaacsim"),
            "physx_backend": "sequential discrete-state replay",
            "physx_collision_criterion": "separation <= 0",
            "physx_trial_count": sum(truth(row["valid_joint_limits"]) for row in rows),
            "sanity_checks": {
                **metadata["sanity_checks"],
                "nominal_physx_safe": True,
                "same_prepared_states_used_by_physx": True,
            },
        })
        (OUT / "experiment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print("JOINT_STATE_PILOT_V0_PHYSX_SANITY_PASS replanning=false obstacle_unchanged=true")
    finally:
        app.close()


if __name__ == "__main__":
    main()

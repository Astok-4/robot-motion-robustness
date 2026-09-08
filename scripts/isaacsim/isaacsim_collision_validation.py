#!/usr/bin/env python3
"""Independently replay and collision-check the frozen P0 trajectory in Isaac Sim."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
FINGER_JOINT_NAMES = ["panda_finger_joint1", "panda_finger_joint2"]
FINGER_POSITION_M = 0.04
COLLISION_SEPARATION_THRESHOLD_M = 0.0
FRANKA_PRIM = "/World/Franka"
FRANKA_ASSET = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
OBSTACLE_PRIM = "/World/P0Obstacle"
OBSTACLE_SIZE_M = [0.1, 0.1, 0.1]
NEGATIVE_CONTROL_POSITION_M = [2.0, 2.0, 2.0]
POSITIVE_CONTROL_POSITION_M = [0.0, 0.0, 0.2]
NOMINAL_P0_POSITION_M = [0.3615, 0.0, 0.4245]


def parse_args() -> argparse.Namespace:
    """Parse paths and an output label before starting Isaac Sim."""
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=project_dir / "results_main" / "nominal_trajectory.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "results_isaacsim",
    )
    parser.add_argument("--run-label", default="collision_validation")
    return parser.parse_args()


def load_trajectory(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load and strictly validate the frozen seven-joint trajectory."""
    if not path.is_file():
        raise FileNotFoundError(f"trajectory file does not exist: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "position" not in payload or "joint_names" not in payload:
        raise ValueError("trajectory must be a dict containing position and joint_names")
    position = payload["position"]
    joint_names = list(payload["joint_names"])
    if not isinstance(position, torch.Tensor) or position.ndim != 2 or position.shape[1] != 7:
        raise ValueError(f"trajectory position must have shape (T, 7), got {getattr(position, 'shape', None)}")
    if position.shape[0] != 41:
        raise ValueError(f"expected the frozen P0 trajectory to contain 41 points, got {position.shape[0]}")
    if joint_names != ARM_JOINT_NAMES:
        raise ValueError(f"unexpected trajectory joint names: {joint_names}")
    if not bool(torch.isfinite(position).all()):
        raise ValueError("trajectory contains non-finite values")
    return position.detach().cpu().numpy().astype(np.float32, copy=True), joint_names


def path_string(encoded_path: int, converter: Any) -> str:
    """Convert a PhysX path id to a USD path string."""
    return str(converter(encoded_path))


def contact_records(
    obstacle_path: str,
    robot_path: str,
    simulation_interface: Any,
    path_converter: Any,
) -> list[dict[str, Any]]:
    """Read this step's PhysX contact report and retain obstacle/robot contacts."""
    headers, data = simulation_interface.get_contact_report()
    records: list[dict[str, Any]] = []
    for header in headers:
        actor0 = path_string(header.actor0, path_converter)
        actor1 = path_string(header.actor1, path_converter)
        collider0 = path_string(header.collider0, path_converter)
        collider1 = path_string(header.collider1, path_converter)
        paths = (actor0, actor1, collider0, collider1)
        has_obstacle = any(path == obstacle_path or path.startswith(obstacle_path + "/") for path in paths)
        has_robot = any(path == robot_path or path.startswith(robot_path + "/") for path in paths)
        if not (has_obstacle and has_robot) or header.num_contact_data <= 0:
            continue
        separations = [
            float(data[index].separation)
            for index in range(
                header.contact_data_offset,
                header.contact_data_offset + header.num_contact_data,
            )
        ]
        records.append(
            {
                "actor0": actor0,
                "actor1": actor1,
                "collider0": collider0,
                "collider1": collider1,
                "contact_count": int(header.num_contact_data),
                "minimum_separation_m": min(separations),
            }
        )
    return records


def run_case(
    *,
    test_case: str,
    obstacle_position: list[float],
    trajectory: np.ndarray,
    franka: Any,
    obstacle: Any,
    world: Any,
    arm_indices: list[int],
    finger_indices: list[int],
    simulation_frame: int,
    simulation_interface: Any,
    path_converter: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Replay every trajectory point once and collect PhysX contact reports."""
    obstacle.set_world_pose(position=np.asarray(obstacle_position, dtype=np.float32))
    simulation_interface.get_contact_report()  # discard any report from the prior case

    collision_indices: list[int] = []
    collision_frames: list[int] = []
    contact_report_indices: list[int] = []
    contact_report_frames: list[int] = []
    collision_objects: list[dict[str, Any]] = []
    first_joint_state: list[float] | None = None
    rows: list[dict[str, Any]] = []

    for trajectory_index, arm_position in enumerate(trajectory):
        full_position = np.asarray(franka.get_joint_positions(), dtype=np.float32)
        if full_position.shape != (9,):
            raise RuntimeError(f"expected 9 Franka DOF positions, got {full_position.shape}")
        full_position[arm_indices] = arm_position
        full_position[finger_indices] = FINGER_POSITION_M
        franka.set_joint_positions(full_position)
        franka.set_joint_velocities(np.zeros(9, dtype=np.float32))

        world.step(render=False)
        simulation_frame += 1
        contacts = contact_records(
            OBSTACLE_PRIM,
            FRANKA_PRIM,
            simulation_interface,
            path_converter,
        )
        penetrating_contacts = [
            item
            for item in contacts
            if item["minimum_separation_m"] <= COLLISION_SEPARATION_THRESHOLD_M
        ]
        contact_reported = bool(contacts)
        collided = bool(penetrating_contacts)
        if contact_reported:
            contact_report_indices.append(trajectory_index)
            contact_report_frames.append(simulation_frame)
        if collided:
            collision_indices.append(trajectory_index)
            collision_frames.append(simulation_frame)
            collision_objects.extend(penetrating_contacts)
            if first_joint_state is None:
                first_joint_state = [float(value) for value in arm_position]
        rows.append(
            {
                "test_case": test_case,
                "trajectory_index": trajectory_index,
                "simulation_frame": simulation_frame,
                "contact_report_detected": contact_reported,
                "collision_detected": collided,
                "minimum_contact_separation_m": min(
                    (item["minimum_separation_m"] for item in contacts),
                    default=None,
                ),
                "contact_objects": json.dumps(contacts, separators=(",", ":")),
            }
        )

    unique_objects = list({json.dumps(item, sort_keys=True): item for item in collision_objects}.values())
    result = {
        "test_case": test_case,
        "obstacle_position_m": obstacle_position,
        "obstacle_size_m": OBSTACLE_SIZE_M,
        "collision_criterion": "minimum PhysX contact separation <= 0.0 m",
        "contact_report_detected": bool(contact_report_indices),
        "contact_report_trajectory_indices": contact_report_indices,
        "contact_report_simulation_frames": contact_report_frames,
        "collision_detected": bool(collision_indices),
        "first_collision_trajectory_index": collision_indices[0] if collision_indices else None,
        "first_collision_simulation_frame": collision_frames[0] if collision_frames else None,
        "collision_trajectory_indices": collision_indices,
        "collision_simulation_frames": collision_frames,
        "joint_state_at_first_collision": first_joint_state,
        "contact_objects": unique_objects,
        "trajectory_points_replayed": int(trajectory.shape[0]),
        "status": "completed",
        "error": None,
    }
    print(
        "CASE_RESULT "
        + json.dumps(
            {
                "test_case": test_case,
                "collision_detected": result["collision_detected"],
                "first_collision_trajectory_index": result["first_collision_trajectory_index"],
                "first_collision_simulation_frame": result["first_collision_simulation_frame"],
                "contact_report_frames": len(contact_report_frames),
                "collision_frames": len(collision_frames),
            },
            sort_keys=True,
        )
    )
    return result, rows, simulation_frame


def write_results(output_dir: Path, run_label: str, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write one structured JSON summary and one per-trajectory-point CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{run_label}.json"
    csv_path = output_dir / f"{run_label}_frames.csv"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"RESULT_JSON {json_path}")
    print(f"RESULT_CSV {csv_path}")


def main() -> None:
    """Build the scene, run detector controls, then validate nominal P0."""
    args = parse_args()
    trajectory_path = args.trajectory.resolve()
    trajectory, trajectory_joint_names = load_trajectory(trajectory_path)
    print(f"TRAJECTORY_LOADED path={trajectory_path} shape={trajectory.shape}")

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": True})
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
            raise RuntimeError("Isaac Sim asset server is unavailable")
        add_reference_to_stage(assets_root + FRANKA_ASSET, FRANKA_PRIM)
        franka = world.scene.add(SingleArticulation(prim_path=FRANKA_PRIM, name="franka"))
        obstacle = world.scene.add(
            FixedCuboid(
                prim_path=OBSTACLE_PRIM,
                name="p0_obstacle",
                position=np.asarray(NEGATIVE_CONTROL_POSITION_M, dtype=np.float32),
                scale=np.asarray(OBSTACLE_SIZE_M, dtype=np.float32),
                color=np.asarray([0.8, 0.1, 0.1], dtype=np.float32),
            )
        )
        obstacle_prim = world.stage.GetPrimAtPath(OBSTACLE_PRIM)
        if not obstacle_prim.IsValid():
            raise RuntimeError(f"obstacle prim was not created: {OBSTACLE_PRIM}")
        PhysxSchema.PhysxContactReportAPI.Apply(obstacle_prim).CreateThresholdAttr().Set(0.0)
        world.reset()

        actual_dof_names = list(franka.dof_names)
        expected_dof_names = ARM_JOINT_NAMES + FINGER_JOINT_NAMES
        if len(actual_dof_names) != 9 or set(actual_dof_names) != set(expected_dof_names):
            raise RuntimeError(
                f"unexpected Franka DOF names; expected {expected_dof_names}, got {actual_dof_names}"
            )
        joint_mapping = {name: int(franka.get_dof_index(name)) for name in trajectory_joint_names}
        finger_mapping = {name: int(franka.get_dof_index(name)) for name in FINGER_JOINT_NAMES}
        if any(index < 0 for index in [*joint_mapping.values(), *finger_mapping.values()]):
            raise RuntimeError(f"failed to resolve all Franka DOF names: {actual_dof_names}")
        arm_indices = [joint_mapping[name] for name in trajectory_joint_names]
        finger_indices = [finger_mapping[name] for name in FINGER_JOINT_NAMES]
        print("JOINT_MAPPING " + json.dumps(joint_mapping, sort_keys=True))
        print("FINGER_MAPPING " + json.dumps(finger_mapping, sort_keys=True))

        simulation_interface = get_physx_simulation_interface()
        if not hasattr(simulation_interface, "get_contact_report"):
            raise RuntimeError("Isaac Sim PhysX get_contact_report API is unavailable")

        cases: list[dict[str, Any]] = []
        all_rows: list[dict[str, Any]] = []
        simulation_frame = 0
        for test_case, position in (
            ("negative_control", NEGATIVE_CONTROL_POSITION_M),
            ("positive_control", POSITIVE_CONTROL_POSITION_M),
            ("nominal_p0", NOMINAL_P0_POSITION_M),
        ):
            result, rows, simulation_frame = run_case(
                test_case=test_case,
                obstacle_position=position,
                trajectory=trajectory,
                franka=franka,
                obstacle=obstacle,
                world=world,
                arm_indices=arm_indices,
                finger_indices=finger_indices,
                simulation_frame=simulation_frame,
                simulation_interface=simulation_interface,
                path_converter=PhysicsSchemaTools.intToSdfPath,
            )
            cases.append(result)
            all_rows.extend(rows)
            if test_case == "negative_control" and result["collision_detected"]:
                raise RuntimeError("negative control unexpectedly detected a collision")
            if test_case == "positive_control" and not result["collision_detected"]:
                raise RuntimeError("positive control did not detect a collision")

        payload = {
            "schema_version": 1,
            "status": "completed",
            "error": None,
            "run_label": args.run_label,
            "trajectory_file": "results_main/nominal_trajectory.pt",
            "trajectory_shape": list(trajectory.shape),
            "trajectory_joint_names": trajectory_joint_names,
            "isaac_sim_dof_names": actual_dof_names,
            "joint_mapping": joint_mapping,
            "finger_joint_mapping": finger_mapping,
            "finger_position_m": FINGER_POSITION_M,
            "simulation_frames_per_trajectory_point": 1,
            "total_validation_simulation_frames": simulation_frame,
            "isaac_sim_version": importlib.metadata.version("isaacsim"),
            "collision_api": (
                "pxr.PhysxSchema.PhysxContactReportAPI + "
                "omni.physx.get_physx_simulation_interface().get_contact_report"
            ),
            "collision_separation_threshold_m": COLLISION_SEPARATION_THRESHOLD_M,
            "cases": cases,
        }
        write_results(args.output_dir, args.run_label, payload, all_rows)
        print("ISAACSIM_COLLISION_VALIDATION_PASS")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run formal coarse or adaptive refinement joint-bias replay in PhysX."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from run_joint_state_perturbation_pilot_v0 import (
    ARM_NAMES, FINGER_NAMES, FRANKA_ASSET, FRANKA_PATH, replay, truth,
)

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_formal/joint_state_perturbation_v1"
MATRIX = ROOT / "configs/experiments/formal_safety_margin_matrix_v1.json"
OBSTACLE_PATH = "/World/PilotObstacle"


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["coarse", "refinement"], required=True)
    args = parser.parse_args()
    metadata = json.loads((OUT / "experiment.json").read_text(encoding="utf-8"))
    if metadata["motion_planner_invoked"] or metadata["replanned_after_perturbation"]:
        raise RuntimeError("replanning forbidden")
    prepared = read_csv(OUT / "prepared_coarse_trials.csv")
    payload = json.loads((OUT / "prepared_coarse_trajectories.json").read_text(encoding="utf-8"))
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    zero = {c["scene_id"]: np.asarray(c["trajectory"]["position"], dtype=np.float32)
            for c in matrix["conditions"] if c["inflation_mm"] == 0}

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

        world = World(stage_units_in_meters=1.0); world.scene.add_default_ground_plane()
        assets = get_assets_root_path()
        if assets is None: raise RuntimeError("Isaac asset root unavailable")
        add_reference_to_stage(assets + FRANKA_ASSET, FRANKA_PATH)
        franka = world.scene.add(SingleArticulation(prim_path=FRANKA_PATH, name="formal_joint_franka"))
        obstacle = world.scene.add(FixedCuboid(
            prim_path=OBSTACLE_PATH, name="formal_joint_obstacle",
            position=np.zeros(3,dtype=np.float32), scale=np.ones(3,dtype=np.float32)*0.1,
        ))
        PhysxSchema.PhysxContactReportAPI.Apply(world.stage.GetPrimAtPath(OBSTACLE_PATH)).CreateThresholdAttr().Set(0.0)
        world.reset()
        if set(franka.dof_names) != set(ARM_NAMES + FINGER_NAMES):
            raise RuntimeError("unexpected Franka DOFs")
        arm_indices = [int(franka.get_dof_index(n)) for n in ARM_NAMES]
        finger_indices = [int(franka.get_dof_index(n)) for n in FINGER_NAMES]
        interface = get_physx_simulation_interface(); converter = PhysicsSchemaTools.intToSdfPath
        for _ in range(3): world.step(render=False); interface.get_contact_report()

        output_rows = []
        if args.phase == "coarse":
            for row in prepared:
                scene = row["scene_id"]
                obstacle.set_world_pose(position=np.asarray(payload["scene_obstacles"][scene]["nominal_position_m"],dtype=np.float32))
                valid = truth(row["joint_limit_valid"])
                result = replay(np.asarray(payload["trajectories"][row["trial_id"]],dtype=np.float32),
                                franka,obstacle,world,arm_indices,finger_indices,interface,converter) if valid else {
                    "physx_collision": None,"physx_min_separation_m": None,
                    "physx_min_separation_state_index": None,"physx_first_collision_state_index": None}
                cu = truth(row["curobo_collision"]) if row["curobo_collision"] else None
                output_rows.append({**row,**result,
                    "engine_agreement": (cu == result["physx_collision"]) if valid else None,
                    "obstacle_unchanged": True,"trajectory_frozen": True,
                    "replanned_after_perturbation": False})
            nominal = [r for r in output_rows if float(r["magnitude_deg"]) == 0]
            if len(output_rows) != 396 or len(nominal) != 4 or any(r["physx_collision"] for r in nominal):
                raise RuntimeError("coarse/nominal audit failed")
            write_csv(OUT / "coarse_trials.csv", output_rows)
            print("FORMAL_JOINT_STATE_V1_COARSE_PASS rows=396")
        else:
            coarse = read_csv(OUT / "coarse_trials.csv")
            groups: dict[tuple[str,str],list[dict[str,str]]] = defaultdict(list)
            for row in coarse:
                if float(row["magnitude_deg"]) > 0: groups[(row["scene_id"],row["direction_id"])].append(row)
            direction_info = {(r["scene_id"],r["direction_id"]):r for r in coarse if float(r["magnitude_deg"])>0}
            limits = payload["joint_limits_rad"]
            for key in sorted(groups):
                values = sorted(groups[key], key=lambda r:float(r["magnitude_deg"]))
                prior_mag, prior_collision = 0.0, False
                bracket = None
                for row in values:
                    collision = truth(row["physx_collision"])
                    magnitude = float(row["magnitude_deg"])
                    if not prior_collision and collision:
                        bracket = [prior_mag,magnitude]; break
                    prior_mag,prior_collision = magnitude,collision
                if bracket is None: continue
                info = direction_info[key]; scene = key[0]
                joint_index = int(info["joint_index"]); joint_name = info["joint_name"]
                axis = 1.0 if info["sign"] == "positive" else -1.0
                obstacle.set_world_pose(position=np.asarray(payload["scene_obstacles"][scene]["nominal_position_m"],dtype=np.float32))
                while bracket[1] - bracket[0] > 0.1 + 1e-12:
                    magnitude = (bracket[0] + bracket[1]) / 2.0
                    trajectory = zero[scene].copy(); trajectory[:,joint_index] += axis*math.radians(magnitude)
                    lower,upper = limits[joint_name]
                    invalid = np.flatnonzero((trajectory[:,joint_index]<lower)|(trajectory[:,joint_index]>upper))
                    if len(invalid): raise RuntimeError(f"refinement joint-limit invalid: {key} {magnitude}")
                    result = replay(trajectory,franka,obstacle,world,arm_indices,finger_indices,interface,converter)
                    trial_id = f"{scene}__{info['direction_id']}__refine_{magnitude:08.5f}deg"
                    output_rows.append({
                        "trial_id":trial_id,"scene_id":scene,"direction_id":info["direction_id"],
                        "joint_index":joint_index,"joint_name":joint_name,"sign":info["sign"],
                        "magnitude_deg":magnitude,"magnitude_rad":math.radians(magnitude),
                        "joint_limit_valid":True,"first_invalid_state_index":"",**result,
                        "obstacle_unchanged":True,"trajectory_frozen":True,
                        "replanned_after_perturbation":False})
                    bracket[1 if result["physx_collision"] else 0] = magnitude
            if output_rows: write_csv(OUT / "physx_refinement_trials.csv", output_rows)
            print(f"FORMAL_JOINT_STATE_V1_REFINEMENT_PASS trials={len(output_rows)}")
    finally:
        app.close()


if __name__ == "__main__": main()

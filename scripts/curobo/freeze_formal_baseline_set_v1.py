#!/usr/bin/env python3
"""Freeze four independently validated scenes as Formal Baseline Set V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MULTISCENE_IDS = [
    "multiscene_candidate_001",
    "multiscene_candidate_005",
    "multiscene_candidate_006",
]


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-baseline-v1",
        type=Path,
        default=project_dir / "configs" / "experiments" / "formal_baseline_v1.json",
    )
    parser.add_argument(
        "--multiscene-calibration",
        type=Path,
        default=project_dir
        / "results_calibration"
        / "multiscene_candidates"
        / "scenario_candidates.json",
    )
    parser.add_argument(
        "--multiscene-isaac-validation",
        type=Path,
        default=project_dir
        / "results_calibration"
        / "multiscene_candidates"
        / "isaacsim_validation"
        / "candidate_validation.json",
    )
    parser.add_argument(
        "--v1-local-danger-scan",
        type=Path,
        default=project_dir
        / "results_pilot"
        / "local_danger_direction"
        / "local_danger_direction_scan.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "configs" / "experiments" / "formal_baseline_set_v1.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trajectory(trajectory: Any, label: str) -> list[list[float]]:
    if (
        not isinstance(trajectory, list)
        or len(trajectory) != 41
        or any(not isinstance(row, list) or len(row) != 7 for row in trajectory)
        or any(not math.isfinite(float(value)) for row in trajectory for value in row)
    ):
        raise ValueError(f"{label}: trajectory is not finite 41x7")
    return trajectory


def build_v1_scene(
    baseline: dict[str, Any], danger_scan: dict[str, Any]
) -> dict[str, Any]:
    if baseline.get("baseline_id") != "formal_baseline_v1" or baseline.get("baseline_status") != "FROZEN":
        raise ValueError("Formal Baseline V1 is not the expected frozen baseline")
    trajectory = validate_trajectory(baseline["nominal_trajectory"]["position"], "formal_baseline_v1")
    geometry = danger_scan.get("geometry", {})
    if geometry.get("trajectory_index") != baseline["curobo_validation"][
        "minimum_clearance_trajectory_index"
    ]:
        raise ValueError("V1 local danger scan does not match frozen minimum index")
    direction = geometry.get("obstacle_toward_sphere_translation_direction")
    if direction != [0.0, 0.0, -1.0]:
        raise ValueError("unexpected Formal Baseline V1 local danger direction")
    return {
        "scene_id": "formal_baseline_v1",
        "source_candidate": baseline["source_candidate"],
        "robot": baseline["robot"],
        "joint_names": baseline["joint_names"],
        "start_joint_configuration": baseline["start_joint_state"],
        "goal_joint_configuration": baseline["goal_joint_state"],
        "obstacle": baseline["obstacle"],
        "nominal_trajectory": {
            "shape": [41, 7],
            "position": trajectory,
            "frozen": True,
        },
        "trajectory_frozen": True,
        "curobo_nominal": {
            "minimum_clearance_m": baseline["curobo_validation"]["minimum_clearance_m"],
            "minimum_clearance_trajectory_index": baseline["curobo_validation"][
                "minimum_clearance_trajectory_index"
            ],
            "collision": baseline["curobo_validation"]["nominal_collision"],
        },
        "isaac_nominal": {
            "minimum_separation_m": baseline["isaac_sim_validation"][
                "minimum_reported_separation_m"
            ],
            "minimum_separation_trajectory_index": baseline["isaac_sim_validation"][
                "minimum_separation_trajectory_index"
            ],
            "collision": baseline["isaac_sim_validation"]["nominal_collision"],
        },
        "bottleneck": {
            "sphere_index": geometry["sphere_index"],
            "robot_link": "panda_link6",
            "local_danger_direction": direction,
        },
    }


def build_multiscene(
    calibration: dict[str, Any], isaac: dict[str, Any]
) -> list[dict[str, Any]]:
    calibration_by_id = {
        item.get("candidate_id"): item for item in calibration.get("candidates", [])
    }
    isaac_by_id = {item.get("candidate_id"): item for item in isaac.get("candidates", [])}
    scenes: list[dict[str, Any]] = []
    for candidate_id in MULTISCENE_IDS:
        if candidate_id not in calibration_by_id or candidate_id not in isaac_by_id:
            raise ValueError(f"missing verified source for {candidate_id}")
        candidate = calibration_by_id[candidate_id]
        validation = isaac_by_id[candidate_id]
        trajectory = validate_trajectory(candidate.get("trajectory_positions"), candidate_id)
        if (
            not candidate.get("planning_success")
            or candidate.get("nominal_collision") is not False
            or validation.get("status") != "completed"
            or validation.get("isaac_collision_detected") is not False
            or validation.get("baseline_set_eligible") is not True
        ):
            raise ValueError(f"{candidate_id} has not passed both nominal validators")
        if candidate["obstacle_position_m"] != validation["obstacle_position_m"]:
            raise ValueError(f"{candidate_id}: obstacle position differs across sources")
        if candidate["obstacle_size_m"] != validation["obstacle_size_m"]:
            raise ValueError(f"{candidate_id}: obstacle size differs across sources")
        scenes.append(
            {
                "scene_id": candidate_id,
                "source_candidate": candidate_id,
                "robot": "Franka Panda",
                "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
                "start_joint_configuration": candidate["start_joint_state"],
                "goal_joint_configuration": candidate["goal_joint_state"],
                "obstacle": {
                    "type": "axis_aligned_box",
                    "nominal_position_m": candidate["obstacle_position_m"],
                    "size_m": candidate["obstacle_size_m"],
                },
                "nominal_trajectory": {
                    "shape": [41, 7],
                    "position": trajectory,
                    "frozen": True,
                },
                "trajectory_frozen": True,
                "curobo_nominal": {
                    "minimum_clearance_m": candidate["minimum_clearance_m"],
                    "minimum_clearance_trajectory_index": candidate[
                        "minimum_clearance_trajectory_index"
                    ],
                    "collision": candidate["nominal_collision"],
                },
                "isaac_nominal": {
                    "minimum_separation_m": validation[
                        "isaac_minimum_reported_separation_m"
                    ],
                    "minimum_separation_trajectory_index": validation[
                        "isaac_minimum_separation_trajectory_index"
                    ],
                    "collision": validation["isaac_collision_detected"],
                },
                "bottleneck": {
                    "sphere_index": candidate["minimum_sphere_index"],
                    "robot_link": candidate["minimum_sphere_link"],
                    "local_danger_direction": candidate["danger_direction"],
                },
            }
        )
    return scenes


def main() -> None:
    args = parse_args()
    source_paths = {
        "formal_baseline_v1": args.formal_baseline_v1.resolve(),
        "multiscene_calibration": args.multiscene_calibration.resolve(),
        "multiscene_isaac_validation": args.multiscene_isaac_validation.resolve(),
        "v1_local_danger_scan": args.v1_local_danger_scan.resolve(),
    }
    baseline = load_json(source_paths["formal_baseline_v1"])
    calibration = load_json(source_paths["multiscene_calibration"])
    isaac = load_json(source_paths["multiscene_isaac_validation"])
    danger_scan = load_json(source_paths["v1_local_danger_scan"])
    scenes = [build_v1_scene(baseline, danger_scan), *build_multiscene(calibration, isaac)]
    if [scene["scene_id"] for scene in scenes] != ["formal_baseline_v1", *MULTISCENE_IDS]:
        raise RuntimeError("unexpected scene order")
    payload = {
        "schema_version": 1,
        "baseline_set_id": "formal_baseline_set_v1",
        "baseline_set_status": "FROZEN",
        "scene_count": 4,
        "trajectory_frozen": True,
        "replanning_allowed_during_robustness_evaluation": False,
        "scenes": scenes,
        "provenance": {
            key: {
                "file": str(path.relative_to(Path(__file__).resolve().parents[2])),
                "sha256": sha256(path),
            }
            for key, path in source_paths.items()
        },
        "interpretation": (
            "Four public synthetic Franka single-box baselines for preliminary "
            "trajectory-robustness experiments; not a real-robot collision probability model."
        ),
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to overwrite a different frozen baseline set: {args.output}")
        print(f"FORMAL_BASELINE_SET_V1_VERIFIED {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(f"FORMAL_BASELINE_SET_V1_FROZEN {args.output}")


if __name__ == "__main__":
    main()

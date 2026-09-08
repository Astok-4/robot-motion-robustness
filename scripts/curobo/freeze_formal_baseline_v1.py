#!/usr/bin/env python3
"""Freeze verified calibration candidate_001 as Formal Baseline V1."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


SOURCE_CANDIDATE = "candidate_001"
BASELINE_ID = "formal_baseline_v1"
ARM_JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]


def parse_args() -> argparse.Namespace:
    """Parse verified source and frozen output paths."""
    project_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calibration-results",
        type=Path,
        default=project_dir / "results_calibration" / "scenario_candidates.json",
    )
    parser.add_argument(
        "--isaac-validation",
        type=Path,
        default=project_dir
        / "results_calibration"
        / "isaacsim_validation"
        / "candidate_validation.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "configs" / "experiments" / "formal_baseline_v1.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    """Load one required JSON object."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def sha256(path: Path) -> str:
    """Hash a source artifact for provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(project_dir: Path, calibration_path: Path, isaac_path: Path) -> dict[str, Any]:
    """Extract candidate_001 only after validating both engine records."""
    calibration = load_json(calibration_path)
    isaac = load_json(isaac_path)
    matches = [
        item for item in calibration.get("candidates", []) if item.get("candidate_id") == SOURCE_CANDIDATE
    ]
    isaac_matches = [
        item for item in isaac.get("candidates", []) if item.get("candidate_id") == SOURCE_CANDIDATE
    ]
    if len(matches) != 1 or len(isaac_matches) != 1:
        raise ValueError("candidate_001 must occur exactly once in each verified source")
    candidate = matches[0]
    isaac_candidate = isaac_matches[0]
    trajectory = candidate.get("trajectory_positions")
    if not isinstance(trajectory, list) or len(trajectory) != 41 or any(len(row) != 7 for row in trajectory):
        raise ValueError("candidate_001 trajectory is not 41x7")
    if not candidate.get("planning_success") or candidate.get("nominal_collision") is not False:
        raise ValueError("candidate_001 is not a successful collision-free cuRobo candidate")
    if isaac_candidate.get("status") != "completed" or isaac_candidate.get("isaac_collision_detected") is not False:
        raise ValueError("candidate_001 is not a completed collision-free Isaac Sim validation")
    if candidate["box_center_m"] != isaac_candidate["obstacle_position_m"]:
        raise ValueError("candidate_001 obstacle position differs between source artifacts")
    if candidate["box_dims_m"] != isaac_candidate["obstacle_size_m"]:
        raise ValueError("candidate_001 obstacle size differs between source artifacts")

    curobo_commit = subprocess.check_output(
        ["git", "-C", str(project_dir / "curobo"), "rev-parse", "HEAD"], text=True
    ).strip()
    return {
        "schema_version": 1,
        "baseline_id": BASELINE_ID,
        "baseline_status": "FROZEN",
        "source_candidate": SOURCE_CANDIDATE,
        "robot": "Franka Panda",
        "joint_names": ARM_JOINT_NAMES,
        "start_joint_state": candidate["start_q"],
        "goal_joint_state": candidate["goal_q"],
        "obstacle": {
            "type": "axis_aligned_box",
            "nominal_position_m": candidate["box_center_m"],
            "size_m": candidate["box_dims_m"],
        },
        "nominal_trajectory": {
            "shape": [41, 7],
            "position": trajectory,
            "frozen": True,
            "replanning_allowed_during_robustness_evaluation": False,
        },
        "curobo_validation": {
            "commit": curobo_commit,
            "minimum_clearance_m": candidate["minimum_clearance_m"],
            "minimum_clearance_trajectory_index": candidate[
                "minimum_clearance_trajectory_index"
            ],
            "nominal_collision": False,
        },
        "isaac_sim_validation": {
            "version": isaac.get("isaac_sim_version"),
            "minimum_reported_separation_m": isaac_candidate[
                "isaac_minimum_reported_separation_m"
            ],
            "minimum_separation_trajectory_index": isaac_candidate[
                "isaac_minimum_separation_trajectory_index"
            ],
            "nominal_collision": False,
            "collision_criterion": isaac.get("collision_criterion"),
        },
        "provenance": {
            "calibration_source": "results_calibration/scenario_candidates.json",
            "calibration_source_sha256": sha256(calibration_path),
            "isaac_validation_source": (
                "results_calibration/isaacsim_validation/candidate_validation.json"
            ),
            "isaac_validation_source_sha256": sha256(isaac_path),
        },
        "interpretation": (
            "Public synthetic baseline for preliminary trajectory-robustness experiments; "
            "not a real-robot collision probability model."
        ),
    }


def main() -> None:
    """Create the frozen file once, or verify an identical existing copy."""
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[2]
    payload = build_payload(
        project_dir,
        args.calibration_results.resolve(),
        args.isaac_validation.resolve(),
    )
    serialized = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output.exists():
        if args.output.read_text(encoding="utf-8") != serialized:
            raise RuntimeError(f"refusing to overwrite a different frozen baseline: {args.output}")
        print(f"FORMAL_BASELINE_V1_VERIFIED {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized, encoding="utf-8")
    print(f"FORMAL_BASELINE_V1_FROZEN {args.output}")


if __name__ == "__main__":
    main()

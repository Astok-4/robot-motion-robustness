#!/usr/bin/env python3
"""Finalize and audit the frozen Formal RQ2 Planning-Time Benchmark V1."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_formal/rq2_planning_time_v1"
BASELINE = ROOT / "configs/experiments/formal_baseline_set_v1.json"
MATRIX = ROOT / "configs/experiments/formal_safety_margin_matrix_v1.json"
PROTOCOL = ROOT / "configs/experiments/formal_rq1_rq2_protocol_v1.json"
EXPECTED = {
    BASELINE: "3a3b4806e2a407253006be509063ccf4b584c873bc5e8e8f9110949de94e8f6b",
    MATRIX: "8869b9709cf6c84274d77e2d28a2bf8354eb15f3e3740362da0b2d8871880d04",
    PROTOCOL: "eb270c2219460b44a1e9ff445b66dc40b18d1f4690383c66d6478419dba3022f",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truth(value: str) -> bool:
    return value.lower() == "true"


def stats(values: list[float], prefix: str) -> dict[str, float | None]:
    if not values:
        return {f"{prefix}_{name}_s": None for name in ["median", "q1", "q3", "iqr", "minimum", "maximum"]}
    q1, median, q3 = np.percentile(np.asarray(values), [25, 50, 75], method="linear")
    return {
        f"{prefix}_median_s": float(median), f"{prefix}_q1_s": float(q1),
        f"{prefix}_q3_s": float(q3), f"{prefix}_iqr_s": float(q3-q1),
        f"{prefix}_minimum_s": min(values), f"{prefix}_maximum_s": max(values),
    }


def main() -> None:
    actual = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in EXPECTED}
    for path, expected in EXPECTED.items():
        if actual[str(path.relative_to(ROOT))] != expected:
            raise RuntimeError(f"frozen asset SHA mismatch: {path}")
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    conditions = {row["condition_id"]: row for row in matrix["conditions"]}
    timing = protocol["planning_timing_protocol"]
    warmup, measured = read(OUT / "warmup.csv"), read(OUT / "measured_runs.csv")
    if len(warmup) != 20 or len(measured) != 600:
        raise RuntimeError(f"incomplete timing population warmup={len(warmup)} measured={len(measured)}")
    if len({r["attempt_id"] for r in warmup}) != 20 or len({r["attempt_id"] for r in measured}) != 600:
        raise RuntimeError("duplicate attempt identity")
    expected_warmup = [f"warmup_{i:02d}__{cid}" for i, cid in enumerate(timing["warmup_schedule"], start=1)]
    if [r["attempt_id"] for r in warmup] != expected_warmup:
        raise RuntimeError("warmup order mismatch")
    expected_measured = []
    global_position = 0
    for schedule in timing["execution_schedule"]:
        round_id = int(schedule["round_index"])
        for position, condition_id in enumerate(schedule["condition_order"], start=1):
            global_position += 1
            expected_measured.append((
                f"round_{round_id:02d}__position_{position:02d}__{condition_id}",
                round_id, global_position, position, condition_id,
            ))
    for row, expected in zip(measured, expected_measured):
        attempt_id, round_id, global_position, position, condition_id = expected
        if (row["attempt_id"] != attempt_id or int(row["round_id"]) != round_id or
            int(row["execution_position"]) != global_position or
            int(row["position_within_round"]) != position or row["condition_id"] != condition_id):
            raise RuntimeError(f"measured execution order mismatch at {global_position}")
    counts = Counter(row["condition_id"] for row in measured)
    if set(counts) != set(conditions) or any(value != 30 for value in counts.values()):
        raise RuntimeError("per-condition attempt count audit failed")

    summary = []
    for condition_id, condition in conditions.items():
        rows = [row for row in measured if row["condition_id"] == condition_id]
        successes = [row for row in rows if truth(row["planning_success"])]
        solve = [float(row["planning_reported_solve_time_s"]) for row in successes if row["planning_reported_solve_time_s"]]
        wall = [float(row["measured_wall_time_s"]) for row in successes if row["measured_wall_time_s"]]
        summary.append({
            "condition_id": condition_id, "scene_id": condition["scene_id"],
            "inflation_mm": condition["inflation_mm"], "measured_attempts": 30,
            "success_count": len(successes), "failure_count": 30-len(successes),
            "success_rate": len(successes)/30, "c_curobo_m": condition["c_curobo_m"],
            "c_physx_m": condition["c_physx_m"],
            "frozen_trajectory_length_rad": condition["trajectory_length_rad"],
            **stats(solve, "planner_solve"), **stats(wall, "wall_time"),
        })
    write(OUT / "condition_summary.csv", summary)
    consistency_fields = [
        "attempt_id", "round_id", "execution_position", "condition_id", "scene_id",
        "inflation_mm", "planning_success", "trajectory_shape", "trajectory_finite",
        "maximum_absolute_joint_difference_rad", "mean_absolute_joint_difference_rad",
        "trajectory_length_rad", "frozen_trajectory_length_rad",
        "trajectory_length_difference_rad", "exact_frozen_trajectory_match",
        "tolerance_frozen_trajectory_match_1e_7_rad",
    ]
    consistency = [{field: row[field] for field in consistency_fields} for row in measured]
    write(OUT / "trajectory_consistency.csv", consistency)

    failures = [row for row in measured if not truth(row["planning_success"])]
    invalid_success = [row for row in measured if truth(row["planning_success"]) and (not truth(row["trajectory_finite"]) or not row["trajectory_shape"])]
    exact_matches = sum(truth(row["exact_frozen_trajectory_match"]) for row in measured if truth(row["planning_success"]))
    tolerance_matches = sum(truth(row["tolerance_frozen_trajectory_match_1e_7_rad"]) for row in measured if truth(row["planning_success"]))
    execution = json.loads((OUT / "execution_status.json").read_text(encoding="utf-8"))
    audit = {
        "status": "PASS", "audit_label": "FORMAL_RQ2_PLANNING_TIME_V1_AUDIT_PASS",
        "warmup_attempt_count": 20, "warmup_success_count": sum(truth(r["planning_success"]) for r in warmup),
        "measured_attempt_count": 600, "condition_count": 20,
        "attempts_per_condition": 30, "measured_success_count": 600-len(failures),
        "measured_failure_count": len(failures), "invalid_success_count": len(invalid_success),
        "execution_order_matches_frozen_protocol": True,
        "duplicate_attempt_count": 0, "missing_scheduled_attempt_count": 0,
        "failures_retained": True, "exact_frozen_trajectory_match_count": exact_matches,
        "tolerance_frozen_trajectory_match_count": tolerance_matches,
        "frozen_sha256": actual, "robustness_statistical_analysis_run": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    experiment = {
        **execution, "status": "completed_audit_pass", "audit_label": audit["audit_label"],
        "condition_summary_file": "condition_summary.csv",
        "measured_runs_file": "measured_runs.csv",
        "trajectory_consistency_file": "trajectory_consistency.csv",
    }
    (OUT / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print("FORMAL_RQ2_PLANNING_TIME_V1_AUDIT_PASS")


if __name__ == "__main__":
    main()

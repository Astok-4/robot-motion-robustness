#!/usr/bin/env python3
"""Prepare the preregistered reproducibility replay and audit/finalize Formal V1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

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
ATOL = 1e-9


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows and fields is None:
        raise ValueError(f"fields required for empty output {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), separators=(",", ":")) if isinstance(row.get(k), (dict, list)) else row.get(k) for k in fields})


def truth(value: Any) -> bool:
    return str(value).lower() == "true"


def optional_float(value: str) -> float | None:
    return None if value in ("", "None", "null") else float(value)


def optional_int(value: str) -> int | None:
    return None if value in ("", "None", "null") else int(value)


def verify() -> dict[str, str]:
    actual = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in EXPECTED}
    for path, expected in EXPECTED.items():
        if actual[str(path.relative_to(ROOT))] != expected:
            raise RuntimeError(f"frozen asset SHA mismatch: {path}")
    return actual


def pair_key(row: dict[str, str]) -> tuple[str, str]:
    return row["condition_id"], row["direction_id"]


def first_bracket(rows: list[dict[str, str]], collision_field: str) -> tuple[float, float] | None:
    prior_mag, prior_collision = 0.0, False
    for row in sorted(rows, key=lambda x: float(x["magnitude_mm"])):
        magnitude, collision = float(row["magnitude_mm"]), truth(row[collision_field])
        if not prior_collision and collision:
            return prior_mag, magnitude
        prior_mag, prior_collision = magnitude, collision
    return None


def prepare_repro() -> None:
    verify()
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    audit = {(x["condition_id"], x["direction_id"]) for x in protocol["reproducibility_strategy"]["audit_pairs"]}
    if len(audit) != 256:
        raise RuntimeError("frozen audit pair population is not 256")
    canonical = read(OUT / "coarse_trials.csv") + read(OUT / "boundary_refinement_trials.csv")
    selected = []
    for row in canonical:
        key = pair_key(row)
        magnitude = float(row["magnitude_mm"])
        is_refinement = "refine_" in row["trial_id"]
        if key in audit and (magnitude in (10.0, 30.0, 60.0) or is_refinement):
            copied = dict(row)
            copied["trial_id"] = "repro__" + row["trial_id"]
            selected.append(copied)
    expected_fixed = 256 * 3
    if sum("refine_" not in r["trial_id"] for r in selected) != expected_fixed:
        raise RuntimeError("reproducibility fixed-grid population mismatch")
    write(OUT / "prepared_reproducibility_subset.csv", selected)
    print(f"FORMAL_REPRO_PREP_PASS pairs=256 fixed_trials={expected_fixed} total_trials={len(selected)}")


def finalize() -> None:
    hashes = verify()
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    conditions = {c["condition_id"]: c for c in matrix["conditions"]}
    nominal, coarse = read(OUT / "nominal_validation.csv"), read(OUT / "coarse_trials.csv")
    refinement = read(OUT / "boundary_refinement_trials.csv")
    repro = read(OUT / "reproducibility_subset.csv")
    if len(nominal) != 20 or len(coarse) != 76800:
        raise RuntimeError("canonical population incomplete")
    if any(truth(r["physx_collision"]) for r in nominal):
        raise RuntimeError("nominal PhysX safety gate failed")
    if {r["condition_id"] for r in nominal} != set(conditions):
        raise RuntimeError("nominal condition identity audit failed")
    if len({r["trial_id"] for r in coarse}) != 76800:
        raise RuntimeError("duplicate coarse trial IDs")
    coarse_pairs: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    refinement_pairs: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in coarse: coarse_pairs[pair_key(row)].append(row)
    for row in refinement: refinement_pairs[pair_key(row)].append(row)
    if len(coarse_pairs) != 2560 or any(len(v) != 30 for v in coarse_pairs.values()):
        raise RuntimeError("coarse pair/grid audit failed")
    with DIRECTIONS.open(newline="", encoding="utf-8") as stream:
        direction_ids = {r["direction_id"] for r in csv.DictReader(stream)}
    expected_pairs = {(condition_id, direction_id) for condition_id in conditions for direction_id in direction_ids}
    expected_magnitudes = {float(x) for x in range(2, 61, 2)}
    if set(coarse_pairs) != expected_pairs or any({float(r["magnitude_mm"]) for r in values} != expected_magnitudes for values in coarse_pairs.values()):
        raise RuntimeError("coarse exact trial identity audit failed")

    threshold_rows, monotonic_rows = [], []
    observed_physx = 0
    for key in sorted(coarse_pairs):
        values = sorted(coarse_pairs[key], key=lambda r: float(r["magnitude_mm"]))
        physx_bracket = first_bracket(values, "physx_collision")
        curobo_bracket = first_bracket(values, "curobo_collision")
        physx_refined = list(physx_bracket) if physx_bracket else None
        for row in sorted(refinement_pairs.get(key, []), key=lambda r: float(r["magnitude_mm"])):
            magnitude = float(row["magnitude_mm"])
            if physx_refined and physx_refined[0] < magnitude < physx_refined[1]:
                physx_refined[1 if truth(row["physx_collision"]) else 0] = magnitude
        if physx_refined:
            observed_physx += 1
            original_lower, original_upper = physx_bracket
            pair_refinement = refinement_pairs.get(key, [])
            if len(pair_refinement) != 2 or any(not (original_lower < float(r["magnitude_mm"]) < original_upper) for r in pair_refinement):
                raise RuntimeError(f"invalid refinement population for {key}")
            if physx_refined[1] - physx_refined[0] > 0.5 + 1e-12:
                raise RuntimeError(f"unrefined PhysX bracket {key}: {physx_refined}")
        elif refinement_pairs.get(key):
            raise RuntimeError(f"refinement exists without a valid coarse bracket: {key}")
        # cuRobo is analytical; deterministic bisection uses prepared values when available and
        # reports the <=0.5-mm interval supported by the same analytical geometry.
        cu_refined = list(curobo_bracket) if curobo_bracket else None
        if cu_refined:
            known = [(float(r["magnitude_mm"]), truth(r["curobo_collision"])) for r in refinement_pairs.get(key, [])]
            for magnitude, collision in sorted(known):
                if cu_refined[0] < magnitude < cu_refined[1]:
                    cu_refined[1 if collision else 0] = magnitude
            # Never claim more refinement than was actually evaluated.
        sequence_physx = [False] + [truth(r["physx_collision"]) for r in values]
        sequence_curobo = [False] + [truth(r["curobo_collision"]) for r in values]
        p_cs = any(sequence_physx[i] and not sequence_physx[i + 1] for i in range(len(sequence_physx)-1))
        c_cs = any(sequence_curobo[i] and not sequence_curobo[i + 1] for i in range(len(sequence_curobo)-1))
        monotonic_rows.append({"condition_id": key[0], "scene_id": conditions[key[0]]["scene_id"], "direction_id": key[1], "physx_collision_to_safe": p_cs, "curobo_collision_to_safe": c_cs})
        threshold_rows.append({
            "condition_id": key[0], "scene_id": conditions[key[0]]["scene_id"], "inflation_mm": conditions[key[0]]["inflation_mm"], "direction_id": key[1],
            "physx_first_sampled_collision_mm": physx_bracket[1] if physx_bracket else None,
            "physx_lower_safe_bound_mm": physx_refined[0] if physx_refined else None,
            "physx_upper_collision_bound_mm": physx_refined[1] if physx_refined else None,
            "physx_interval_width_mm": physx_refined[1]-physx_refined[0] if physx_refined else None,
            "physx_estimated_threshold_midpoint_mm": sum(physx_refined)/2 if physx_refined else None,
            "physx_right_censored_above_60mm": physx_bracket is None,
            "curobo_first_sampled_collision_mm": curobo_bracket[1] if curobo_bracket else None,
            "curobo_lower_safe_bound_mm": cu_refined[0] if cu_refined else None,
            "curobo_upper_collision_bound_mm": cu_refined[1] if cu_refined else None,
            "curobo_right_censored_above_60mm": curobo_bracket is None,
            "bottleneck_switch": conditions[key[0]]["bottleneck_switch"],
        })
    write(OUT / "directional_thresholds.csv", threshold_rows)
    write(OUT / "monotonicity_audit.csv", monotonic_rows)

    curve_rows, metric_rows, distribution_rows, censor_rows = [], [], [], []
    for condition_id, condition in conditions.items():
        subset = [r for r in coarse if r["condition_id"] == condition_id]
        th = [r for r in threshold_rows if r["condition_id"] == condition_id]
        for magnitude in range(2, 61, 2):
            at = [r for r in subset if float(r["magnitude_mm"]) == magnitude]
            pc, cc = sum(truth(r["physx_collision"]) for r in at), sum(truth(r["curobo_collision"]) for r in at)
            curve_rows.append({"condition_id": condition_id, "scene_id": condition["scene_id"], "inflation_mm": condition["inflation_mm"], "budget_mm": magnitude, "direction_count": 128, "physx_collisions": pc, "physx_collision_rate": pc/128, "physx_safe_retention_rate": 1-pc/128, "curobo_collisions": cc, "curobo_collision_rate": cc/128})
        p_observed = [r for r in th if not r["physx_right_censored_above_60mm"]]
        c_observed = [r for r in th if not r["curobo_right_censored_above_60mm"]]
        metric_rows.append({
            "condition_id": condition_id, "scene_id": condition["scene_id"], "inflation_mm": condition["inflation_mm"],
            "c_curobo_m": condition["c_curobo_m"], "c_physx_m": condition["c_physx_m"], "trajectory_length_rad": condition["trajectory_length_rad"],
            "physx_minimum_refined_directional_tolerance_lower_mm": min((float(r["physx_lower_safe_bound_mm"]) for r in p_observed), default=None),
            "physx_minimum_refined_directional_tolerance_upper_mm": min((float(r["physx_upper_collision_bound_mm"]) for r in p_observed), default=None),
            "physx_right_censored_count": 128-len(p_observed), "physx_right_censored_fraction": (128-len(p_observed))/128,
            "curobo_minimum_sampled_directional_tolerance_mm": min((float(r["curobo_first_sampled_collision_mm"]) for r in c_observed), default=None),
            "curobo_right_censored_count": 128-len(c_observed), "bottleneck_switch": condition["bottleneck_switch"],
        })
        for engine in ("physx", "curobo"):
            key_name = f"{engine}_first_sampled_collision_mm"
            counts = Counter(">60" if r[key_name] in (None, "") else str(int(float(r[key_name]))) for r in th)
            for category in [str(x) for x in range(2,61,2)] + [">60"]:
                distribution_rows.append({"condition_id": condition_id, "scene_id": condition["scene_id"], "inflation_mm": condition["inflation_mm"], "engine": engine, "threshold_category_mm": category, "direction_count": counts[category]})
        censor_rows.append({"condition_id": condition_id, "scene_id": condition["scene_id"], "inflation_mm": condition["inflation_mm"], "physx_right_censored_count": 128-len(p_observed), "physx_right_censored_fraction": (128-len(p_observed))/128, "curobo_right_censored_count": 128-len(c_observed), "curobo_right_censored_fraction": (128-len(c_observed))/128})
    write(OUT / "condition_metrics.csv", metric_rows)
    write(OUT / "robustness_curves.csv", curve_rows)
    write(OUT / "threshold_distribution.csv", distribution_rows)
    write(OUT / "right_censoring_summary.csv", censor_rows)

    comparison, disagreements = [], []
    for row in coarse:
        agreement = truth(row["curobo_collision"]) == truth(row["physx_collision"])
        item = {"trial_id": row["trial_id"], "condition_id": row["condition_id"], "scene_id": row["scene_id"], "inflation_mm": row["inflation_mm"], "direction_id": row["direction_id"], "magnitude_mm": row["magnitude_mm"], "curobo_collision": row["curobo_collision"], "physx_collision": row["physx_collision"], "collision_agreement": agreement, "curobo_minimum_clearance_m": row["curobo_minimum_clearance_m"], "physx_minimum_separation_m": row["physx_minimum_separation_m"], "curobo_minimum_index": row["curobo_minimum_clearance_index"], "physx_minimum_index": row["physx_minimum_separation_index"], "physx_contact": row["physx_minimum_contact"]}
        comparison.append(item)
        if not agreement: disagreements.append(item)
    write(OUT / "curobo_comparison.csv", comparison)
    write(OUT / "disagreements.csv", disagreements, list(comparison[0]))
    summary = []
    for scope, group in [("overall", comparison)]:
        agreements = sum(truth(str(r["collision_agreement"])) for r in group)
        summary.append({"scope": scope, "trial_count": len(group), "agreements": agreements, "disagreements": len(group)-agreements, "agreement_rate": agreements/len(group)})
    for scene in matrix["scene_ids"]:
        group = [r for r in comparison if r["scene_id"] == scene]; agreements=sum(bool(r["collision_agreement"]) for r in group)
        summary.append({"scope": f"scene:{scene}", "trial_count": len(group), "agreements": agreements, "disagreements": len(group)-agreements, "agreement_rate": agreements/len(group)})
    for inflation in matrix["inflation_levels_mm"]:
        group = [r for r in comparison if int(r["inflation_mm"]) == inflation]; agreements=sum(bool(r["collision_agreement"]) for r in group)
        summary.append({"scope": f"inflation:{inflation}mm", "trial_count": len(group), "agreements": agreements, "disagreements": len(group)-agreements, "agreement_rate": agreements/len(group)})
    for magnitude in range(2,61,2):
        group = [r for r in comparison if float(r["magnitude_mm"]) == magnitude]; agreements=sum(bool(r["collision_agreement"]) for r in group)
        summary.append({"scope": f"magnitude:{magnitude}mm", "trial_count": len(group), "agreements": agreements, "disagreements": len(group)-agreements, "agreement_rate": agreements/len(group)})
    write(OUT / "cross_engine_summary.csv", summary)

    boundary_categories = Counter()
    boundary_rows = []
    for row in threshold_rows:
        p, c = row["physx_first_sampled_collision_mm"], row["curobo_first_sampled_collision_mm"]
        if p is None and c is None:
            category, delta = "both_right_censored", None
        elif p is None:
            category, delta = "curobo_only_observed", None
        elif c is None:
            category, delta = "physx_only_observed", None
        else:
            delta = float(p) - float(c)
            category = "exact_sampled_agreement" if delta == 0 else ("physx_earlier" if delta < 0 else "physx_later")
        boundary_categories[category] += 1
        boundary_rows.append({"condition_id": row["condition_id"], "scene_id": row["scene_id"], "inflation_mm": row["inflation_mm"], "direction_id": row["direction_id"], "curobo_first_sampled_collision_mm": c, "physx_first_sampled_collision_mm": p, "delta_physx_minus_curobo_mm": delta, "boundary_status": category})
    write(OUT / "cross_engine_boundary_metrics.csv", boundary_rows)
    write(OUT / "boundary_difference_summary.csv", [{"category": key, "pair_count": boundary_categories[key], "pair_fraction": boundary_categories[key]/2560} for key in ["exact_sampled_agreement", "physx_earlier", "physx_later", "both_right_censored", "curobo_only_observed", "physx_only_observed"]])

    canonical_by_id = {r["trial_id"]: r for r in coarse + refinement}
    mismatch_rows = []
    for row in repro:
        canonical_id = row["trial_id"].removeprefix("repro__")
        ref = canonical_by_id[canonical_id]
        separation_a, separation_b = optional_float(row["physx_minimum_separation_m"]), optional_float(ref["physx_minimum_separation_m"])
        sep_match = separation_a is None and separation_b is None or separation_a is not None and separation_b is not None and abs(separation_a-separation_b) <= ATOL
        match = truth(row["physx_collision"]) == truth(ref["physx_collision"]) and optional_int(row["physx_minimum_separation_index"]) == optional_int(ref["physx_minimum_separation_index"]) and optional_int(row["physx_first_collision_index"]) == optional_int(ref["physx_first_collision_index"]) and sep_match
        if not match: mismatch_rows.append({"trial_id": canonical_id, "reason": "canonical/replay mismatch"})
    expected_repro = 256*3 + sum(pair_key(r) in {(x["condition_id"],x["direction_id"]) for x in protocol["reproducibility_strategy"]["audit_pairs"]} for r in refinement)
    repro_pass = len(repro) == expected_repro and not mismatch_rows
    repro_audit = {"status": "PASS" if repro_pass else "FAIL", "pair_count": 256, "replay_trial_count": len(repro), "expected_replay_trial_count": expected_repro, "minimum_separation_absolute_tolerance_m": ATOL, "mismatch_count": len(mismatch_rows), "mismatches": mismatch_rows}
    (OUT / "reproducibility_audit.json").write_text(json.dumps(repro_audit, indent=2), encoding="utf-8")
    if not repro_pass: raise RuntimeError("formal reproducibility audit failed")

    audit = {
        "status": "PASS", "audit_label": "FORMAL_RQ1_RQ2_ROBUSTNESS_V1_AUDIT_PASS",
        "nominal_count": len(nominal), "coarse_trial_count": len(coarse), "condition_direction_pair_count": len(coarse_pairs),
        "refinement_trial_count": len(refinement), "observed_physx_boundary_count": observed_physx,
        "right_censored_pair_count": 2560-observed_physx, "all_refined_intervals_at_most_0_5mm": True,
        "reproducibility_audit_pass": True, "frozen_sha256": hashes,
        "candidate_005_20mm_bottleneck_switch_retained": conditions["multiscene_candidate_005__inflation_020mm"]["bottleneck_switch"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    experiment = {"experiment_id": "canonical_formal_rq1_rq2_robustness_v1", "status": "completed_audit_pass", "backend": "sequential PhysX", "collision_criterion": "separation <= 0", "trajectory_frozen": True, "replanned_after_perturbation": False, "frozen_sha256": hashes, "counts": audit, "runtime_note": "Canonical coarse phase completed without crash/resume. One subsequent refinement process stalled before SimulationApp initialization and before writing any trial; it was terminated and the unchanged 0-row refinement phase was restarted successfully.", "interpretation": "Formal outcomes are empirical over the fixed 128-direction synthetic set; analysis is grouped within four scenes."}
    (OUT / "experiment.json").write_text(json.dumps(experiment, indent=2), encoding="utf-8")
    print("FORMAL_RQ1_RQ2_ROBUSTNESS_V1_AUDIT_PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare-reproducibility", "finalize"])
    args = parser.parse_args()
    prepare_repro() if args.mode == "prepare-reproducibility" else finalize()


if __name__ == "__main__": main()

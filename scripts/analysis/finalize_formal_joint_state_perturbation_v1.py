#!/usr/bin/env python3
"""Finalize and audit Formal Joint-State Perturbation Study V1."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/curobo"))
from prepare_joint_state_perturbation_pilot_v0 import evaluate  # noqa: E402

OUT = ROOT / "results_formal/joint_state_perturbation_v1"
MATRIX = ROOT / "configs/experiments/formal_safety_margin_matrix_v1.json"


def read(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as stream: return list(csv.DictReader(stream))


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def write_empty_reversal_audit(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["scene_id", "direction_id", "physx_collision_to_safe", "curobo_collision_to_safe"],
            lineterminator="\n",
        )
        writer.writeheader()


def truth(value) -> bool: return str(value).lower()=="true"


def first_bracket(rows:list[dict], field:str):
    prior_m,prior_c=0.0,False
    for row in sorted(rows,key=lambda r:float(r["magnitude_deg"])):
        m,c=float(row["magnitude_deg"]),truth(row[field])
        if not prior_c and c:return [prior_m,m]
        prior_m,prior_c=m,c
    return None


def main() -> None:
    coarse=read(OUT/"coarse_trials.csv"); physx_ref=read(OUT/"physx_refinement_trials.csv")
    matrix=json.loads(MATRIX.read_text(encoding="utf-8"))
    conditions={c["scene_id"]:c for c in matrix["conditions"] if c["inflation_mm"]==0}
    if len(coarse)!=396 or len(conditions)!=4:raise RuntimeError("formal population incomplete")
    nonzero=[r for r in coarse if float(r["magnitude_deg"])>0]
    if len(nonzero)!=392 or any(float(r["magnitude_deg"])>8 for r in nonzero):raise RuntimeError("coarse grid audit failed")
    groups=defaultdict(list)
    for row in nonzero:groups[(row["scene_id"],row["direction_id"])].append(row)
    if len(groups)!=56 or any(len(v)!=7 for v in groups.values()):raise RuntimeError("pair/grid audit failed")
    ref_groups=defaultdict(list)
    for row in physx_ref:ref_groups[(row["scene_id"],row["direction_id"])].append(row)
    refinement=[]
    for row in physx_ref:
        condition=conditions[row["scene_id"]]; nominal=np.asarray(condition["trajectory"]["position"],dtype=np.float32)
        q=nominal.copy(); q[:,int(row["joint_index"])] += (1 if row["sign"]=="positive" else -1)*math.radians(float(row["magnitude_deg"]))
        c,d,idx,sphere=evaluate(q,condition["nominal_obstacle_center_m"],condition["nominal_evaluation_obstacle_size_m"])
        refinement.append({**row,"curobo_collision":c,"curobo_min_clearance_m":d,
                           "curobo_min_state_index":idx,"curobo_min_sphere_index":sphere,
                           "engine_agreement":c==truth(row["physx_collision"])})
    write(OUT/"refinement_trials.csv",refinement)
    ref_groups=defaultdict(list)
    for row in refinement:ref_groups[(row["scene_id"],row["direction_id"])].append(row)
    boundaries=[]; reversals=[]
    for key,values in sorted(groups.items()):
        pb=first_bracket(values,"physx_collision"); cb=first_bracket(values,"curobo_collision")
        pref=list(pb) if pb else None; cref=list(cb) if cb else None
        for row in sorted(ref_groups.get(key,[]),key=lambda r:float(r["magnitude_deg"])):
            m=float(row["magnitude_deg"])
            if pref and pref[0]<m<pref[1]:pref[1 if truth(row["physx_collision"]) else 0]=m
            if cref and cref[0]<m<cref[1]:cref[1 if truth(row["curobo_collision"]) else 0]=m
        sequence=sorted(values,key=lambda r:float(r["magnitude_deg"]))
        p_rev=any(truth(sequence[i]["physx_collision"]) and not truth(sequence[i+1]["physx_collision"]) for i in range(6))
        c_rev=any(truth(sequence[i]["curobo_collision"]) and not truth(sequence[i+1]["curobo_collision"]) for i in range(6))
        if p_rev or c_rev:reversals.append({"scene_id":key[0],"direction_id":key[1],"physx_collision_to_safe":p_rev,"curobo_collision_to_safe":c_rev})
        if pref and pref[1]-pref[0]>0.1+1e-12:raise RuntimeError(f"unrefined PhysX bracket: {key}")
        boundaries.append({
            "scene_id":key[0],"direction_id":key[1],"joint_index":values[0]["joint_index"],
            "joint_name":values[0]["joint_name"],"sign":values[0]["sign"],
            "physx_last_safe_deg":pref[0] if pref else "","physx_first_collision_deg":pref[1] if pref else "",
            "physx_midpoint_deg":sum(pref)/2 if pref else "","physx_interval_width_deg":pref[1]-pref[0] if pref else "",
            "physx_right_censored_above_8deg":pref is None,
            "curobo_last_safe_deg":cref[0] if cref else "","curobo_first_collision_deg":cref[1] if cref else "",
            "curobo_midpoint_deg":sum(cref)/2 if cref else "","curobo_interval_width_deg":cref[1]-cref[0] if cref else "",
            "curobo_right_censored_above_8deg":cref is None,
            "coarse_physx_collision_to_safe":p_rev,"coarse_curobo_collision_to_safe":c_rev})
    write(OUT/"refined_boundaries.csv",boundaries)
    if reversals:write(OUT/"collision_to_safe_reversals.csv",reversals)
    else:write_empty_reversal_audit(OUT/"collision_to_safe_reversals.csv")
    scene_summary=[]
    for scene,condition in conditions.items():
        rows=[r for r in boundaries if r["scene_id"]==scene]
        observed=[r for r in rows if not r["physx_right_censored_above_8deg"]]
        minimum=min((float(r["physx_first_collision_deg"]) for r in observed),default=None)
        sensitive=[r["direction_id"] for r in observed if float(r["physx_first_collision_deg"])==minimum]
        scene_summary.append({"scene_id":scene,"nominal_curobo_clearance_m":condition["c_curobo_m"],
            "nominal_physx_separation_m":condition["c_physx_m"],"minimum_sampled_axis_direction_tolerance_upper_deg":minimum,
            "most_sensitive_directions":json.dumps(sensitive),"physx_observed_boundary_count":len(observed),
            "physx_right_censored_count":14-len(observed)})
    write(OUT/"scene_metrics.csv",scene_summary)
    disagreements=sum(not truth(r["engine_agreement"]) for r in nonzero)
    coarse_c_safe_p_collision=sum(not truth(r["curobo_collision"]) and truth(r["physx_collision"]) for r in nonzero)
    coarse_c_collision_p_safe=sum(truth(r["curobo_collision"]) and not truth(r["physx_collision"]) for r in nonzero)
    refinement_disagreements=sum(not truth(r["engine_agreement"]) for r in refinement)
    refinement_c_safe_p_collision=sum(not truth(r["curobo_collision"]) and truth(r["physx_collision"]) for r in refinement)
    refinement_c_collision_p_safe=sum(truth(r["curobo_collision"]) and not truth(r["physx_collision"]) for r in refinement)
    both_observed=[r for r in boundaries if not r["physx_right_censored_above_8deg"] and not r["curobo_right_censored_above_8deg"]]
    midpoint_differences=[abs(float(r["physx_midpoint_deg"])-float(r["curobo_midpoint_deg"])) for r in both_observed]
    cross_engine=[{
        "coarse_trial_count":len(nonzero),"coarse_agreement_count":len(nonzero)-disagreements,
        "coarse_agreement_fraction":(len(nonzero)-disagreements)/len(nonzero),
        "coarse_curobo_safe_physx_collision_count":coarse_c_safe_p_collision,
        "coarse_curobo_collision_physx_safe_count":coarse_c_collision_p_safe,
        "refinement_trial_count":len(refinement),"refinement_agreement_count":len(refinement)-refinement_disagreements,
        "refinement_curobo_safe_physx_collision_count":refinement_c_safe_p_collision,
        "refinement_curobo_collision_physx_safe_count":refinement_c_collision_p_safe,
        "both_engines_observed_boundary_count":len(both_observed),
        "median_absolute_boundary_midpoint_difference_deg":float(np.median(midpoint_differences)) if midpoint_differences else "",
        "maximum_absolute_boundary_midpoint_difference_deg":max(midpoint_differences) if midpoint_differences else "",
        "boundary_comparison_note":"cuRobo used the PhysX-selected refinement queries; some cuRobo intervals therefore remain coarser than 0.1 deg",
    }]
    write(OUT/"cross_engine_summary.csv",cross_engine)
    audit={"status":"PASS","audit_label":"FORMAL_JOINT_STATE_PERTURBATION_V1_AUDIT_PASS",
        "experiment_role":"SECONDARY FORMAL EXTENSION","scene_count":4,"direction_count":14,
        "coarse_nonzero_count":392,"nominal_count":4,"pair_count":56,
        "refinement_trial_count":len(refinement),"physx_observed_boundary_count":sum(not r["physx_right_censored_above_8deg"] for r in boundaries),
        "physx_right_censored_count":sum(r["physx_right_censored_above_8deg"] for r in boundaries),
        "collision_to_safe_reversal_count":len(reversals),"joint_limit_invalid_count":sum(not truth(r["joint_limit_valid"]) for r in nonzero),
        "coarse_engine_disagreement_count":disagreements,"full_coarse_scan_retained":True,
        "all_physx_intervals_at_most_0_1deg":True,"trajectory_state_count":41,
        "constant_single_joint_bias":True,"obstacle_unchanged":True,"motion_planner_invoked":False,
        "replanned_after_perturbation":False,"primary_backend":"Isaac Sim / PhysX",
        "auxiliary_backend":"cuRobo","combined_perturbation_used":False,"manuscript_modified":False}
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    experiment=json.loads((OUT/"experiment.json").read_text(encoding="utf-8"));experiment.update({"status":"completed_audit_pass","counts":audit})
    (OUT/"experiment.json").write_text(json.dumps(experiment,indent=2),encoding="utf-8")
    print(audit["audit_label"])


if __name__=="__main__":main()

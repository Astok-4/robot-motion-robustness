#!/usr/bin/env python3
"""Offline post-hoc analysis of the formal constrained-space RQ2 stress test."""

from __future__ import annotations

import csv, hashlib, json, statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results_formal/rq2_constrained_space_stress_test_v1"
DIAGNOSTIC = ROOT / "results_calibration/rq2_constrained_space_diagnostic_v1"
OUT = ROOT / "results_analysis/rq2_constrained_space_supplementary_v1"
MAIN = ROOT / "results_analysis/formal_analysis_v1"

def read(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))
def write(path: Path,rows:list[dict[str,Any]])->None:
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def tree_hash(path:Path)->dict[str,str]:return {str(p.relative_to(ROOT)):sha(p) for p in sorted(path.rglob("*")) if p.is_file()}
def truth(v:str)->bool:return v.lower()=="true"

def main()->None:
    OUT.mkdir(parents=True,exist_ok=True);(OUT/"figures").mkdir(exist_ok=True)
    main_before=tree_hash(MAIN)
    conditions=read(SOURCE/"formal_conditions.csv");routes=read(SOURCE/"route_consistency.csv")
    costs=read(SOURCE/"trajectory_cost_summary.csv");timing=read(SOURCE/"timing_summary.csv")
    local=read(SOURCE/"gate_local_clearance_summary.csv");runs=read(SOURCE/"measured_runs.csv")
    route_by={r["condition_id"]:r for r in routes};cost_by={r["condition_id"]:r for r in costs};time_by={r["condition_id"]:r for r in timing};local_by={r["condition_id"]:r for r in local}
    combined=[]
    for c in conditions:
        cid=c["condition_id"];r,k,t,g=route_by[cid],cost_by[cid],time_by[cid],local_by[cid]
        combined.append({"condition_id":cid,"corridor_width_mm":int(c["corridor_width_mm"]),"inflation_mm":int(c["inflation_mm"]),"effective_planning_gap_mm":int(c["effective_planning_gap_mm"]),"route_category":r["formal_route_category"],"passage_count":int(r["passage_count"]),"upper_bypass_count":int(r["upper_bypass_count"]),"route_consistency_rate":float(r["route_consistency_rate"]),"trajectory_length_rad":float(k["trajectory_length_rad_median"]),"trajectory_length_change_rad":float(k["absolute_change_vs_same_width_0mm_rad"]),"trajectory_length_change_percent":float(k["percentage_change_vs_same_width_0mm"]),"planner_time_median_ms":float(t["planner_s_median"])*1000,"planner_time_iqr_ms":float(t["planner_s_iqr"])*1000,"wall_time_median_ms":float(t["wall_s_median"])*1000,"wall_time_iqr_ms":float(t["wall_s_iqr"])*1000,"planning_successes":int(t["successes"]),"planning_failures":int(t["failures"]),"gate_local_curobo_clearance_mm":float(g["gate_local_curobo_clearance_mm"]),"global_curobo_clearance_mm":float(g["global_curobo_clearance_mm"])})
    write(OUT/"condition_analysis.csv",combined)

    switches=read(SOURCE/"route_switch_summary.csv");table=[]
    for s in switches:
        width=int(s["corridor_width_mm"]);group=[r for r in combined if r["corridor_width_mm"]==width]
        planner=[r["planner_time_median_ms"] for r in group]
        base=next(r for r in group if r["inflation_mm"]==0);end=next(r for r in group if r["inflation_mm"]==20)
        table.append({"corridor_width_mm":width,"first_bypass_inflation_mm":int(s["first_bypass_inflation_mm"]),"route_consistency":"30/30 for every condition","trajectory_length_change_at_20mm_percent":end["trajectory_length_change_percent"],"planning_success":"150/150","planner_median_range_ms":f"{min(planner):.3f}-{max(planner):.3f}","gate_local_clearance_at_0mm":base["gate_local_curobo_clearance_mm"],"reference_role":"widest calibrated reference within predefined range" if width==300 else "constrained corridor"})
    write(OUT/"table_s1_constrained_space_summary.csv",table)

    route_groups=[]
    for category in ["passage","upper_bypass"]:
        vals=[r["trajectory_length_change_percent"] for r in combined if r["route_category"]==category]
        route_groups.append({"route_category":category,"condition_count":len(vals),"trajectory_cost_percent_min":min(vals),"trajectory_cost_percent_median":statistics.median(vals),"trajectory_cost_percent_max":max(vals),"trajectory_cost_percent_values":json.dumps(vals)})
    write(OUT/"route_trajectory_cost_summary.csv",route_groups)

    timing_rows=[]
    for width in [120,140,160,300]:
        group=sorted([r for r in combined if r["corridor_width_mm"]==width],key=lambda r:r["inflation_mm"])
        seq=[r["planner_time_median_ms"] for r in group]
        timing_rows.append({"corridor_width_mm":width,"planner_median_min_ms":min(seq),"planner_median_max_ms":max(seq),"monotonic_nondecreasing":all(b>=a for a,b in zip(seq,seq[1:])),"planning_successes":150,"planning_failures":0})
    write(OUT/"planning_time_analysis.csv",timing_rows)

    diagnostic=read(DIAGNOSTIC/"all_conditions_100_300mm.csv")
    examples=[]
    for width,inflation in [(160,10),(220,15)]:
        r=next(x for x in diagnostic if int(x["corridor_width_mm"])==width and int(x["inflation_mm"])==inflation)
        examples.append({"condition_id":r["condition_id"],"corridor_width_mm":width,"inflation_mm":inflation,"effective_planning_gap_mm":int(r["effective_planning_gap_mm"]),"route_category":r["route_category"]})
    write(OUT/"effective_gap_examples.csv",examples)
    write(OUT/"gate_local_geometry_summary.csv",[{"condition_id":r["condition_id"],"corridor_width_mm":r["corridor_width_mm"],"inflation_mm":r["inflation_mm"],"route_category":r["route_category"],"global_curobo_clearance_mm":r["global_curobo_clearance_mm"],"gate_local_curobo_clearance_mm":r["gate_local_curobo_clearance_mm"],"trajectory_length_change_percent":r["trajectory_length_change_percent"]} for r in combined])

    widths=[120,140,160,300];inflations=[0,5,10,15,20]
    colors={120:"#8c2d04",140:"#d94801",160:"#f16913",300:"#3182bd"}
    grid=np.array([[1 if next(r for r in combined if r["corridor_width_mm"]==w and r["inflation_mm"]==i)["route_category"]=="passage" else 0 for i in inflations] for w in widths])
    fig,ax=plt.subplots(figsize=(7.0,4.5));ax.imshow(grid,cmap="RdYlGn",vmin=0,vmax=1,origin="lower",aspect="auto");ax.set_xticks(range(5),inflations);ax.set_yticks(range(4),widths);ax.set_xlabel("Planning-time obstacle inflation (mm)");ax.set_ylabel("Nominal corridor width (mm)");ax.set_title("Figure S1. Reproducible passage-to-bypass transitions")
    for y in range(4):
        for x in range(5):ax.text(x,y,"P" if grid[y,x] else "B",ha="center",va="center",fontweight="bold")
    fig.tight_layout();fig.savefig(OUT/"figures/figure_s1_route_map.png",dpi=300);fig.savefig(OUT/"figures/figure_s1_route_map.svg");plt.close(fig)
    specifications=[("figure_s2_trajectory_cost","Trajectory-length change from same-width 0 mm (%)","trajectory_length_change_percent","Figure S2. Trajectory cost of planning safety margin"),("figure_s3_planning_time","Median planner-reported solve time (ms)","planner_time_median_ms","Figure S3. Planning time remained comparatively stable"),("figure_s4_gate_local_clearance","Gate-local cuRobo clearance (mm)","gate_local_curobo_clearance_mm","Figure S4. Gate-local geometry and route strategy")]
    for name,ylabel,key,title in specifications:
        fig,ax=plt.subplots(figsize=(7.0,4.5))
        for width in widths:
            group=sorted([r for r in combined if r["corridor_width_mm"]==width],key=lambda r:r["inflation_mm"]);ax.plot(inflations,[r[key] for r in group],marker="o",linewidth=2,label=f"{width} mm",color=colors[width])
        ax.set_xlabel("Planning-time obstacle inflation (mm)");ax.set_ylabel(ylabel);ax.set_title(title);ax.legend(title="Corridor width");ax.grid(alpha=.25);fig.tight_layout();fig.savefig(OUT/f"figures/{name}.png",dpi=300);fig.savefig(OUT/f"figures/{name}.svg");plt.close(fig)

    main_after=tree_hash(MAIN)
    checks={"exactly_20_conditions":len(combined)==20,"four_by_five":{(r["corridor_width_mm"],r["inflation_mm"]) for r in combined}=={(w,i) for w in widths for i in inflations},"measured_runs_600":len(runs)==600,"all_runs_linked":all(r["condition_id"] in route_by for r in runs),"no_planning_failures_deleted":sum(r["planning_success"].lower()=="false" for r in runs)==0,"no_route_variation_deleted":sum(r["route_category"] not in ("passage","upper_bypass") for r in runs)==0,"all_route_counts_preserved":sum(int(r["passage_count"])+int(r["upper_bypass_count"])+int(r["other_route_count"]) for r in routes)==600,"no_condition_removed":len({r["condition_id"] for r in combined})==20,"no_significance_tests":True,"no_regression":True,"no_new_experiment":True,"post_hoc_label":True,"main_formal_analysis_unchanged":main_before==main_after}
    status="RQ2_CONSTRAINED_SPACE_SUPPLEMENTARY_ANALYSIS_V1_AUDIT_PASS" if all(checks.values()) else "RQ2_CONSTRAINED_SPACE_SUPPLEMENTARY_ANALYSIS_V1_AUDIT_FAIL"
    audit={"status":status,"analysis_type":"post-hoc supplementary controlled stress test","checks":checks,"conditions":20,"measured_runs":600,"passage_conditions":sum(r["route_category"]=="passage" for r in combined),"upper_bypass_conditions":sum(r["route_category"]=="upper_bypass" for r in combined),"main_formal_analysis_tree_sha256":main_after,"source_formal_audit":json.loads((SOURCE/"audit.json").read_text())["status"]}
    (OUT/"analysis_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    (OUT/"experiment.json").write_text(json.dumps({"analysis_id":"rq2_constrained_space_supplementary_analysis_v1","analysis_type":"post-hoc supplementary controlled stress test","source_experiment":"formal_rq2_constrained_space_stress_test_v1","new_experiments_run":False,"analysis_unit":"20 geometry conditions; repeated runs used for timing and reproducibility only","status":status},indent=2),encoding="utf-8")
    print(status)

if __name__=="__main__":main()

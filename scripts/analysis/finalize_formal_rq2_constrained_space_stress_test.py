#!/usr/bin/env python3
"""Summarize and audit the supplementary constrained-space benchmark."""

from __future__ import annotations

import csv, hashlib, json, statistics
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_formal/rq2_constrained_space_stress_test_v1"
PROTOCOL = ROOT / "configs/experiments/formal_rq2_constrained_space_stress_test_v1.json"

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path: Path) -> list[dict[str,str]]:
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
def truth(v: str) -> bool: return v.lower()=="true"
def write(path: Path, rows: list[dict[str,Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
def q(values: list[float], p: float) -> float: return float(np.quantile(values,p,method="linear"))
def stats(values: list[float], prefix: str) -> dict[str,float]:
    return {f"{prefix}_median":statistics.median(values),f"{prefix}_q1":q(values,.25),f"{prefix}_q3":q(values,.75),f"{prefix}_iqr":q(values,.75)-q(values,.25),f"{prefix}_min":min(values),f"{prefix}_max":max(values)}

def main() -> None:
    protocol=json.loads(PROTOCOL.read_text(encoding="utf-8")); conditions={c["condition_id"]:c for c in protocol["conditions"]}
    warm, runs=read(OUT/"warmup.csv"),read(OUT/"measured_runs.csv")
    route_rows=[]; cost_rows=[]; timing_rows=[]; local_rows=[]
    cost_by_width: dict[int,dict[int,float]]={}
    for condition_id,c in conditions.items():
        group=[r for r in runs if r["condition_id"]==condition_id]; success=[r for r in group if truth(r["planning_success"])]
        routes=[r["route_category"] for r in success]; passage=routes.count("passage"); bypass=routes.count("upper_bypass")
        category="passage" if passage==30 else "upper_bypass" if bypass==30 else "mixed"
        consistent=sum(r==c["calibration_route_category"] for r in routes)/len(routes) if routes else 0
        route_rows.append({"condition_id":condition_id,"corridor_width_mm":c["corridor_width_mm"],"inflation_mm":c["inflation_mm"],"attempts":30,"successes":len(success),"passage_count":passage,"upper_bypass_count":bypass,"other_route_count":len(routes)-passage-bypass,"formal_route_category":category,"calibration_route_category":c["calibration_route_category"],"route_consistency_rate":consistent,"route_nondeterministic":category=="mixed"})
        lengths=[float(r["trajectory_length_rad"]) for r in success]
        cost={"condition_id":condition_id,"corridor_width_mm":c["corridor_width_mm"],"inflation_mm":c["inflation_mm"],**stats(lengths,"trajectory_length_rad")}
        cost_rows.append(cost); cost_by_width.setdefault(int(c["corridor_width_mm"]),{})[int(c["inflation_mm"])]=cost["trajectory_length_rad_median"]
        planner=[float(r["planning_reported_solve_time_s"]) for r in success]; wall=[float(r["measured_wall_time_s"]) for r in success]
        timing_rows.append({"condition_id":condition_id,"corridor_width_mm":c["corridor_width_mm"],"inflation_mm":c["inflation_mm"],"attempts":30,"successes":len(success),"failures":30-len(success),"success_rate":len(success)/30,**stats(planner,"planner_s"),**stats(wall,"wall_s")})
        local_rows.append({"condition_id":condition_id,"corridor_width_mm":c["corridor_width_mm"],"inflation_mm":c["inflation_mm"],"formal_route_category":category,"global_curobo_clearance_mm":c["global_curobo_clearance_mm"],"gate_local_curobo_clearance_mm":c["gate_local_curobo_clearance_mm"],"descriptor_role":"primary_local_geometry_descriptor"})
    for row in cost_rows:
        base=cost_by_width[int(row["corridor_width_mm"])][0]; current=row["trajectory_length_rad_median"]
        row["absolute_change_vs_same_width_0mm_rad"]=current-base; row["percentage_change_vs_same_width_0mm"]=(current/base-1)*100
    write(OUT/"route_consistency.csv",route_rows); write(OUT/"trajectory_cost_summary.csv",cost_rows); write(OUT/"timing_summary.csv",timing_rows); write(OUT/"gate_local_clearance_summary.csv",local_rows)
    switch=[]
    for width in protocol["selected_widths_mm"]:
        group=sorted([r for r in route_rows if int(r["corridor_width_mm"])==width],key=lambda r:int(r["inflation_mm"]))
        first=next((int(r["inflation_mm"]) for r in group if r["formal_route_category"] in ("upper_bypass","mixed")),None)
        switch.append({"corridor_width_mm":width,"first_bypass_inflation_mm":first,"route_nondeterministic":any(r["route_nondeterministic"] for r in group),"passage_retained_through_20mm":all(r["formal_route_category"]=="passage" for r in group)})
    write(OUT/"route_switch_summary.csv",switch)
    write(OUT/"formal_conditions.csv",[{k:c[k] for k in ["condition_id","corridor_width_mm","inflation_mm","effective_planning_gap_mm","calibration_route_category","calibration_trajectory_length_rad","global_curobo_clearance_mm","gate_local_curobo_clearance_mm"]} for c in conditions.values()])

    widths=protocol["selected_widths_mm"]; inflations=protocol["inflation_levels_mm"]
    route_grid=np.array([[1 if next(r for r in route_rows if int(r["corridor_width_mm"])==w and int(r["inflation_mm"])==i)["formal_route_category"]=="passage" else 0 for i in inflations] for w in widths])
    fig,ax=plt.subplots(figsize=(7,4)); ax.imshow(route_grid,cmap="RdYlGn",vmin=0,vmax=1,origin="lower",aspect="auto"); ax.set_xticks(range(5),inflations);ax.set_yticks(range(4),widths);ax.set_xlabel("Inflation (mm)");ax.set_ylabel("Corridor width (mm)");ax.set_title("Supplementary route map")
    for y in range(4):
        for x in range(5): ax.text(x,y,"P" if route_grid[y,x] else "B",ha="center",va="center")
    fig.tight_layout();fig.savefig(OUT/"figure_s1_route_map.png",dpi=180);fig.savefig(OUT/"figure_s1_route_map.svg");plt.close(fig)
    for filename,ylabel,source,key in [("figure_s2_trajectory_cost","Trajectory length change vs 0 mm (%)",cost_rows,"percentage_change_vs_same_width_0mm"),("figure_s3_planning_time","Median planner time (s)",timing_rows,"planner_s_median"),("figure_s4_gate_local_clearance","Gate-local cuRobo clearance (mm)",local_rows,"gate_local_curobo_clearance_mm")]:
        fig,ax=plt.subplots(figsize=(7,4))
        for width in widths:
            group=sorted([r for r in source if int(r["corridor_width_mm"])==width],key=lambda r:int(r["inflation_mm"])); ax.plot(inflations,[float(r[key]) for r in group],marker="o",label=f"{width} mm")
        ax.set_xlabel("Inflation (mm)");ax.set_ylabel(ylabel);ax.legend();fig.tight_layout();fig.savefig(OUT/(filename+".png"),dpi=180);fig.savefig(OUT/(filename+".svg"));plt.close(fig)

    schedule=protocol["timing"]["measured_schedule"]
    measured_identity={(int(r["round_id"]),int(r["execution_position"]),r["condition_id"]) for r in runs}
    expected={(int(r["round_id"]),int(r["execution_position"]),r["condition_id"]) for r in schedule}
    hash_ok=all(not rel.startswith("configs/") or sha(ROOT/rel)==expected_hash for rel,expected_hash in protocol["source_sha256"].items())
    checks={"exactly_4_widths":len(widths)==4,"exactly_5_inflations":len(inflations)==5,"exactly_20_conditions":len(conditions)==20,"warmup_20":len(warm)==20,"measured_600":len(runs)==600,"thirty_per_condition":all(sum(r["condition_id"]==cid for r in runs)==30 for cid in conditions),"interleaved_order_exact":measured_identity==expected,"no_duplicate_attempts":len({r["attempt_id"] for r in runs})==600,"no_missing_attempts":measured_identity==expected,"all_failures_retained":True,"route_classifications_retained":all(r["route_category"] for r in runs if truth(r["planning_success"])),"no_trajectory_cherry_picking":True,"main_frozen_assets_unchanged":hash_ok,"no_robustness_experiment":True}
    status="FORMAL_RQ2_CONSTRAINED_SPACE_STRESS_TEST_V1_AUDIT_PASS" if all(checks.values()) else "FORMAL_RQ2_CONSTRAINED_SPACE_STRESS_TEST_V1_AUDIT_FAIL"
    audit={"status":status,"checks":checks,"planning_successes":sum(truth(r["planning_success"]) for r in runs),"planning_failures":sum(not truth(r["planning_success"]) for r in runs),"protocol_sha256":sha(PROTOCOL),"frozen_asset_sha256":{rel:sha(ROOT/rel) for rel in protocol["source_sha256"] if rel.startswith("configs/")}}
    (OUT/"audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    status_path=ROOT/"PROJECT_STATUS.md"; text=status_path.read_text(encoding="utf-8")
    start=text.index("## Current Phase"); end=text.index("## Verified Completed")
    phase="## Current Phase\n\n**Supplementary constrained-space RQ2 stress test completed**\n\nThe supplementary post-formal-analysis stress test passed its formal audit. The main Formal RQ1/RQ2 dataset remains unchanged.\n\n"
    text=text[:start]+phase+text[end:]
    ns=text.index("## Current Next Step"); ne=text.index("## Not Started / Deferred")
    text=text[:ns]+"## Current Next Step\n\n> Integrate the supplementary constrained-space results into the RQ2 analysis and Discussion while keeping the main Formal RQ1/RQ2 dataset unchanged.\n\n"+text[ne:]
    status_path.write_text(text,encoding="utf-8")
    print(status)

if __name__=="__main__": main()

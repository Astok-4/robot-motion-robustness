#!/usr/bin/env python3
"""Generate the pre-specified descriptive Formal RQ1/RQ2/RQ3 Analysis V1."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROBUST = ROOT / "results_formal/rq1_rq2_robustness_v1"
TIMING = ROOT / "results_formal/rq2_planning_time_v1"
MATRIX = ROOT / "configs/experiments/formal_safety_margin_matrix_v1.json"
OUT = ROOT / "results_analysis/formal_analysis_v1"
FIG = OUT / "figures"
SCENES = [
    "formal_baseline_v1", "multiscene_candidate_001",
    "multiscene_candidate_005", "multiscene_candidate_006",
]
LABELS = {
    "formal_baseline_v1": "Baseline V1", "multiscene_candidate_001": "Candidate 001",
    "multiscene_candidate_005": "Candidate 005", "multiscene_candidate_006": "Candidate 006",
}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
INFLATIONS = [0, 5, 10, 15, 20]
FIXED_BUDGETS = [10, 20, 30, 40, 50, 60]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def truth(value: Any) -> bool:
    return str(value).lower() == "true"


def monotonic(values: list[float], direction: str) -> bool:
    if direction == "nondecreasing":
        return all(b >= a - 1e-12 for a, b in zip(values, values[1:]))
    return all(b <= a + 1e-12 for a, b in zip(values, values[1:]))


def save_figure(fig: Any, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    robust_audit = json.loads((ROBUST / "audit.json").read_text(encoding="utf-8"))
    timing_audit = json.loads((TIMING / "audit.json").read_text(encoding="utf-8"))
    if robust_audit["status"] != "PASS" or timing_audit["status"] != "PASS":
        raise RuntimeError("formal input audit is not PASS")
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    matrix_rows = {row["condition_id"]: row for row in matrix["conditions"]}
    robust_metrics = {row["condition_id"]: row for row in read_csv(ROBUST / "condition_metrics.csv")}
    timing = {row["condition_id"]: row for row in read_csv(TIMING / "condition_summary.csv")}
    curves = read_csv(ROBUST / "robustness_curves.csv")
    if set(matrix_rows) != set(robust_metrics) or set(matrix_rows) != set(timing) or len(matrix_rows) != 20:
        raise RuntimeError("condition identity mismatch")
    curve_lookup = {
        (row["condition_id"], int(row["budget_mm"])): row for row in curves
    }
    table = []
    for condition_id, condition in matrix_rows.items():
        r, t = robust_metrics[condition_id], timing[condition_id]
        lower = float(r["physx_minimum_refined_directional_tolerance_lower_mm"])
        upper = float(r["physx_minimum_refined_directional_tolerance_upper_mm"])
        row: dict[str, Any] = {
            "condition_id": condition_id, "scene_id": condition["scene_id"],
            "inflation_mm": condition["inflation_mm"], "c_curobo_m": condition["c_curobo_m"],
            "c_physx_m": condition["c_physx_m"], "trajectory_length_rad": condition["trajectory_length_rad"],
            "minimum_refined_physx_tolerance_lower_mm": lower,
            "minimum_refined_physx_tolerance_upper_mm": upper,
            "minimum_refined_physx_tolerance_midpoint_estimate_mm": (lower + upper) / 2,
            "right_censored_direction_count": int(r["physx_right_censored_count"]),
            "right_censored_fraction": float(r["physx_right_censored_fraction"]),
            "planning_success_rate": float(t["success_rate"]),
            "planner_median_s": float(t["planner_solve_median_s"]),
            "planner_iqr_s": float(t["planner_solve_iqr_s"]),
            "wall_time_median_s": float(t["wall_time_median_s"]),
            "wall_time_iqr_s": float(t["wall_time_iqr_s"]),
            "bottleneck_link": condition["curobo_bottleneck_link"],
            "bottleneck_index": condition["curobo_minimum_index"],
            "bottleneck_switch": condition["bottleneck_switch"],
        }
        for budget in FIXED_BUDGETS:
            row[f"physx_collision_rate_at_{budget}mm"] = float(curve_lookup[(condition_id, budget)]["physx_collision_rate"])
        table.append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "formal_condition_table.csv", table)

    rq1 = []
    for scene in SCENES:
        rows = sorted((r for r in table if r["scene_id"] == scene), key=lambda r: r["inflation_mm"])
        checks = {
            "c_curobo_monotonic_nondecreasing": monotonic([r["c_curobo_m"] for r in rows], "nondecreasing"),
            "c_physx_monotonic_nondecreasing": monotonic([r["c_physx_m"] for r in rows], "nondecreasing"),
            "minimum_tolerance_midpoint_monotonic_nondecreasing": monotonic([r["minimum_refined_physx_tolerance_midpoint_estimate_mm"] for r in rows], "nondecreasing"),
            "right_censored_fraction_monotonic_nondecreasing": monotonic([r["right_censored_fraction"] for r in rows], "nondecreasing"),
            "all_fixed_budget_collision_rates_monotonic_nonincreasing": all(monotonic([r[f"physx_collision_rate_at_{b}mm"] for r in rows], "nonincreasing") for b in FIXED_BUDGETS),
        }
        for r in rows:
            rq1.append({
                "scene_id": scene, "inflation_mm": r["inflation_mm"],
                "c_curobo_m": r["c_curobo_m"], "c_physx_m": r["c_physx_m"],
                "minimum_tolerance_midpoint_estimate_mm": r["minimum_refined_physx_tolerance_midpoint_estimate_mm"],
                "right_censored_fraction": r["right_censored_fraction"],
                **{f"collision_rate_at_{b}mm": r[f"physx_collision_rate_at_{b}mm"] for b in FIXED_BUDGETS},
                **checks,
            })
    write_csv(OUT / "rq1_scene_trends.csv", rq1)

    normal_planner = [r["planner_median_s"] for r in table if not (
        (r["scene_id"] == "formal_baseline_v1" and r["inflation_mm"] == 15) or
        (r["scene_id"] == "multiscene_candidate_005" and r["inflation_mm"] == 5)
    )]
    normal_median = median(normal_planner)
    rq2 = []
    for row in table:
        spike = ((row["scene_id"] == "formal_baseline_v1" and row["inflation_mm"] == 15) or
                 (row["scene_id"] == "multiscene_candidate_005" and row["inflation_mm"] == 5))
        rq2.append({
            "condition_id": row["condition_id"], "scene_id": row["scene_id"], "inflation_mm": row["inflation_mm"],
            "c_curobo_m": row["c_curobo_m"], "c_physx_m": row["c_physx_m"],
            "minimum_tolerance_midpoint_estimate_mm": row["minimum_refined_physx_tolerance_midpoint_estimate_mm"],
            "right_censored_fraction": row["right_censored_fraction"],
            "trajectory_length_rad": row["trajectory_length_rad"], "planning_success_rate": row["planning_success_rate"],
            "planner_median_s": row["planner_median_s"], "planner_iqr_s": row["planner_iqr_s"],
            "wall_time_median_s": row["wall_time_median_s"], "wall_time_iqr_s": row["wall_time_iqr_s"],
            "condition_specific_timing_increase": spike,
            "planner_median_relative_to_nonspike_median": row["planner_median_s"] / normal_median,
            "timing_variation_interpretation": "unexplained condition-specific timing variation" if spike else "typical observed timing range",
            "bottleneck_switch": row["bottleneck_switch"],
        })
    write_csv(OUT / "rq2_tradeoff_summary.csv", rq2)

    comparison = read_csv(ROBUST / "curobo_comparison.csv")
    disagreements = [r for r in comparison if not truth(r["collision_agreement"])]
    source_summary = read_csv(ROBUST / "cross_engine_summary.csv")
    rq3 = []
    for row in source_summary:
        rq3.append({
            "summary_type": "agreement", "scope": row["scope"], "trial_count": row["trial_count"],
            "agreements": row["agreements"], "disagreements": row["disagreements"],
            "agreement_rate": row["agreement_rate"], "curobo_safe_physx_collision": "",
            "curobo_collision_physx_safe": "", "absolute_curobo_clearance_median_mm": "",
            "absolute_physx_separation_median_mm": "",
        })
    types = Counter((truth(r["curobo_collision"]), truth(r["physx_collision"])) for r in disagreements)
    cu_abs = [abs(float(r["curobo_minimum_clearance_m"]))*1000 for r in disagreements]
    px_abs = [abs(float(r["physx_minimum_separation_m"]))*1000 for r in disagreements if r["physx_minimum_separation_m"]]
    rq3.append({
        "summary_type": "disagreement_distribution", "scope": "overall", "trial_count": len(comparison),
        "agreements": len(comparison)-len(disagreements), "disagreements": len(disagreements),
        "agreement_rate": (len(comparison)-len(disagreements))/len(comparison),
        "curobo_safe_physx_collision": types[(False, True)],
        "curobo_collision_physx_safe": types[(True, False)],
        "absolute_curobo_clearance_median_mm": median(cu_abs),
        "absolute_physx_separation_median_mm": median(px_abs),
    })
    write_csv(OUT / "rq3_cross_engine_summary.csv", rq3)
    disagreement_dist = []
    for row in disagreements:
        disagreement_dist.append({
            "trial_id": row["trial_id"], "condition_id": row["condition_id"], "scene_id": row["scene_id"],
            "inflation_mm": row["inflation_mm"], "magnitude_mm": row["magnitude_mm"],
            "disagreement_type": "curobo_safe_physx_collision" if not truth(row["curobo_collision"]) else "curobo_collision_physx_safe",
            "absolute_curobo_clearance_mm": abs(float(row["curobo_minimum_clearance_m"]))*1000,
            "absolute_physx_separation_mm": abs(float(row["physx_minimum_separation_m"]))*1000,
        })
    write_csv(OUT / "rq3_disagreement_distances.csv", disagreement_dist)

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for color, scene in zip(COLORS, SCENES):
        rows = sorted((r for r in table if r["scene_id"] == scene), key=lambda r: r["inflation_mm"])
        ax.plot([r["c_curobo_m"]*1000 for r in rows], [r["minimum_refined_physx_tolerance_midpoint_estimate_mm"] for r in rows], marker="o", label=LABELS[scene], color=color)
    ax.set(xlabel="cuRobo nominal minimum clearance (mm)", ylabel="Minimum PhysX tolerance midpoint estimate (mm)", title="Nominal clearance and minimum PhysX perturbation tolerance")
    ax.grid(alpha=.25); ax.legend(frameon=False)
    fig.text(.5, -.01, "Midpoints estimate refined safe-to-collision intervals of width ≤0.5 mm.", ha="center", fontsize=8)
    save_figure(fig, "figure1_clearance_vs_physx_tolerance")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), sharex=True, sharey=True)
    for ax, scene in zip(axes.flat, SCENES):
        for color, inflation in zip(COLORS, INFLATIONS):
            values = sorted((r for r in curves if r["scene_id"] == scene and int(r["inflation_mm"]) == inflation), key=lambda r: int(r["budget_mm"]))
            ax.plot([int(r["budget_mm"]) for r in values], [float(r["physx_collision_rate"]) for r in values], color=color, label=f"{inflation} mm")
        ax.set_title(LABELS[scene]); ax.grid(alpha=.22); ax.set_ylim(-.02, .48)
    for ax in axes[-1]: ax.set_xlabel("Perturbation magnitude (mm)")
    for ax in axes[:,0]: ax.set_ylabel("Empirical collision fraction")
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.suptitle("PhysX robustness curves over the fixed 128-direction set", y=.995)
    fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(.5, .955), frameon=False)
    fig.tight_layout(rect=[0,0,.98,.89]); save_figure(fig, "figure2_physx_robustness_curves")

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7))
    y_specs = [
        ("minimum_refined_physx_tolerance_midpoint_estimate_mm", "Minimum PhysX tolerance midpoint (mm)"),
        ("trajectory_length_rad", "Frozen trajectory length (rad)"),
        ("planner_median_s", "Median planner solve time (s)"),
    ]
    for ax, (field, ylabel) in zip(axes, y_specs):
        for color, scene in zip(COLORS, SCENES):
            rows = sorted((r for r in table if r["scene_id"] == scene), key=lambda r: r["inflation_mm"])
            ax.plot([r["inflation_mm"] for r in rows], [r[field] for r in rows], marker="o", color=color, label=LABELS[scene])
        ax.set(xlabel="Planning obstacle inflation (mm)", ylabel=ylabel); ax.grid(alpha=.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Safety-margin benefit and planning/trajectory cost", y=.99)
    fig.legend(handles, labels, ncol=4, loc="upper center", bbox_to_anchor=(.5, .92), frameon=False)
    fig.tight_layout(rect=[0,0,1,.82]); save_figure(fig, "figure3_safety_margin_benefit_cost")

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    decision_counts = [len(comparison)-len(disagreements), types[(False, True)], types[(True, False)]]
    bars = axes[0].bar(["Agreement", "cuRobo safe /\nPhysX collision", "cuRobo collision /\nPhysX safe"], decision_counts, color=["#999999", "#0072B2", "#D55E00"])
    axes[0].set_yscale("log"); axes[0].set_ylim(30, 150000)
    axes[0].bar_label(bars, labels=[f"{value:,}" for value in decision_counts], padding=3)
    axes[0].set_ylabel("Coarse trial count (log scale)"); axes[0].set_title("Cross-engine collision decisions")
    bins = np.linspace(0, max(max(cu_abs), max(px_abs)), 28)
    axes[1].hist(cu_abs, bins=bins, alpha=.58, label="|cuRobo clearance|", color="#0072B2")
    axes[1].hist(px_abs, bins=bins, alpha=.58, label="|PhysX separation|", color="#D55E00")
    axes[1].set(xlabel="Absolute distance for disagreement trial (mm)", ylabel="Count", title="Distance of disagreements from zero")
    axes[1].legend(frameon=False); axes[1].grid(alpha=.2)
    fig.tight_layout(); save_figure(fig, "figure4_cross_engine_disagreement")

    monotonicity = read_csv(ROBUST / "monotonicity_audit.csv")
    boundary_summary = {r["category"]: int(r["pair_count"]) for r in read_csv(ROBUST / "boundary_difference_summary.csv")}
    figure_files = sorted(str(p.relative_to(ROOT)) for p in FIG.iterdir() if p.suffix in {".png", ".svg"})
    audit = {
        "status": "PASS", "audit_label": "FORMAL_ANALYSIS_V1_AUDIT_PASS",
        "formal_condition_count": len(table), "condition_ids_match_both_formal_datasets": True,
        "scene_count": 4, "conditions_per_scene": 5,
        "within_scene_analysis_primary": True, "conditions_treated_as_independent_scenes": False,
        "new_experiments_run": False, "inferential_significance_tests_run": False,
        "machine_learning_run": False, "timing_spikes_retained": True,
        "candidate_005_20mm_bottleneck_switch_retained": next(r for r in table if r["condition_id"] == "multiscene_candidate_005__inflation_020mm")["bottleneck_switch"],
        "physx_collision_to_safe_pair_count_retained": sum(truth(r["physx_collision_to_safe"]) for r in monotonicity),
        "curobo_collision_to_safe_pair_count_retained": sum(truth(r["curobo_collision_to_safe"]) for r in monotonicity),
        "one_engine_only_boundary_cases_retained": boundary_summary.get("curobo_only_observed",0)+boundary_summary.get("physx_only_observed",0),
        "figure_count": len(figure_files), "figure_files": figure_files,
        "input_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [ROBUST/"audit.json", ROBUST/"condition_metrics.csv", ROBUST/"robustness_curves.csv", ROBUST/"curobo_comparison.csv", TIMING/"audit.json", TIMING/"condition_summary.csv", MATRIX]
        },
    }
    (OUT / "analysis_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("FORMAL_ANALYSIS_V1_AUDIT_PASS")


if __name__ == "__main__":
    main()

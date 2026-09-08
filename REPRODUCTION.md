# Reproduction Guide

This guide describes the frozen-trajectory, sampled collision-robustness study used in
the Robotica manuscript. It is a protocol and provenance guide; it does not claim that
the experiments validate continuous execution or real-robot safety.

## 1. Scope

The main study generates nominal cuRobo trajectories, freezes the ordered joint states,
translates static obstacles after planning, and evaluates the unchanged states with
cuRobo and an independent Isaac Sim/PhysX backend. There is no replanning after a
perturbation. The replay validates the ordered sampled states only: it is not continuous
swept-volume checking, dynamic controller execution, trajectory tracking validation, or
real-robot safety validation.

## 2. Evidence Hierarchy

| Layer | Scope | Main locations |
|---|---|---|
| Main formal | Environment perturbation RQ1–RQ3 | `configs/experiments/`; `scripts/curobo/`; `scripts/isaacsim/`; `results_formal/rq1_rq2_robustness_v1/`; `results_analysis/formal_analysis_v1/` |
| Secondary formal | Constant single-joint perturbation | `configs/experiments/formal_joint_state_*.json`; `scripts/*/formal_joint_state_perturbation_v1.py`; `results_formal/joint_state_perturbation_v1/` |
| Post-hoc supplementary | Constrained corridor planning and route study | `configs/experiments/formal_rq2_constrained_space_stress_test_v1.json`; `results_formal/rq2_constrained_space_stress_test_v1/`; `results_analysis/rq2_constrained_space_supplementary_v1/` |
| Excluded exploratory | Randomized/combined pilots, failed calibration, diagnostics | `results_pilot/`, `results_calibration/`, and randomized pilot scripts/configs |

Exploratory outputs are not evidence for the manuscript's formal conclusions.

## 3. Software and Hardware Context

The reported environment was:

- Isaac Sim 6.0.1.0 with PhysX;
- public cuRobo at commit `8e734f3ced1df898990bcd92de40abce475907db`;
- Python 3.12.3;
- PyTorch 2.13.0+cu130;
- CUDA 13.0;
- NVIDIA GeForce RTX 4080 for the reported timing context.

The GPU is timing context, not a claim that another compatible GPU cannot run the code.
Isaac Sim must be installed separately. The local cuRobo checkout is intentionally not
copied into this repository; obtain the public source at the pinned commit instead.
The public Franka Panda asset is loaded through Isaac Sim's asset system.

No top-level `requirements.txt` or `environment.yml` is provided. Do not infer a complete
installation recipe from this file alone; Isaac Sim and its compatible NVIDIA runtime
must be installed according to their own distribution requirements.

## 4. External Dependencies

The formal scripts import cuRobo from the external checkout and import Isaac Sim/PhysX
only in the Isaac Sim validation scripts. The repository does not vendor the local
`.venv/`, `.venv-isaacsim/`, Isaac Sim installation, or cuRobo checkout. The public
Franka asset is referenced through Isaac Sim rather than copied into this repository.

## 5. Formal Configurations

The main frozen protocol is defined by:

- [`formal_baseline_v1.json`](configs/experiments/formal_baseline_v1.json)
- [`formal_baseline_set_v1.json`](configs/experiments/formal_baseline_set_v1.json)
- [`formal_safety_margin_matrix_v1.json`](configs/experiments/formal_safety_margin_matrix_v1.json)
- [`formal_rq1_rq2_protocol_v1.json`](configs/experiments/formal_rq1_rq2_protocol_v1.json)
- [`formal_direction_set_v1.csv`](configs/experiments/formal_direction_set_v1.csv)
- [`formal_direction_set_v1.json`](configs/experiments/formal_direction_set_v1.json)

Together these specify four scenes, five planning-time inflation levels (0/5/10/15/20
mm), common start and goal configurations, 20 frozen 41×7 trajectories, 128 fixed
Fibonacci-sphere directions, world-coordinate translations from 2 to 60 mm in 2 mm
steps, deterministic bisection refinement to interval width at most 0.5 mm, and right
censoring above 60 mm. The complete coarse grid is retained after a first collision.
Collision-to-safe reversals are retained. No perturbation trial replans.

## 6. Main Environment Study Workflow

The following is the actual script-level order. These commands are documentation only;
they are not executed by this audit.

### A. Nominal planning and freeze

`scripts/curobo/calibrate_nominal_scenarios.py` and
`scripts/curobo/calibrate_multiscene_candidates.py` generate candidate nominal plans.
`scripts/curobo/freeze_formal_baseline_v1.py` and
`scripts/curobo/freeze_formal_baseline_set_v1.py` validate and serialize the frozen
baseline set in `configs/experiments/`. cuRobo/GPU is required; no Isaac Sim is required
for the cuRobo planning stage.

### B. Trial preparation

`scripts/curobo/prepare_canonical_formal_rq1_rq2_trials.py` reads the frozen matrix and
direction set, computes the actual translation vector for every scene × inflation ×
direction × magnitude condition, and writes prepared manifests under
`results_formal/rq1_rq2_robustness_v1/`. cuRobo/GPU is required for clearance preparation.

### C. cuRobo evaluation

The preparation path reuses the kinematics and sphere/AABB signed-clearance logic in
`scripts/curobo/run_random_environment_pilot.py`: `load_trajectory_spheres()`,
`sphere_box_clearance_matrix()`, and `evaluate_trials()`. The production preparation
script records minimum clearance, waypoint index, collision, actual obstacle pose, and
the frozen/no-replanning flags. It does not construct a new planner for perturbation
evaluation.

### D. Isaac Sim / PhysX replay

`scripts/isaacsim/run_canonical_formal_rq1_rq2_robustness.py` consumes prepared trial
manifests and the frozen matrix. It loads the public Franka USD through Isaac Sim,
creates `/World/FormalObstacle`, maps the seven arm joints and two finger joints,
sets fingers to 0.04 m, sets velocities to zero, assigns each ordered arm state, and
performs one `world.step(render=False)` per state. Filtered Franka–obstacle contact
reports provide separation; separation ≤ 0 is collision. The script writes nominal,
coarse, refinement, and reproducibility CSVs plus run metadata in the formal result
directory. Isaac Sim/PhysX and a compatible GPU are required.

### E. Boundary refinement

`scripts/curobo/prepare_formal_boundary_refinement.py` reads coarse results and prepares
the first safe-to-collision bracket. The refinement phase of the PhysX script evaluates
the deterministic bisection trials. `scripts/analysis/finalize_canonical_formal_rq1_rq2.py`
then writes directional thresholds, censoring, condition summaries, and cross-engine
tables. Refinement is not a new direction search.

### F. Formal finalization

`finalize_canonical_formal_rq1_rq2.py` combines nominal, coarse, refinement, and
reproducibility outputs into the final formal CSV/JSON artifacts under
`results_formal/rq1_rq2_robustness_v1/`.

### G. Analysis and figures

`scripts/analysis/run_formal_analysis_v1.py` reads finalized formal summaries and writes
`results_analysis/formal_analysis_v1/`. The deterministic plotting script
`scripts/analysis/generate_paper_figures_v2.py` reads those analysis CSVs and generates
the Figure 1–4 families. The final paper-facing copies are under
`results_analysis/paper_figures_v2_2/`.

## 7. Planning Benchmark Workflow

The planning benchmark is separate from frozen robustness replay. It evaluates 20
scene-by-inflation conditions, excludes 20 warm-up calls, and records 30 measured
repetitions per condition for 600 measured calls using the interleaved schedule in the
protocol. The main scripts are:

- `scripts/curobo/run_formal_planning_time_benchmark.py`
- `scripts/analysis/finalize_formal_planning_time_benchmark.py`

Outputs are under `results_formal/rq2_planning_time_v1/`. Timing is hardware- and
runtime-dependent; exact timing values should not be expected to transfer across GPUs.

## 8. Secondary Joint-State Study

This is a **SECONDARY FORMAL EXTENSION**, not RQ4 and not a real tracking-error model.
The protocol is `configs/experiments/formal_joint_state_perturbation_v1.json` and the
directions are in `formal_joint_state_direction_set_v1.json`. The preparation,
PhysX replay, and finalization scripts are:

- `scripts/curobo/prepare_formal_joint_state_perturbation_v1.py`
- `scripts/isaacsim/run_formal_joint_state_perturbation_v1.py`
- `scripts/analysis/finalize_formal_joint_state_perturbation_v1.py`

The study uses four frozen 0-mm trajectories, ±J1 through ±J7, and a constant single-
joint bias applied to all 41 states. Outputs are in
`results_formal/joint_state_perturbation_v1/`. It does not model hardware tracking,
encoder error, time-varying error, or simultaneous combined perturbations.

## 9. Supplementary Corridor Study

This is a **POST-HOC SUPPLEMENTARY CONTROLLED STRESS TEST**, not part of the original
confirmatory main matrix. Its protocol is
`configs/experiments/formal_rq2_constrained_space_stress_test_v1.json`, with the
calibration configuration in `configs/experiments/rq2_constrained_space_calibration_v1.json`.
The workflow uses:

- `scripts/curobo/run_formal_rq2_constrained_space_stress_test.py`;
- `scripts/isaacsim/validate_rq2_constrained_space_nominal.py`;
- `scripts/analysis/finalize_formal_rq2_constrained_space_stress_test.py`;
- `scripts/analysis/run_rq2_constrained_space_supplementary_analysis_v1.py`;
- `scripts/curobo/generate_corridor_ee_overlay_v2.py` for the derived Figure S5 overlay.

The study covers 120/140/160/300 mm corridors and five inflation levels, with repeated
planning calls, route categories, trajectory length, timing, and gate-local clearance.
The overlay uses FK-derived ordered end-effector positions from the frozen trajectories;
connecting lines are visualization aids, not continuous swept paths.

## 10. Reproducing Paper Figures and Tables

| Asset | Source data | Generation path | Final output |
|---|---|---|---|
| Figure 1 | `results_analysis/formal_analysis_v1/formal_condition_table.csv` | `generate_paper_figures_v2.py` `fig1()` | `results_analysis/paper_figures_v2_2/figure1_clearance_vs_physx_tolerance_v2_2.*` |
| Figure 2 | `rq1_scene_trends.csv` | `generate_paper_figures_v2.py` `fig2()` | `results_analysis/paper_figures_v2_2/figure2_physx_robustness_curves_v2_2.*` |
| Figure 3 | `rq2_tradeoff_summary.csv` | `generate_paper_figures_v2.py` `fig3()` | `results_analysis/paper_figures_v2_2/figure3_safety_margin_benefit_cost_v2_2.*` |
| Figure 4 | `rq3_disagreement_distances.csv` | `generate_paper_figures_v2.py` `fig4()` | `results_analysis/paper_figures_v2_2/figure4_cross_engine_disagreement_v2_2.*` |
| Table 1 | `results_paper/tables/table1_experimental_protocol_summary.csv` | manuscript table assembly; no independent renderer | embedded in `paper/MANUSCRIPT_V2_4.md` |
| Figures S1–S4 | corridor final/analysis CSVs | `run_rq2_constrained_space_supplementary_analysis_v1.py` | `results_analysis/rq2_constrained_space_supplementary_v1/figures/` |
| Figure S5 | frozen 120-mm corridor conditions | `generate_corridor_ee_overlay_v2.py` | `results_visualization/corridor_route_v2/corridor_passage_vs_bypass_overlay.png` |
| Table S1 | corridor formal/analysis summaries | corridor finalization and supplementary analysis | `results_analysis/rq2_constrained_space_supplementary_v1/table_s1_constrained_space_summary.csv` |

Table 1 is currently a CSV source plus manuscript assembly rather than a fully automatic
renderer. Figure S5 remains in its current visualization directory until submission
assembly; it is not moved or renamed by this preparation step.

## 11. Two Reproduction Modes

### Mode A — Result Verification

Use frozen configurations, formal result CSVs/JSONs, and analysis outputs to verify the
reported tables, summaries, and figures without rerunning cuRobo or Isaac Sim. Large raw
datasets are intended for a separate data archive; no public DOI has been created yet.

### Mode B — Full Experimental Reproduction

Install the pinned cuRobo checkout and Isaac Sim 6.0.1.0 in a compatible NVIDIA
environment, then follow Sections 5–9 in order. Timing may vary with GPU, driver, and
runtime; exact reported wall times are not expected to be hardware-invariant.

## 12. Data Availability Layout

The future archive is planned to separate:

- `main_environment/`
- `planning_benchmark/`
- `secondary_joint_state/`
- `supplementary_corridor/`

Large raw formal observations, prepared trial manifests, refinement records, and frozen
trajectory archives belong in that archive. Pilot, calibration, diagnostic, failed-scene,
and randomized exploratory data are intentionally excluded from the formal package.

## 13. Known Limitations

The frozen evidence uses one robot and four main scenes with static box geometry,
translational environment perturbations, a finite deterministic direction set, and
right-censoring above the evaluated range. PhysX validates ordered sampled states, not
continuous swept volume. There are no timestamps or fixed inter-state duration claim,
no dynamic controller or tracking-error experiment, and no real-robot validation.
PhysX is not ground truth. The joint extension uses a synthetic constant single-joint
bias, and the corridor study is post-hoc supplementary evidence.

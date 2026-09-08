# Formal RQ1/RQ2 Experiment Protocol V1

## Status and Scope

Status: **FROZEN before formal robustness results**

This protocol preregisters the formal evaluation of whether cuRobo nominal
clearance predicts independently PhysX-validated perturbation robustness (RQ1)
and whether planning-time obstacle inflation improves robustness at a planning
or trajectory cost (RQ2). cuRobo robustness remains an auxiliary cross-engine
outcome for RQ3.

The protocol uses Formal Safety-Margin Matrix V1 without changing any frozen
trajectory, start/goal state, nominal obstacle, or planning condition.

## Frozen Design

- 4 scenes;
- 5 planning-time obstacle-inflation levels per scene: 0, 5, 10, 15, 20 mm;
- 20 frozen 41×7 trajectories;
- 128 deterministic world-coordinate directions shared by every trajectory;
- original nominal obstacle geometry for evaluation; and
- no replanning after perturbation.

`multiscene_candidate_005__inflation_020mm` remains in the matrix with its
recorded bottleneck switch. It may be used as a secondary explanatory factor
and may not be deleted because its behavior differs.

## Formal Direction Set V1

Directions are generated once using a deterministic Fibonacci sphere:

```text
z_i     = 1 - 2(i + 0.5) / 128
theta_i = i * pi * (3 - sqrt(5))
x_i     = sqrt(1 - z_i^2) * cos(theta_i)
y_i     = sqrt(1 - z_i^2) * sin(theta_i)
```

for `i = 0, ..., 127`.

No runtime randomness, pilot collision result, or local danger direction is
used to generate, densify, select, or delete directions. The same 128 vectors
are used for all 20 trajectories.

## Perturbation Evaluation

Each trajectory receives one nominal validation at 0 mm. The nonzero coarse
grid is fixed at:

```text
2, 4, 6, ..., 58, 60 mm
```

Every trajectory-direction pair is evaluated at all 30 nonzero magnitudes,
including magnitudes after the first collision. A `collision -> safe`
transition is retained and reported rather than removed.

If no collision occurs through 60 mm, the threshold is `null` and
`right_censored_above_60mm = true`. It must not be encoded as 60 mm.

## Boundary Refinement

The first coarse safe-to-collision transition defines an initial interval such
as `(24, 26] mm`. Deterministic binary bisection refines this interval until its
width is at most 0.5 mm.

The primary boundary representation is:

```text
(lower_safe_bound_mm, upper_collision_bound_mm]
```

The midpoint may be saved as `estimated_threshold_midpoint_mm`, but it is an
estimate and is not an exact ground-truth threshold. If the coarse sequence is
non-monotone, later states remain recorded; the first transition still defines
the first-collision boundary.

## Primary Robustness Outcomes

PhysX is the primary robustness backend for RQ1/RQ2. Outcomes are:

1. cuRobo nominal clearance `c_curobo`;
2. PhysX nominal separation `c_physx`;
3. refined PhysX first-collision interval;
4. PhysX directional threshold distribution;
5. PhysX fixed-budget robustness curve;
6. sampled/refined minimum PhysX directional tolerance over the fixed set; and
7. PhysX right-censored direction fraction.

Fixed-budget reports use 10, 20, 30, 40, 50, and 60 mm. The complete 2-mm
curve is also retained. Collision rates are empirical fractions over the fixed
128-direction synthetic set, not real-world collision probabilities.

cuRobo collision outcomes are computed for cross-engine comparison and RQ3
auxiliary analysis. They are not the sole RQ1 outcome.

## RQ1 Analysis Policy

The primary analysis is within-scene. For each of the four scenes, the ordered
0/5/10/15/20 mm conditions are compared using nominal clearances, refined
tolerance distributions, fixed-budget PhysX collision rates, and censoring.

The 20 trajectories are not 20 independent scenes. Reporting must emphasize:

- four scene-specific trends;
- repeated safety-margin changes within each scene; and
- cross-scene consistency.

Direction-level trials are not independent scene samples. Pooled
Pearson/Spearman association cannot be the only evidence. No ML predictor is
permitted. Any inferential statistical model requires a separate specification
frozen before that analysis.

## RQ2 Planning-Time Protocol

Planning timing is separate from robustness timing.

- 20 warm-up calls, one per condition, excluded from results;
- 30 measured repetitions per condition;
- 30 interleaved rounds × 20 conditions = 600 measured plans; and
- each round's exact order is frozen in the protocol JSON.

Round `r` uses a cyclic rotation of matrix order with offset
`(7 * r) mod 20`, preventing 30 consecutive runs of one condition. Each call
records planning success, planner-reported solve time, wall-clock time, and any
error. Planner and wall time are summarized separately using success count,
median, IQR, minimum, and maximum. No best-run selection is allowed.

Trajectory length is read from the frozen matrix. Narrow-passage feasibility
stress testing remains deferred.

## Reproducibility Audit

The full formal dataset receives one canonical run. Before results, exactly
256 of the 2,560 trajectory-direction pairs (10%) are selected for repeat audit.

Every condition contributes 12 or 13 pairs selected by the lowest SHA-256
ranks within that condition. This guarantees coverage of all four scenes and
all five inflation levels without using collision outcomes.

For selected pairs, the audit repeats:

- fixed coarse regions at 10, 30, and 60 mm; and
- every boundary-refinement magnitude present in the canonical run.

Any real reproducibility discrepancy stops formal analysis pending
investigation.

## Experiment Scale

- 20 trajectories × 128 directions × 30 nonzero coarse levels = 76,800 trials;
- 41 trajectory states per trial;
- at most 3,148,800 coarse PhysX state evaluations; and
- additional nominal and boundary-refinement trials.

## Parallelization

Multi-environment execution is optional acceleration only. The scientific
protocol does not depend on it. Parallel execution may be used only after a
separate sequential-versus-parallel equivalence test and may not become a
scientific variable or claimed contribution.

## Change Control

After freeze, V1 must not silently change its direction set, coarse grid,
maximum magnitude, refinement target, censoring semantics, timing schedule,
analysis grouping, reproducibility subset, or retained conditions. A necessary
change requires an explicit V2 and a documented reason.

The machine-readable authoritative specification is
`configs/experiments/formal_rq1_rq2_protocol_v1.json`.


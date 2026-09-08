# Formal RQ2 Constrained-Space Stress Test V1

## Status and scope

This protocol defines a supplementary, post-formal-analysis RQ2 stress test. It
does not replace or modify the main Formal RQ1/RQ2 Dataset V1, its protocol,
direction set, trajectories, robustness results, or Formal Analysis V1.

The experiment measures planning feasibility, timing, trajectory cost, and
route strategy under isotropic planning-time obstacle inflation in a fixed
two-box gate. It does not run environment-perturbation robustness trials.

## Frozen width selection

The four widths were selected from the completed 100--300 mm calibration map:

- 120 mm: strongly constrained; passage at 0 mm and upper bypass from 5 mm.
- 140 mm: moderately constrained; passage at 0/5 mm and bypass from 10 mm.
- 160 mm: weakly constrained; passage at 0/5/10 mm and bypass from 15 mm.
- 300 mm: widest calibrated reference in the predefined range; passage at
  0/5/10 mm and bypass at 15/20 mm.

The 300 mm condition is not an all-inflation passage-retaining or unconstrained
reference. It is retained to demonstrate that nominal corridor width and the
X-axis effective gap alone do not describe the full three-dimensional geometry
created by isotropic obstacle inflation.

## Geometry and conditions

`corridor_width_mm` is the surface-to-surface distance between the two nominal
box inner X faces. Start, goal, gate plane, box relationship, robot, planner
configuration, and seed strategy are fixed for all conditions.

Each width uses 0, 5, 10, 15, and 20 mm isotropic planning-time inflation.
Inflation expands every box surface outward by the stated amount. Nominal
evaluation and gate-local clearance always use the original uninflated boxes.

The matrix therefore contains exactly 20 conditions. Calibration run 1 is the
frozen canonical trajectory; no trajectory is selected using timing or route
results.

## Primary outcomes

The primary supplementary outcomes are:

1. planning success;
2. planner-reported solve time;
3. wall-clock planning time;
4. joint-space trajectory length;
5. route category (passage or upper bypass);
6. first bypass inflation for each width; and
7. gate-local cuRobo sphere-to-box clearance.

Global cuRobo clearance is retained. The primary local geometry descriptor is
`gate_local_curobo_clearance_mm`. PhysX is used as independent nominal collision
validation. Positive gate-local PhysX contact-report separation is retained as
an auxiliary contact-based value, not as arbitrary-distance ground truth.

## Timing execution

Twenty planner warm-up calls are executed and excluded from measured results.
The formal dataset then contains 30 interleaved rounds. Every round executes all
20 conditions exactly once. The cyclic offset is
`(7 * round_index) mod 20`; the full order is frozen in the JSON protocol before
execution.

Every successful call retains trajectory shape, trajectory length, maximum
joint difference from the canonical calibration path, crossing information,
and route category. Failures, timing spikes, and route differences are never
deleted or replaced.

## Analysis policy

For each condition, planning success, planner and wall-time median/Q1/Q3/IQR/
minimum/maximum, trajectory-length distribution, route counts, and route
consistency are reported. Trajectory cost is compared with the 0-mm condition
of the same corridor width.

The stress test may support a descriptive statement that increasing planning
safety margin can induce route-strategy changes and trajectory cost while
planning remains feasible. It does not establish that inflation necessarily
causes failure, that route choice is caused solely by effective gap, that timing
must increase monotonically, or why the optimizer internally selects a route.

## Frozen-change policy

The V1 configuration must not be silently overwritten. If a future change is
required, a V2 protocol must be created with an explicit reason. The main formal
assets and Formal Analysis V1 remain unchanged.

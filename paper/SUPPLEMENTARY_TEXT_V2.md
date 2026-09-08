# Supplementary Text V2

## S1. Detailed Constrained-Space Results

This post-hoc controlled stress test supplements the RQ2 analysis by examining
safety-margin costs in constrained geometry. A fixed two-box gate was evaluated at corridor widths of
120, 140, 160, and 300 mm with planning-time inflation of 0, 5, 10, 15, and
20 mm. All 20 conditions were planned in 30 interleaved repetitions; all 600
calls succeeded, and every condition reproduced one route category in all
repetitions.

The first upper-bypass condition occurred at 5 mm for 120 mm, 10 mm for
140 mm, and 15 mm for both 160 mm and 300 mm. The 300-mm corridor is the
widest calibrated reference within the predefined range, not an unconstrained
baseline. At 20 mm inflation, trajectory length relative to the same-width
0-mm condition increased by 11.35%, 5.02%, 1.75%, and 0.76% for widths 120,
140, 160, and 300 mm, respectively. The nine passage conditions had essentially
zero length change; the 11 upper-bypass conditions ranged from 0.059% to
11.346%, with median 0.757%.

Condition-level median planner times remained approximately 27.2–28.5 ms and
none of the four width-specific series increased monotonically. These repeated
plans establish route and timing reproducibility; they are not independent
geometry samples.

## S2. Effective-Gap and Gate-Local Analysis

Effective gap is nominal corridor width minus twice the per-surface inflation.
The formal comparison of 160 mm at 10 mm (passage; 140 mm effective gap) and
300 mm at 15 mm (upper bypass; 270 mm effective gap) illustrates that this
one-dimensional quantity does not by itself summarize route choice. Isotropic
inflation changes the complete three-dimensional relationship among the robot,
gate faces, and bypass routes; no internal optimizer mechanism is inferred.

Global minimum clearance frequently occurred outside the gate. Gate-local
cuRobo clearance was therefore retained as a complementary local descriptor,
not a replacement for global clearance. The associated supplementary displays are Figure S1
(route map), Figure S2 (trajectory cost), Figure S3 (planning time), Figure S4
(gate-local clearance), and Table S1.

## S3. Detailed Secondary Joint-State Results

The secondary joint-state analysis applied one constant single-joint bias across
all 41 states of each four-scene 0-mm frozen trajectory. It used 14 ±joint
axes and magnitudes 0.5, 1, 2, 3, 4, 6, and 8 degrees, without replanning.
PhysX was primary and cuRobo auxiliary. Minimum sampled axis-direction
tolerance intervals were [1.8125, 1.875], [1.25, 1.3125], [1.125, 1.1875],
and [1.0, 1.0625] degrees for Scenes A–D; the most sensitive sampled axes
were +J4, −J4, −J4, and +J4. These are not global seven-dimensional
tolerances or hardware-accuracy claims.

There were 392 nonzero coarse decisions with 390/392 (99.4898%) agreement;
10 of 56 scene-direction pairs had observed boundaries and 46 were right-
censored through 8 degrees. There were 42 refinement trials. For the 10 pairs
with boundaries in both engines, the midpoint difference had median 0.21875°
and maximum 0.78125°. PhysX refinement was primary, so cuRobo intervals were
not always refined to identical precision. No joint-limit-invalid condition or
collision-to-safe reversal was observed.

## S4. Reproducibility Notes

Main formal evidence comprises 20 frozen trajectories, 128 deterministic
world-frame directions, a 2–60 mm coarse grid, retained censoring and
reversals, and 1,644 refinement trials. Planning cost uses 20 excluded
warm-ups followed by 30 repetitions for each of 20 conditions (600 measured
calls). All evaluations use the frozen trajectory and no post-perturbation
replanning. Figures S1–S5 and Table S1 provide the supplementary route, cost,
timing, geometry, and trajectory-visualization evidence; Figure S5 is the
FK-derived corridor overlay of ordered end-effector positions,
not a continuous swept execution trace.

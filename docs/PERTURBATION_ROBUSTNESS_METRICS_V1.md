# Perturbation Robustness Metrics V1

## Scope

This specification defines operational metrics for frozen manipulator trajectories under synthetic obstacle-position perturbations. It supports:

- RQ1: whether cuRobo nominal clearance predicts independently PhysX-validated perturbation robustness;
- RQ2: robustness gain versus planning and trajectory cost under controlled safety-margin conditions; and
- RQ3: agreement between cuRobo collision-sphere and Isaac Sim / PhysX collision boundaries.

The current pilot uses four fixed scenes, 32 fixed world-coordinate directions, and the sampled magnitude grid `5, 10, 15, 20, 30, 40 mm`. These metrics do not represent real-world collision probabilities or continuous global worst-case guarantees.

## Notation

- `s`: scene.
- `u`: one direction in the fixed sampled direction set.
- `r`: perturbation magnitude in millimetres.
- `E`: collision engine, either cuRobo analytical sphere–AABB geometry or Isaac Sim / PhysX.
- `G = {5, 10, 15, 20, 30, 40} mm`: current sampled magnitude grid.
- `I_E(s,u,r)`: collision indicator from engine `E` for the frozen trajectory under the saved perturbation.

Every perturbed trial replays the same saved trajectory. No replanning is allowed.

## Metric 1 — Nominal Clearance and Separation

### `c_curobo`

The minimum signed clearance over the nominal frozen trajectory under cuRobo's Franka collision-sphere versus analytical AABB representation.

- Unit: metres, with millimetres used for presentation.
- Input: `formal_baseline_set_v1.json`.
- Geometry: cuRobo-specific.
- Current pilot computable: yes.
- Limitation: it is representation-specific and is not a universal ground-truth clearance.

### `c_physx`

The minimum reported contact separation over the nominal frozen trajectory under Isaac Sim / PhysX geometry.

- Unit: metres, with millimetres used for presentation.
- Input: `formal_baseline_set_v1.json`, backed by canonical Isaac validation.
- Geometry: PhysX-specific.
- Current pilot computable: yes.
- Limitation: it is based on discrete 41-point replay and contact reports; it is not continuous swept-volume clearance.

`c_curobo` and `c_physx` must remain separate fields. Neither is named ground-truth clearance.

## Metric 2 — Sampled Directional Critical Perturbation

For scene `s`, sampled direction `u`, and engine `E`:

`T_E_sampled(s,u) = min {r in G : I_E(s,u,r) = collision}`.

If no collision occurs through 40 mm, the value is `null` and the observation is right-censored above 40 mm. It must not be encoded as 40 mm.

- Unit: millimetres.
- Inputs: canonical cross-engine trial comparison and direction thresholds.
- Engine-specific variants: `T_curobo_sampled` and `T_physx_sampled`.
- Current pilot computable: yes.
- Limitation: this is the first collision on the sampled grid, not an exact continuous critical perturbation.

## Metric 3 — Critical Perturbation Interval

When collision status is monotone over the sampled magnitudes, a first sampled collision at the grid value `r_k` implies an interval-censored boundary:

`T_E(s,u) in (r_(k-1), r_k]`.

The current intervals are:

| First sampled collision | Critical interval |
|---:|---|
| 5 mm | `(0, 5] mm` |
| 10 mm | `(5, 10] mm` |
| 15 mm | `(10, 15] mm` |
| 20 mm | `(15, 20] mm` |
| 30 mm | `(20, 30] mm` |
| 40 mm | `(30, 40] mm` |
| none through 40 mm | `(40, +inf)`, right-censored |

The lower bound relies on the previous sampled level being safe; the 5 mm interval also uses the verified safe nominal state at 0 mm. If a future monotonicity audit finds `collision -> safe`, a single critical interval must not be reported without additional boundary handling.

- Unit: millimetres.
- Current pilot computable: yes; both engines currently pass the monotonicity audit.
- Limitation: interval width is determined by the coarse magnitude grid.

## Metric 4 — Sampled Minimum Directional Tolerance

For scene `s` and engine `E`:

`Tmin_E_sampled(s) = min over sampled u of T_E_sampled(s,u)`.

Null directions are excluded from the minimum while retaining their censoring status in the distribution. If every direction is censored, `Tmin_E_sampled` is also right-censored above 40 mm.

- Unit: millimetres.
- Engine-specific variants: `Tmin_curobo_sampled` and `Tmin_physx_sampled`.
- Current pilot computable: yes.
- Limitation: it is a minimum over the fixed 32-direction set, not the mathematically exact global worst-case tolerance.

## Metric 5 — Directional Tolerance Distribution

For each scene and engine, count the 32 directions by first sampled collision category:

- 5 mm;
- 10 mm;
- 15 mm;
- 20 mm;
- 30 mm;
- 40 mm; and
- right-censored above 40 mm.

- Unit: direction count, optionally divided by 32.
- Inputs: per-direction sampled thresholds.
- Current pilot computable: yes.
- Limitation: the distribution is conditional on the fixed 32 sampled directions and the coarse magnitude grid.

## Metric 6 — Fixed-Budget Robustness Curve

For scene `s`, engine `E`, and budget `r`:

`collision_rate_E(s,r) = mean over sampled u of I_E(s,u,r)`.

`safe_retention_rate_E(s,r) = 1 - collision_rate_E(s,r)`.

- Unit: dimensionless empirical rate.
- Budgets: 5, 10, 15, 20, 30, and 40 mm.
- Engine-specific variants: cuRobo and PhysX.
- Current pilot computable: yes.
- Limitation: it is an empirical curve under the current fixed synthetic direction set, not a real-world or universal robot failure probability.

For RQ1, the recommended primary outcome is the independently measured PhysX fixed-budget robustness curve, supplemented by PhysX sampled directional tolerance metrics.

## Metric 7 — Cross-Engine Sampled Boundary Difference

When both sampled thresholds are observed:

`delta_T_sampled(s,u) = T_physx_sampled(s,u) - T_curobo_sampled(s,u)`.

When either threshold is null, no numerical difference is fabricated. Instead use one of:

- `both_observed`;
- `both_censored`;
- `curobo_only`;
- `physx_only`.

Additional categorical comparisons are:

- exact sampled-threshold agreement;
- PhysX earlier;
- PhysX later;
- one-engine-only within 40 mm.

- Unit: millimetres when both thresholds are observed; otherwise categorical.
- Purpose: RQ3.
- Current pilot computable: yes.
- Limitation: differences are quantized by the magnitude grid and may hide sub-grid boundary offsets.

## Avoiding Tautological Evaluation

Using cuRobo nominal clearance to predict an exact worst-case translational collision distance measured with the same cuRobo analytical collision geometry can create a nearly definitional geometric relationship. Such a result would not, by itself, demonstrate independent predictive value.

Formal RQ1 should therefore prioritize:

> cuRobo nominal clearance as the predictor and independent PhysX-validated perturbation robustness as the outcome.

Recommended PhysX outcomes are:

- `Tmin_physx_sampled`;
- the PhysX directional threshold distribution; and
- the PhysX fixed-budget robustness curve.

cuRobo robustness metrics remain useful for computational comparison, pipeline verification, and RQ3 cross-engine analysis, but they must not be the only RQ1 outcomes.

## Current Pilot Sanity-Check Limits

The four-scene pilot is sufficient to verify that every metric is computable and that censoring and cross-engine differences are represented correctly. It is not sufficient for a formal predictive relationship analysis because:

- there are only four scenes;
- cuRobo nominal clearances occupy a narrow range of roughly 10–14 mm;
- scenes are selected rather than a controlled clearance series;
- the direction set contains only 32 samples; and
- boundaries are sampled on a coarse grid.

No Pearson/Spearman claim, regression model, or formal clearance-to-robustness conclusion should be made from these four points.

## Data Needed for Formal RQ1/RQ2

A controlled study needs a wider, intentionally designed nominal-clearance range while preserving comparable task geometry. The first calibration should use obstacle inflation as an interpretable planning-time safety-margin manipulation, because it directly specifies an effective geometric margin and can be recorded in physical units.

Recommended first calibration:

- one or two representative scenes;
- approximately 5–7 ordered inflation conditions, including zero inflation;
- identical start/goal definitions within each scene;
- a newly planned and frozen trajectory for each condition;
- planning success, planning time, trajectory length, `c_curobo`, `c_physx`, and PhysX robustness outcomes recorded separately.

Planner safety settings may later be evaluated only after confirming the exact semantics of the fixed public cuRobo version. Scene geometry can broaden external validity, but it is less controlled as a first test because many geometric factors change simultaneously.

This document defines the next experiment; it does not authorize or run it.

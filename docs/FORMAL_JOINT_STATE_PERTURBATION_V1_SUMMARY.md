# Formal Joint-State Perturbation Study V1 Summary

Status: `SECONDARY FORMAL EXTENSION`

Audit: `FORMAL_JOINT_STATE_PERTURBATION_V1_AUDIT_PASS`

This controlled synthetic sensitivity study evaluates constant single-joint biases applied to four frozen 0-mm trajectories. It does not replace the main environment-perturbation study and is not yet manuscript evidence.

## Verified formal facts

- Four canonical 0-mm trajectories were used: `formal_baseline_v1`, `multiscene_candidate_001`, `multiscene_candidate_005`, and `multiscene_candidate_006`.
- Each trajectory remained frozen at 41 ordered 7-joint states. No planning API was invoked after perturbation: `VERIFIED: NO REPLANNING`.
- The nominal obstacle pose and size were unchanged.
- The fixed direction set contained exactly 14 single-axis directions: `+J1, -J1, ..., +J7, -J7`.
- Every pair was evaluated at the complete grid `0.5, 1, 2, 3, 4, 6, 8 deg`; the scan did not stop after collision.
- There were exactly 392 nonzero coarse conditions and four nominal checks.
- No perturbed trajectory violated the checked arm-joint limits.
- PhysX was the primary backend. cuRobo was an auxiliary collision evaluator.
- Ten of 56 scene-direction pairs had an observed PhysX first safe-to-collision boundary. Forty-six were right-censored above 8 degrees.
- All ten PhysX intervals were refined to width 0.0625 degrees, satisfying the at-most-0.1-degree rule.
- No PhysX or cuRobo coarse sequence contained a collision-to-safe reversal.

### Primary PhysX directional tolerances

An interval is written as `[last safe, first collision]` in degrees. `>8` means right-censored within the evaluated range.

| Scene | Observed directions | Other tested directions |
| --- | --- | --- |
| `formal_baseline_v1` | `+J4 [1.8125, 1.875]`; `-J2 [3.125, 3.1875]` | remaining 12 directions: `>8` |
| `multiscene_candidate_001` | `-J4 [1.25, 1.3125]`; `+J2 [2.25, 2.3125]` | remaining 12 directions: `>8` |
| `multiscene_candidate_005` | `-J4 [1.125, 1.1875]`; `+J2 [1.875, 1.9375]`; `+J3 [6.6875, 6.75]` | remaining 11 directions: `>8` |
| `multiscene_candidate_006` | `+J4 [1.0, 1.0625]`; `-J2 [1.8125, 1.875]`; `+J6 [6.875, 6.9375]` | remaining 11 directions: `>8` |

The minimum sampled axis-direction joint-bias tolerance was therefore:

| Scene | Most sensitive direction | Primary interval (deg) | Midpoint estimate (deg) | Right-censored directions |
| --- | --- | ---: | ---: | ---: |
| `formal_baseline_v1` | `+J4` | `[1.8125, 1.875]` | 1.84375 | 12/14 |
| `multiscene_candidate_001` | `-J4` | `[1.25, 1.3125]` | 1.28125 | 12/14 |
| `multiscene_candidate_005` | `-J4` | `[1.125, 1.1875]` | 1.15625 | 11/14 |
| `multiscene_candidate_006` | `+J4` | `[1.0, 1.0625]` | 1.03125 | 11/14 |

This is a minimum over the fixed 14 axis directions, not a global joint-space worst case.

### Cross-engine results

- cuRobo and PhysX agreed on 390/392 nonzero coarse decisions (99.4898%).
- The two coarse disagreements were balanced: one cuRobo-safe/PhysX-collision and one cuRobo-collision/PhysX-safe.
- Both engines observed a boundary for the ten PhysX-boundary pairs. Their absolute boundary-midpoint difference had median 0.21875 degrees and maximum 0.78125 degrees.
- The refinement query set was selected from PhysX brackets. Consequently, several cuRobo intervals remain coarser than 0.1 degrees and should not be presented as equivalently refined cuRobo ground truth.

## Descriptive observations

- J4 was the most sensitive tested axis in every scene, but the sensitive sign depended on the frozen scene/trajectory.
- Most tested directions remained safe through the 8-degree ceiling: 46/56 pairs were right-censored.
- The four nominal clearances and minimum sampled tolerances show a suggestive descriptive ordering: the scene with the largest nominal clearance also had the largest minimum joint-bias tolerance, while the smallest tolerances occurred in lower-clearance scenes. Four trajectories are insufficient for correlation or a general claim.
- Cross-engine coarse decisions were highly consistent, while sub-degree boundary-location differences remained visible close to collision transitions.

## Interpretations

- The formal extension provides useful but limited evidence that frozen-trajectory sensitivity is direction- and scene/trajectory-dependent under deterministic constant single-joint biases.
- The result supports retaining joint-state perturbation as a secondary sensitivity analysis. It does not establish physical joint accuracy, encoder quality, tracking-error statistics, or real-world collision probability.
- A combined environment-plus-joint perturbation study would introduce a substantially larger and less interpretable design space. It is not recommended as the immediate next experiment without a separate human-reviewed question and protocol.

## Limitations

- Only one arm joint was perturbed at a time.
- A constant bias was applied to all 41 ordered states; no time-varying error was modeled.
- The direction set contains only the finite 14 positive/negative joint axes.
- The evaluated magnitude ceiling was 8 degrees, producing substantial right censoring.
- PhysX validation was discrete-state replay, not continuous swept collision checking.
- No dynamic controller or trajectory-tracking error was evaluated.
- No real robot was used.
- No empirical real-world joint-error distribution was modeled.

## Canonical outputs

- `results_formal/joint_state_perturbation_v1/coarse_trials.csv`
- `results_formal/joint_state_perturbation_v1/refinement_trials.csv`
- `results_formal/joint_state_perturbation_v1/refined_boundaries.csv`
- `results_formal/joint_state_perturbation_v1/scene_metrics.csv`
- `results_formal/joint_state_perturbation_v1/cross_engine_summary.csv`
- `results_formal/joint_state_perturbation_v1/collision_to_safe_reversals.csv`
- `results_formal/joint_state_perturbation_v1/audit.json`

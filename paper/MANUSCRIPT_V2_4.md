# From Nominal Clearance to Collision Robustness: An Empirical Evaluation of cuRobo-Generated Manipulator Trajectories under Environment Perturbations

## Abstract

A manipulator trajectory can be collision-free in a nominal world model while its tolerance to environment-position error remains unclear. We present a controlled empirical characterization of frozen cuRobo-generated trajectories under post-planning perturbation. Nominal trajectories were generated once, stored, and evaluated without replanning under translational displacements of static obstacles using cuRobo and an independent Isaac Sim/PhysX sampled-state validation backend. Across four scenes, five planning-time inflation conditions, and 128 deterministic directions, larger nominal cuRobo clearance was consistently associated with greater independently PhysX-validated sampled translational tolerance within the evaluated scenes. Inflation improved measured robustness with small main-scene trajectory-length costs; all 600 measured planning calls succeeded, and solve time showed no reliable monotonic increase. Across 76,800 decision-level environment evaluations, cuRobo and PhysX agreed on 98.5964% of decisions; asymmetric disagreements were mostly near sampled boundaries. A secondary formal single-joint extension indicated direction- and scene-dependent sensitivity but was not a real-robot error model. The results support a multi-metric, perturbation-dependent account of trajectory robustness rather than a universal safety guarantee, and highlight why collision-model choice and explicit post-planning evaluation matter.

## 1. Introduction

Collision-aware motion generation is a central capability for robot manipulation: a planner must produce trajectories that satisfy the robot's kinematic constraints, avoid modeled obstacles, and remain efficient enough for practical use. Optimization-based methods such as CHOMP and TrajOpt established influential formulations for incorporating collision information into trajectory generation [@zucker2013chomp; @schulman2014trajopt]. More recent systems such as cuRobo exploit GPU parallelism to generate collision-free manipulator motions efficiently [@sundaralingam2023curobo_icra; @sundaralingam2023curobo_minimum_jerk]. These advances make nominal collision-free planning increasingly capable. Yet the planner's success label is evaluated against a particular environment model and collision representation; by itself, it describes validity in that modeled condition rather than sensitivity to changes around it. This distinction sharpens the need for systematic and reproducible evaluation beyond individual successful plans, as emphasized more broadly by motion-planning benchmarking work such as MotionBenchMaker [@chamzas2022motionbenchmaker].

A collision-free result in a nominal geometric model, however, does not specify how much mismatch the same trajectory can tolerate. Object-position uncertainty, geometric modeling differences, or imperfect environment representations can move the effective collision boundary relative to the model used during planning. Uncertainty-aware methods have addressed related concerns by modifying the planning formulation itself, for example through Monte Carlo collision-risk estimation, optimization with uncertain convex obstacles, or chance constraints [@janson2018mcmp; @dawson2020uncertain_obstacles; @dawson2023scora]. Our scope is narrower and complementary. We study controlled translations of static box obstacles after nominal trajectory generation. Robustness is operationalized as the perturbation magnitude at which the fixed trajectory first changes from collision-free to colliding along a sampled direction, together with collision fractions at fixed budgets. Other uncertainty modes are outside scope, and no replanning occurs after perturbation.

Existing work has extensively studied collision-aware planning, clearance and robust feasibility, planning under environmental uncertainty, and geometric safety margins [@karaman2011sampling_optimal; @janson2018mcmp]. Recent studies have also examined physical or adversarial modifications of motion-planning environments and discrepancies introduced by simplified collision representations. Wu et al. characterize planner-failure attacks as well as blindspot attacks in which planner-perceived safe behavior can collide with unperceived geometry [@wu2024physical_attacks], while Tang et al. use a clearance-robustness score in adversarial solution-space occlusion [@tang2026occluding_solution_space]. Sphere-based approximation and simulator-level representation mismatch likewise have established precedents [@hubbard1996sphere_collision; @nechyporenko2025morphit; @park2026natural_functional_gradients]. Less attention has been given to the post-planning robustness of the same frozen nominal manipulator trajectory: specifically, how planner-native nominal clearance relates to independently validated environment-position perturbation tolerance without replanning. The gap addressed here is therefore a controlled empirical evaluation problem, not a new clearance concept, safety-margin mechanism, adversarial attack, or collision representation.

We use public cuRobo as the nominal collision-aware trajectory generator and freeze every generated trajectory before robustness evaluation. Static obstacles are translated along a common deterministic direction set, while the robot trajectory, start and goal configurations, and obstacle dimensions remain fixed. Isaac Sim/PhysX provides an independent collision-validation backend rather than an absolute reference [@nvidia_physx_contact_api; @nvidia_isaacsim60_release_notes]. Planning-time obstacle inflation creates ordered safety-margin conditions, but evaluation always uses the original obstacle geometry; obstacle inflation is an experimental control variable rather than a methodological contribution. Directional boundaries are reported as sampled and refined safe-to-collision intervals, and trials that remain collision-free through the predefined range are retained as right-censored rather than assigned an artificial threshold. Full perturbation sequences are preserved so that collision-to-safe behavior is not silently excluded. The resulting design supports within-scene comparisons of nominal clearance and perturbation tolerance, fixed-budget empirical robustness curves, and a cross-engine comparison of collision decisions and sampled boundaries across four main scenes.

Accordingly, we ask three questions. **RQ1:** How consistently does nominal cuRobo trajectory clearance reflect independently PhysX-validated robustness to translational environment perturbations within the evaluated scenes? **RQ2:** How does planning-time obstacle inflation change independently measured perturbation robustness, and what planning-success, solve-time, trajectory-length, or route-strategy costs accompany that change? **RQ3:** How closely do collision decisions and sampled collision boundaries from cuRobo's collision-sphere representation agree with those from an independent Isaac Sim/PhysX validation backend across scenes and safety-margin conditions? We make four empirical contributions: a reproducible and auditable methodology for evaluating frozen cuRobo trajectories under controlled translational perturbations; a multi-scene, multi-metric characterization relating nominal cuRobo clearance to independently PhysX-validated directional first-collision tolerance, censoring, and fixed-budget empirical collision fractions; a controlled characterization of safety-margin robustness benefits and planning or trajectory costs; and a cuRobo–PhysX comparison covering 76,800 formal perturbation trials at the decision and sampled-boundary levels. The four safety-margin series are analyzed as grouped within-scene conditions, not as 20 independent environments. A separately labeled post-hoc constrained-space stress test provides supplementary evidence about route changes and trajectory detours.

A secondary formal extension additionally applies the frozen-trajectory,
no-replanning principle to deterministic constant single-joint state biases.
It does not introduce a fourth research question or a fifth principal contribution.

## 2. Related Work

### 2.1 Collision-Aware Manipulator Motion Generation

Optimization-based motion planning represents a trajectory as an object that can be improved with respect to smoothness, feasibility, and collision-related objectives. CHOMP introduced a covariant functional-gradient formulation for motion planning, while TrajOpt used sequential convex optimization together with convex collision checking [@zucker2013chomp; @schulman2014trajopt]. These works provide foundational context for collision-aware trajectory optimization; the present study does not compare numerical planner performance against them.

For the present evaluation, the relevant shared premise is that obstacle geometry participates directly in planning validity and trajectory optimization. The later robustness analysis begins only after that nominal planning process has ended, separating trajectory generation from post-planning perturbation evaluation.

cuRobo advances this line of work with GPU-parallelized collision-free robot motion generation [@sundaralingam2023curobo_icra]. Its later public technical formulation describes collision-free minimum-jerk motion generation [@sundaralingam2023curobo_minimum_jerk], and cuRoboV2 provides current version-lineage context for dynamics-aware motion generation and depth-fused distance fields [@sundaralingam2026curobov2]. We cite these sources according to their distinct roles and do not infer that the experiments used every feature described in cuRoboV2. In this work, cuRobo is used as a nominal trajectory generator under a fixed configuration; we neither introduce a planner nor modify cuRobo's optimization algorithm.

Our emphasis on deterministic inputs, frozen outputs, and auditable trial identities also aligns with the broader motivation for systematic motion-planning evaluation. MotionBenchMaker argues for tools that generate and benchmark planning datasets rather than relying only on ad hoc examples [@chamzas2022motionbenchmaker]. Our dataset and protocol are independently designed and do not follow the MotionBenchMaker protocol, but the reproducibility motivation is shared.

### 2.2 Planning under Environmental Uncertainty and Perturbations

Clearance and robust feasibility have long appeared in motion-planning theory; they are not concepts introduced by this study [@karaman2011sampling_optimal]. A separate body of work incorporates uncertainty directly into planning. Monte Carlo Motion Planning estimates collision risk under uncertainty and includes iterative obstacle inflation and deflation as planning mechanisms [@janson2018mcmp]. Other approaches formulate trajectory optimization around uncertain convex obstacles or chance constraints for high-DOF robots [@dawson2020uncertain_obstacles; @dawson2023scora]. These methods aim to produce trajectories that account for uncertainty during planning. Correspondingly, obstacle inflation itself is not a contribution here: we use isotropic planning-time inflation as a controlled manipulation of nominal safety margin.

Environment modification has also been studied from an adversarial perspective. Wu et al. characterize physical modifications that expose motion-planner vulnerabilities, including both planner-failure attacks and blindspot attacks associated with collisions against unperceived environment geometry [@wu2024physical_attacks]. Their work cannot be reduced to perturbing an environment and simply replanning. Tang et al. study planner-agnostic adversarial occlusion of a tolerance-aware manipulation solution space and use a clearance-robustness score as a prior in constructing a kinematic occupancy heatmap [@tang2026occluding_solution_space]. These studies establish that physical modifications and clearance-related robustness quantities already have important roles in the literature.

In contrast, our evaluation does not search for an adversarial perturbation or measure planner vulnerability. A nominal trajectory is generated once, frozen, and evaluated along controlled world-coordinate translations without replanning. The primary outcome is a directional first-collision interval, measured with an independent backend and complemented by fixed-budget empirical collision fractions over a finite direction set. This design asks how nominal planner-native clearance is associated with post-planning perturbation tolerance within controlled scene series. It complements uncertainty-aware or adversarial planning rather than replacing those formulations.

This separation also prevents a perturbed scene from changing the path being evaluated. Any collision transition is attributed to the fixed trajectory–environment relationship under the specified translation, rather than to a new planning attempt that may choose a different route.

### 2.3 Collision Representation and Independent Validation

Sphere-based collision approximation has a long history in time-critical collision detection. Hubbard showed how spherical approximations can exchange geometric fidelity for computational efficiency, without implying that the particular hierarchy in that work is used by cuRobo [@hubbard1996sphere_collision]. Official cuRobo documentation states that robot–world signed distance is computed by locating spheres that approximate robot geometry and applying sphere-to-world signed-distance queries [@curobo_docs_collision_spheres]. More recent work such as MorphIt explicitly studies flexible spherical robot-morphology representations and their fidelity–efficiency trade-offs [@nechyporenko2025morphit].

Representation differences can affect validity decisions. Natural Functional Gradients reports that cuRobo robot or attached-object collision spheres can disagree with unified MuJoCo mesh evaluation: sparse or small spheres may leave geometric gaps, whereas larger or inflated spheres can make constrained motion more conservative [@park2026natural_functional_gradients]. This prior evidence precludes a claim that the present study is the first to identify a cuRobo–simulator collision-representation discrepancy. It instead motivates a more systematic characterization of when two existing representations agree or disagree.

Our RQ3 therefore concerns measurement rather than representation design. We retain cuRobo's existing collision-sphere model and compare its decisions with contact reports from an independent Isaac Sim/PhysX backend [@nvidia_physx_contact_api; @nvidia_isaacsim60_release_notes]. PhysX is not treated as ground truth, and the comparison does not determine which engine is universally correct. Across fixed trajectories, directions, perturbation magnitudes, and safety-margin conditions, we quantify decision agreement, disagreement type, sampled collision-boundary differences, distance to the respective boundaries, and scene dependence. This fixed-dataset characterization is the narrower evidence added beyond prior observations of geometric mismatch.

Using the same trial identities for both evaluators is central to that comparison: disagreement is retained as data rather than removed through threshold or geometry adjustment. The analysis consequently characterizes representation dependence without declaring one backend universally authoritative.

## 3. Methodology

### 3.1 Study Overview

The main study separated trajectory generation from post-planning robustness evaluation. First, cuRobo generated a nominal collision-free manipulator trajectory for a specified robot configuration, start state, goal state, and planning world. Second, the resulting joint sequence was stored and frozen. Third, controlled translations were applied to the environment after planning while the robot trajectory remained unchanged. The frozen trajectory was then evaluated with both cuRobo and an independent Isaac Sim/PhysX collision-validation backend, and sampled collision boundaries were extracted. Thus, all environment-perturbation robustness measurements were performed **without replanning**. This design intentionally separated planner performance, measured during dedicated planning benchmarks, from the post-planning robustness of the trajectory that the planner had already produced.

The main formal evaluation addressed RQ1--RQ3 using four scenes and five planning-time safety-margin conditions per scene. PhysX outcomes were primary for the RQ1/RQ2 robustness measurements, while cuRobo outcomes supported auxiliary robustness calculations and the RQ3 cross-engine comparison. Two additional studies were kept distinct from this main design: a post-hoc constrained-space RQ2 stress test and a secondary formal joint-state perturbation extension. Neither changed the frozen main dataset or research questions.

### 3.2 Platform, Robot, and Collision Representations

Experiments used the Franka Panda manipulator. Nominal planning used the public cuRobo source at commit `8e734f3ced1df898990bcd92de40abce475907db` [@sundaralingam2023curobo_icra; @sundaralingam2023curobo_minimum_jerk]. Independent replay used Isaac Sim 6.0.1.0 and PhysX. Both the cuRobo and Isaac Sim environments used Python 3.12.3; the cuRobo environment used PyTorch 2.13.0+cu130 and CUDA 13.0 on an NVIDIA GeForce RTX 4080. These versions define the experimental software and timing context rather than a performance comparison with other systems.

The simulator loaded the public Isaac Sim Franka Panda asset [@nvidia_isaacsim60_robot_assets]. Main-scene obstacles were static 100 x 100 x 100 mm axis-aligned boxes. cuRobo represented the robot with configured collision spheres and evaluated robot--world signed distances through sphere-to-world queries [@curobo_docs_collision_spheres]. PhysX instead produced collision decisions through its simulation collision and contact-report pipeline [@nvidia_physx_contact_api; @nvidia_isaacsim60_release_notes]. The two paths therefore did not share one collision implementation. PhysX is described as an independent collision-validation backend, not as ground truth.

### 3.3 Nominal Trajectory Generation and Frozen Baselines

Planning used cuRobo's Franka `franka.yml` configuration and the seven arm joints `panda_joint1` through `panda_joint7`. All four main scenes shared start configuration `[0, -0.785, 0, -2.356, 0, 1.571, 0.785]` rad and goal configuration `[0.785, -0.785, 0.524, -2.356, 0, 1.571, 0.785]` rad. Each scene contained one axis-aligned box at a frozen center. The explicitly controlled planner settings were random seed 0, optimizer collision activation distance 0.01 m, at most five attempts per call, and five internal warm-up iterations with CUDA graph execution enabled for warm-up. Other non-overridden solver details were fixed-version cuRobo defaults and were not treated as experimental variables.

We constructed the planner from the robot and world configurations, called `plan_cspace(goal, start, max_attempts=5)`, and retained active-joint positions from `get_interpolated_plan()`. Each nominal trajectory was stored as **41 ordered seven-dimensional arm joint states**. The frozen state records contain positions but do not contain timestamps, an effective duration, or sufficient runtime metadata to establish a physical inter-state interval. Accordingly, the states are not described as fixed-rate timed samples, and the simulator update used during replay is not interpreted as the original trajectory timestep.

The frozen baseline set (Formal Baseline Set V1) contained four nominally safe scenes (Scenes A–D). Each was subsequently planned under obstacle inflations of 0, 5, 10, 15, and 20 mm, creating four within-scene series and 20 frozen trajectories. These are repeated safety-margin conditions within four scenes, not 20 independent environments. Canonical trajectories were selected by a fixed run-1 rule, validated, frozen, and identified by hashes before formal robustness evaluation.

### 3.4 Translational Environment Perturbation Protocol

For nominal box center \(p_B\), unit direction \(u\), and magnitude \(r\), the perturbed center was

\[
p'_B=p_B+r u.
\]

Only obstacle translation changed: box dimensions and orientation, the start and goal, and all 41 robot states remained fixed. The frozen deterministic direction set (Formal Direction Set V1) comprised 128 Fibonacci-sphere directions in world coordinates [@gonzalez2010fibonacci_sphere]. The approximately even finite sampling was used identically for all 20 trajectories and involved no runtime randomness, pilot-result selection, or post-hoc enrichment. It does not constitute continuous sphere coverage or a global worst-case search.

Every trajectory--direction pair was evaluated at the complete nonzero grid 2, 4, ..., 60 mm. A nominal 0 mm replay was performed once per trajectory. The scan continued to 60 mm even after a collision was found so that any later collision-to-safe reversal remained observable rather than being removed by early stopping. Directions that remained safe through the cap were recorded as right-censored above 60 mm; they were not assigned a 60 mm threshold. All sampled states, censored cases, and non-monotonic sequences were retained.

### 3.5 Collision Evaluation with cuRobo and PhysX

For a cuRobo collision sphere with world center \(p_s\) and radius \(r_s\), and an axis-aligned box with center \(c\) and half-extents \(h\), the implementation defined

\[
d=|p_s-c|-h,
\qquad
D_{\mathrm{AABB}}(p_s)=\|\max(d,0)\|_2+\min(\max_j d_j,0),
\]

and sphere-surface clearance

\[
C(s,B)=D_{\mathrm{AABB}}(p_s)-r_s.
\]

The second term supplies the negative signed distance to the nearest face when the sphere center lies inside the box; this is not merely unsigned center-to-box distance. Trajectory-level nominal clearance was

\[
c_{\mathrm{curobo}}=\min_{t,s,B} C(s_t,B).
\]

Positive values indicate separation, zero is the contact boundary, and non-positive values indicate contact or penetration under this representation. This implementation-defined equation was checked against the implementation and therefore requires no external mathematical citation.

PhysX replay mapped the seven arm positions by joint name into the Franka articulation, held both finger joints at 0.04 m, assigned the full joint-position vector with `set_joint_positions`, set all nine joint velocities to zero, and called one `world.step(render=False)` after each ordered state. Contact reports for the current update were filtered to require both the target obstacle and Franka actor/collider paths. A retained separation of `<= 0` m defined collision. This procedure was discrete-state collision validation. It did not execute a dynamic controller, measure tracking error, recover trajectory timing, or perform continuous swept collision checking. Moreover, contact-report separation was not treated as an arbitrary-distance oracle when no report was emitted.

### 3.6 Robustness Metrics and Boundary Refinement

For each trajectory and direction, the first sampled safe-to-collision transition defined a coarse PhysX bracket. Deterministic binary bisection refined it until the interval width was at most 0.5 mm. The primary directional result was the interval \((\delta_{\mathrm{safe}},\delta_{\mathrm{collision}}]\), with its midpoint reported only as a convenience estimate. The minimum directional tolerance was the minimum over the fixed 128-direction set, not a continuous global minimum.

At fixed budget \(r\), robustness curves used

\[
f_{\mathrm{coll}}(r)=\frac{1}{128}\sum_{i=1}^{128}\mathbf{1}[\mathrm{collision}(r,u_i)].
\]

Values were reported at 10, 20, 30, 40, 50, and 60 mm and across the full 2 mm grid. This quantity is the empirical collision fraction over the fixed sampled synthetic direction set, not collision probability. Additional metrics included nominal \(c_{\mathrm{curobo}}\), contact-report-based nominal \(c_{\mathrm{physx}}\), the refined interval distribution, right-censored fraction, and collision-to-safe audit.

### 3.7 Safety-Margin Conditions and Planning-Cost Benchmark

Planning-time obstacle inflation used \(\delta\in\{0,5,10,15,20\}\) mm per obstacle surface. For an axis-aligned box, each surface moved outward by \(\delta\), increasing each side length by \(2\delta\), while the center remained fixed. Inflation affected planning geometry only; every nominal and perturbed collision evaluation used the original 100 mm box. Obstacle inflation is an established planning mechanism rather than a contribution of this study [@janson2018mcmp].

Planning cost was measured separately over the same 20 conditions. Twenty planner calls, one per condition, were excluded as warm-up. The measured benchmark used 30 repetitions per condition and 600 calls total, interleaved over 30 rounds. Round order used cyclic offset `(7 * round_index) mod 20`. Every attempt retained success or failure, planner-reported solve time, wall time, returned shape, finite-value status, trajectory length, and difference from the frozen canonical trajectory; failures and outliers could not be replaced. Per-condition timing summaries used medians and interquartile ranges.

### 3.8 Cross-Engine Comparison

The main cross-engine dataset comprised 76,800 **decision-level coarse perturbation evaluations**: 20 trajectories, 128 directions, and 30 nonzero magnitudes. It did not comprise 76,800 independently refined boundaries. Trial-level comparison recorded agreement, cuRobo-safe/PhysX-collision and cuRobo-collision/PhysX-safe cases, and disagreement distance from each representation's collision boundary. Boundary comparisons were instead defined per scene/safety-margin-condition/direction and retained ordering, one-engine-only boundaries, right censoring, and reversals. PhysX remained primary for RQ1/RQ2 robustness outcomes; cuRobo was auxiliary for those outcomes and central to the RQ3 representation comparison.

### 3.9 Secondary Formal Extension: Joint-State Perturbation

The secondary formal extension evaluated deterministic constant single-joint bias; it did not introduce a fourth research question. For each of the four 0-mm main-scene trajectories and every ordered state \(q_t\in\mathbb{R}^7\), the evaluator applied

\[
q'_t=q_t+\Delta q,
\]

where the same bias vector was applied to all 41 states and exactly one arm-joint component was nonzero. The obstacle remained nominal and unchanged, and no replanning occurred. The 14 fixed directions were \(\pm J_1,\ldots,\pm J_7\), with complete magnitude scans at 0.5, 1, 2, 3, 4, 6, and 8 degrees. Every perturbed state was checked against arm-joint limits without clipping; invalidity was kept distinct from collision.

PhysX was the primary backend and cuRobo the auxiliary evaluator. The first PhysX safe-to-collision bracket was refined by deterministic bisection to width at most 0.1 degrees and reported as \([\theta_{\mathrm{safe}},\theta_{\mathrm{collision}}]\), with midpoint only as an estimate. Directions safe through 8 degrees were right-censored as `>8 deg` within the evaluated range. The minimum sampled axis-direction joint-bias tolerance was the minimum over these 14 axes, not a global seven-dimensional tolerance.

This controlled perturbation is not a real-world joint-error, encoder-noise, or tracking-error model. It perturbs one joint at a time with a trajectory-constant bias over a finite axis set and ceiling. It includes neither time-varying error, simultaneous multi-joint bias, combined environment-plus-joint perturbation, nor an empirical real-robot error distribution. Equal angular offsets across joints are not assumed to produce equal Cartesian effects.

### 3.10 Supplementary Constrained-Space Stress Test

A **post-hoc supplementary controlled stress test** examined safety-margin costs in a two-box gate family. Corridor width was the nominal surface-to-surface gap between inward-facing box surfaces. Four calibrated widths (120, 140, 160, and 300 mm) used the same five inflation levels, fixed start and goal, and 30 interleaved planning repetitions per condition after 20 excluded warm-ups. The 300 mm condition was the widest calibrated reference within the predefined range, not an unconstrained baseline.

The panda-hand path was classified at a fixed gate-plane crossing as passage or upper bypass. Gate-local cuRobo clearance was the minimum signed sphere-to-original-box clearance over states within a fixed 0.08 m slab centered on the gate plane. It supplemented rather than replaced global clearance. Outcomes comprised planning success, solve and wall time, trajectory length, route category, route consistency, and gate-local clearance. Repeated runs assessed timing and reproducibility rather than providing independent geometry samples.

Across all components, reproducibility was supported by frozen configurations, deterministic direction and execution orders, saved perturbation identities, fixed software versions, and explicit preservation of censored and reversal cases. No post-perturbation planning was allowed. Table 1 summarizes the main protocol and separately identifies the secondary joint-state extension; detailed configuration identities and hashes are retained with the reproducibility materials.

**Table 1.** Summary of the main frozen-trajectory robustness protocol and the separately scoped secondary joint-state extension.

| Component | Setting |
|---|---|
| Robot | Franka Panda |
| Main scenes | 4 |
| Safety-margin levels | 0/5/10/15/20 mm per obstacle surface |
| Frozen trajectories | 20 |
| Trajectory states | 41 ordered states; 7 arm joints |
| Formal directions | 128 deterministic Fibonacci-sphere directions |
| Environment perturbation | World-coordinate obstacle translation; 2–60 mm coarse range, 2 mm step |
| Boundary refinement | Deterministic bisection to interval width ≤0.5 mm; >60 mm right censoring |
| Primary robustness backend | Isaac Sim / PhysX; reported separation ≤0 m is collision |
| Planning benchmark | 20 warm-up calls excluded; 30 repetitions per condition; 600 measured calls |
| Replanning under robustness evaluation | Not allowed |
| Secondary extension | Constant single-joint bias on four frozen 0-mm trajectories; 14 axes, 0.5/1/2/3/4/6/8° |

## 4. Results

### 4.1 Nominal Clearance and Perturbation Robustness (RQ1)

The formal evaluation comprised four scenes, each represented by five ordered
planning-time obstacle-inflation conditions (0, 5, 10, 15, and 20 mm), for a
total of 20 frozen trajectories. Each trajectory was evaluated using the same
set of 128 fixed world-coordinate perturbation directions. Collision outcomes
were measured independently with the PhysX validation backend. First-collision
boundaries were represented by refined safe-to-collision intervals with widths
no greater than 0.5 mm; directions without a collision within the predefined
60 mm range were retained as right-censored. The midpoint values reported
below are interval estimates rather than exact continuous thresholds.

Across the 2,560 trajectory--direction pairs, 822 had an observed PhysX
boundary and 1,738 remained right-censored above 60 mm. Refinement required
1,644 additional trials, and every resulting PhysX interval had width no
greater than 0.5 mm.

Within every scene, cuRobo nominal clearance, PhysX nominal separation, the
minimum refined PhysX tolerance midpoint, and the fraction of right-censored
directions were non-decreasing across the five inflation conditions. At every
fixed perturbation budget from 10 to 60 mm, the PhysX collision fraction was
non-increasing within each scene. Thus, across all four evaluated scenes,
larger nominal cuRobo clearance was consistently associated with larger
independently measured PhysX perturbation tolerance (Figures 1 and 2).

For Scene A, cuRobo nominal clearance increased from 13.65 to 32.73
mm between the 0 and 20 mm inflation conditions, while the minimum PhysX
tolerance midpoint increased from 14.25 to 31.75 mm. For Scene B, the
corresponding quantities increased from 10.95 to 33.03 mm and from 10.75 to
33.25 mm. Scene C increased from 9.98 to 35.62 mm in cuRobo clearance and
from 9.25 to 33.25 mm in minimum PhysX tolerance. Scene D increased from
10.48 to 31.50 mm and from 7.75 to 27.75 mm, respectively. PhysX nominal
separation followed the same non-decreasing pattern in all four series.

The right-censored counts also increased between the endpoint conditions:
81 to 99 of 128 directions for Scene A, 78 to 97 for Scene B,
77 to 90 for Scene C, and 78 to 94 for Scene D. The complete
fixed-budget curves show that this change was not restricted to the single
minimum-tolerance direction: increasing the planning margin shifted the
empirical PhysX collision curves toward larger perturbation magnitudes across
each scene (Figure 2). These collision fractions are empirical fractions over
the fixed synthetic direction set, not estimates of real-world collision
probability.

### 4.2 Safety-Margin Benefit and Planning Cost (RQ2)

The robustness changes were produced by isotropic planning-time obstacle
inflation while evaluation retained the original obstacle geometry. Across the
formal planning benchmark, all 600 measured calls succeeded. Every returned
trajectory exactly reproduced its corresponding frozen canonical trajectory;
the maximum observed joint-space difference was 0 rad. Planning success
therefore remained 100% throughout the 20-condition matrix.

The trajectory-length cost from 0 to 20 mm inflation was scene dependent but
modest in the main evaluation scenes. Joint-space trajectory length increased
by approximately 0.32% for Scene A, 0.42% for Scene B, 2.21%
for Scene C, and 0.50% for Scene D. Scene C therefore showed
the largest trajectory-length increase. Its registered 20 mm condition,
including the bottleneck switch to `panda_leftfinger` near trajectory index 38,
was retained in all analyses.

Most condition-level median planner-reported solve times were between 25.86
and 26.81 ms. Two retained conditions were higher: Scene A at 15 mm
inflation had a median of 34.98 ms, and Scene C at 5 mm had a median of
35.08 ms. Their wall-time medians were 47.66 and 47.53 ms, respectively.
Neither condition was removed or rerun. Planning time did not exhibit a
monotonic increase with obstacle inflation, and the two higher values are
reported as unexplained condition-specific timing variation. Figure 3 jointly
summarizes the robustness benefit, trajectory-length change, and planning-time
observations.

### 4.3 Cross-Engine Collision-Boundary Agreement (RQ3)

The formal cross-engine comparison contained 76,800 coarse perturbation
trials. cuRobo and PhysX agreed on 75,722 collision decisions and disagreed on
1,078, corresponding to an overall agreement rate of 98.5964% (Figure 4). The
dominant disagreement type was cuRobo-safe/PhysX-collision (1,009 trials),
whereas 69 trials were cuRobo-collision/PhysX-safe.

Agreement was scene dependent. Scene A reached 98.9427% agreement,
Scene B reached 99.4688%, Scene C reached 98.6146%, and Scene
D reached 97.3594%. Scene D therefore produced the most disagreements
(507), compared with 203, 102, and 266 for the other three scenes.

Disagreement trials were generally close to the respective zero-distance
boundaries. Median absolute cuRobo clearance and PhysX separation were about
1.00 and 0.89 mm; their 75th percentiles were approximately 2.05 and 1.95 mm.
The observed maxima were approximately 4.43 and 4.79 mm. Thus, most
cross-engine disagreements occurred close to the respective collision
boundaries, although non-negligible differences of several millimetres were
also observed.

The complete sequences retained three PhysX collision-to-safe pairs and one
cuRobo collision-to-safe pair. Boundary summaries also retained three
cuRobo-only observed boundaries and 37 PhysX-only observed boundaries. The
dominant disagreement direction was cuRobo-safe/PhysX-collision, but the
frequency varied substantially across scenes; the data do not establish that
either collision backend is universally more conservative.

### 4.4 Post-hoc Constrained-Space Stress Test

This post-hoc supplementary test evaluated four two-box corridor widths (120,
140, 160, and 300 mm) at five inflation levels. All 600 calls succeeded and
each condition reproduced one route category. Inflation produced reproducible
passage-to-upper-bypass transitions; at 20 mm, trajectory length increased by
11.35% in the 120-mm corridor and 0.76% in the 300-mm reference (Figures S1–S4;
Table S1). In constrained geometry, margin cost can therefore appear as a
discrete route change and detour rather than a monotonic solve-time increase.
Detailed transition thresholds, gate-local geometry, and timing distributions
are provided in Supplementary Sections S1–S2.

### 4.5 Secondary Joint-State Perturbation

This experiment was a **secondary formal extension**, not a fourth research
question. It applied deterministic constant single-joint biases to the four
frozen 0-mm trajectories across all 41 ordered states, using 14 positive and
negative axes and magnitudes through 8 degrees without replanning. The minimum
sampled axis-direction tolerance ranged from 1.0 to 1.875 degrees across the
scenes; all four minima occurred on a J4 axis, but the sensitive sign differed.
These are sampled scene-dependent sensitivities, not hardware or encoder
accuracy claims.

Many directions remained safe through 8 degrees, and the extension therefore
does not establish a global joint-space tolerance or a general clearance-to-
joint-robustness relationship. Detailed coarse decisions, censoring, and
cross-engine refinement are reported in Supplementary Section S3.

## 5. Discussion

### 5.1 Nominal Clearance as a Robustness Indicator

The repeated within-scene trends support nominal clearance as an informative
empirical indicator of perturbation robustness in the evaluated conditions.
Nominal cuRobo clearance was not merely associated with the nominal binary
collision label: every increase across the controlled safety-margin series was
accompanied by non-decreasing PhysX nominal separation, non-decreasing minimum
directional tolerance, increasing censoring, and non-increasing fixed-budget
collision fractions. This pattern was reproduced across four different scene
geometries.

The independent PhysX outcome is important for interpreting this relationship.
If both nominal clearance and perturbation tolerance were defined exclusively
using the same cuRobo collision-sphere geometry, part of their relationship
could follow almost definitionally from the shared representation. Using
PhysX as an independent collision-validation backend reduces this same-model
tautology. It does not make PhysX an absolute reference, but it demonstrates
that the observed clearance--robustness pattern persists under a distinct
collision representation.

The result should nevertheless be interpreted within its measured scope. It
does not establish a universal predictor, a guaranteed safety margin, or a
real-world error tolerance. The primary evidence consists of five ordered
conditions within each of only four scenes; the 20 trajectories are not 20
independent environments. The finite direction set and 60 mm censoring cap
also mean that the measured minimum tolerance is a minimum over sampled
directions rather than a continuous global worst case.

### 5.2 What Is the Cost of a Larger Safety Margin?

In the main single-obstacle scenes, planning-time inflation produced a clear
robustness benefit at relatively small trajectory cost. All plans succeeded,
all repeated trajectories matched their frozen references, and three of the
four scenes increased trajectory length by no more than 0.50% at 20 mm
inflation. Scene C showed a larger but still moderate 2.21% increase.
Planning-time medians were stable for most conditions and did not increase
monotonically. The two elevated medians cannot be attributed to inflation from
the current data and remain unexplained condition-specific timing variation.

The post-hoc supplementary constrained-space results indicate that the cost of
a larger safety margin can depend on environmental geometry. In those scenes,
the most visible cost was neither a general solve-time increase nor planning
failure. Instead, the planner reproducibly transitioned from passage traversal
to an upper bypass. This transition was accompanied by a detour whose magnitude
was strongly conditioned on corridor width: the 120 mm corridor accumulated an
11.35% length increase, whereas the widest calibrated 300 mm reference
increased by 0.76%.

These complementary results suggest that a planning margin has multiple forms
of cost. In open geometries it may alter a trajectory only slightly. In a
constrained geometry it may induce a discrete route change while retaining
planning feasibility and nearly unchanged solve time. Because the corridor
study was designed after the main formal analysis, it is supplementary,
post-hoc evidence rather than part of the original formal RQ2 matrix.

### 5.3 Cross-Engine Discrepancy

Although cuRobo and PhysX agreed on 98.5964% of coarse decisions, the 1,078
disagreements were strongly asymmetric: 1,009 were cuRobo-safe/PhysX-collision
and 69 were the reverse. Most disagreement cases remained close to the
respective collision boundaries, but sampled offsets extended to several
millimetres. Scene D exhibited a substantially higher disagreement count than
Scene B, so collision-boundary conclusions are sensitive both to proximity to
contact and to scene geometry.

Differences between collision-sphere approximations and PhysX collision
geometry offer a plausible representation-level context, but the current data
do not isolate individual geometric causes for each disagreement. PhysX is an
independent backend, not absolute ground truth, and the results should not be
described as showing that cuRobo is categorically inaccurate. Similarly, the
asymmetry toward cuRobo-safe/PhysX-collision decisions in this dataset does not
establish universal PhysX conservatism. The retained collision-to-safe and
one-engine-only boundary cases further show that sampled translational paths
can produce more complex behavior than a single monotonic threshold.

Practically, the cross-engine comparison indicates that nominal margins within
a few millimetres of contact can depend on the chosen collision
representation. Reporting disagreement cases and boundary intervals is
therefore more informative than forcing both engines to share a single label.

### 5.4 Perturbation-Model Dependence of Trajectory Robustness

The secondary extension illustrates that robustness depends on the sampled
perturbation family. Environment translation changes the
obstacle--trajectory relationship in fixed world directions, whereas a
constant joint bias moves robot geometry through kinematically different
directions along the frozen sequence. A trajectory that is robust to one
sampled family therefore need not be equivalently robust to another. Within
the four evaluated trajectories, joint-axis sensitivity was direction
dependent and was not fully summarized by nominal clearance alone. This
suggests perturbation-model dependence; it does not establish a universal
theory or a model of real robot errors.

Cross-engine agreement also remained high in the joint-state extension, while
sub-degree sampled boundary offsets persisted. This secondary observation is
consistent with the main RQ3 finding that collision labels can agree broadly
while boundary location remains representation dependent. The datasets were
not pooled.

### 5.5 Limitations

The study is bounded to one manipulator, the Franka Panda, and four main formal
scenes. The main environments use static box obstacles, and the perturbation
model translates obstacle position without rotation, deformation, or dynamics.
The results are simulation based and include no physical-robot validation.
PhysX provides an independent representation but is not absolute ground truth.

The Formal Direction Set contains 128 fixed directions. It provides
deterministic sphere coverage but not a continuous search over all translation
directions. Threshold refinement was capped by a 60 mm coarse range, producing
right-censored directions whose exact boundaries remain unknown. Discrete
41-state trajectory replay also does not establish continuous swept-volume
safety between stored states.

The constrained-space experiment is a post-hoc supplementary stress test. It
contains four selected widths and was not designed specifically around a
feasibility boundary; consequently, no planning failure was observed. That
absence cannot be generalized to narrower passages or larger inflation.
Planning-time measurements also depend on the current GPU, software versions,
planner configuration, and deterministic execution procedure.

The secondary joint-state extension is limited to deterministic constant
single-joint biases along 14 sampled axes and an 8 degree ceiling. It does not
evaluate simultaneous multi-joint bias, time-varying tracking error, an
empirical real-robot error distribution, or combined environment and joint
perturbation. Its PhysX evaluation is discrete at the same 41 ordered states.
It therefore characterizes synthetic axis-direction sensitivity rather than
real-world joint-error robustness. The formal studies also exclude rotational
environment uncertainty, dynamic obstacles, and obstacle-shape uncertainty.

### 5.6 Implications

A nominally collision-free trajectory should not be characterized solely by a
binary collision label. Within the evaluated settings, nominal clearance,
directional perturbation tolerance, fixed-budget robustness curves, censoring,
and cross-engine agreement each describe a different aspect of robustness.
Reporting them together provides a more informative empirical characterization
than any single pass/fail outcome.

The results also clarify the role of geometric safety margins. Increasing
planning-time margins can improve independently measured perturbation
robustness, but its cost may appear as trajectory rerouting rather than simply
longer solve time. The appropriate margin is therefore environment dependent;
the present data do not define a universally optimal inflation or certify safe
deployment.

### 5.7 Future Work

Future evaluations could expand the number and diversity of scenes and robot
models, add rotational and shape perturbations, incorporate dynamic obstacles,
and validate selected boundaries on a physical robot. Denser or adaptive
directional sampling could reduce censoring and better approximate continuous
minimum tolerance. Targeted feasibility-boundary scenes could separately
study planning failure, while richer collision models could help localize the
source of cross-engine disagreement. Joint-related extensions should examine
simultaneous multi-joint perturbations, time-varying state or tracking-error
models grounded in measured robot data, and real-robot validation. A separate
protocol could study combined environment and joint perturbations; no such
combined experiment has been conducted.

## 6. Conclusion

This study separated nominal planning from post-planning robustness: cuRobo generated each trajectory once, its 41 ordered arm states were frozen, and controlled environment translations were evaluated without replanning using an independent PhysX validation path. The evidence therefore characterizes frozen-trajectory robustness across complementary metrics rather than proposing a new planner or safety rule.

For RQ1, nominal cuRobo clearance tracked sampled translational tolerance consistently within the four evaluated scene series. Directional boundaries, censoring, and fixed-budget empirical collision fractions reveal structure that a nominal pass/fail label cannot; the result remains limited to the tested robot, scenes, directions, and 60-mm range.

For RQ2, planning-time inflation improved measured robustness in the main scenes while all 600 benchmark calls succeeded, trajectory-length costs remained small, and solve time showed no reliable monotonic increase. The post-hoc constrained-space test indicates that margin cost can instead appear as a discrete route change and detour, but remains supplementary rather than confirmatory evidence.

For RQ3, cuRobo and PhysX agreed on 75,722 of 76,800 decisions (98.5964%), yet disagreement was asymmetric (1009 versus 69) and sampled boundary offsets reached several millimetres. High aggregate agreement therefore does not imply identical collision boundaries, and PhysX is an independent validation backend rather than ground truth.

The secondary constant single-joint extension suggests dependence on perturbation family and direction, but is neither a fourth research question nor a real-robot error model. Key limitations are the four-scene simulation set, finite directional sampling and censoring, and ordered-state validation without continuous swept checking, dynamic tracking, or physical-robot evidence. Future work can address richer layouts, measured state errors, execution-aware validation, and separately designed combined perturbations. The central implication is that nominal collision-free validity should be complemented by explicit sampled perturbation tolerance and collision-model-aware validation.
## Figure Captions

**Figure 1.** Minimum sampled PhysX tolerance midpoint versus nominal cuRobo clearance for the 20 scene-by-inflation conditions (Scenes A–D). Each plotted value is the midpoint of the condition-level minimum refined safe-to-collision interval over the 128 sampled directions; it is not a continuous or global worst-case tolerance. Direction-level observations without a collision through 60 mm remain right-censored at `>60 mm`, while all plotted condition-level minima were observed within the evaluated range.

**Figure 2.** PhysX empirical collision fraction across the 128 deterministic world-frame directions for frozen trajectories at discrete perturbation magnitudes of 10–60 mm. Points represent sampled magnitudes and evaluations were performed without replanning.

**Figure 3.** Safety-margin benefit and cost. (a) Relative joint-space trajectory-length change from each scene's 0-mm condition; (b) planner solve-time medians with IQR; and (c) minimum sampled PhysX tolerance midpoint. All 600 measured planning calls succeeded. All timing observations, including two elevated condition-level results, were retained; no causal interpretation is assigned to those spikes.

**Figure 4.** Cross-engine disagreement. (a) Asymmetric disagreement counts (1009 cuRobo-safe/PhysX-collision; 69 cuRobo-collision/PhysX-safe). (b) ECDF of the maximum of recorded absolute cuRobo clearance and absolute PhysX separation for each disagreement, used as the distance to the opposite-engine sampled boundary. Dashed 1, 2, and 3 mm lines are reading guides only. PhysX is an independent validation backend, not ground truth.

The corridor passage-versus-bypass end-effector overlay is supplementary only: an x–z projection of ordered end-effector positions derived by forward kinematics from frozen 41-state trajectories, with connecting lines used for visualization rather than continuous swept-path claims.

**Figure S5.** Supplementary corridor visualization: x–z projection of ordered end-effector positions from the frozen 120-mm corridor trajectories planned with 0-mm (passage) and 20-mm (upper bypass) inflation. The traces are derived by forward kinematics from 41 ordered states; connecting lines are visualization aids, not continuous swept paths.

## References

The reference list is generated from the verified bibliography supplied with the manuscript.

# Robot Motion Robustness

**From Nominal Clearance to Collision Robustness: An Empirical Evaluation of cuRobo-Generated Manipulator Trajectories under Environment Perturbations**

This is an ongoing independent empirical robotics study using public NVIDIA cuRobo,
Isaac Sim/PhysX, and the public Franka Panda asset. cuRobo is used as the nominal
collision-aware trajectory generator. Isaac Sim/PhysX is used as an independent
sampled-state collision-validation backend; it is not treated as ground truth.

The core protocol generates a nominal trajectory once, freezes its ordered joint states,
applies controlled post-planning perturbations, and performs **no replanning after
perturbation**. The project does not introduce a new planner, modify the cuRobo
optimizer, claim a safety guarantee, or estimate real-world collision probability.
The 41 ordered states are not continuous swept-volume or dynamic-controller validation.

The frozen evidence hierarchy is:

1. **Main formal evidence:** four environment scene series, five planning-time
   inflation conditions, deterministic translational perturbations, and cuRobo/PhysX
   comparison for RQ1–RQ3.
2. **Secondary formal extension:** constant single-joint perturbations.
3. **Post-hoc supplementary evidence:** constrained corridor route and margin study.
4. **Excluded exploratory work:** randomized combined pilots, failed calibration,
   diagnostic scene redesign, and pilot-only experiments.

## Reproducibility

See [REPRODUCTION.md](REPRODUCTION.md) for the frozen configurations, execution order,
result-verification mode, full-experiment mode, data/archive split, and known limitations.

The reported environment used Isaac Sim 6.0.1.0, Python 3.12.3, PyTorch 2.13.0+cu130,
CUDA 13.0, and cuRobo commit `8e734f3ced1df898990bcd92de40abce475907db`. The RTX 4080
is reported as timing context, not as a requirement that excludes other compatible GPUs.

## License

Original software source code developed for this project is licensed under the MIT
License. The manuscript, supplementary text, research figures, and experimental
datasets are not covered by the MIT License unless explicitly stated otherwise.
External dependencies and third-party assets remain subject to their respective
licenses.

## Third-Party Dependencies

- cuRobo is an external dependency and remains subject to its own license.
- Isaac Sim / PhysX is an external platform subject to its own licensing terms.
- Franka robot assets are loaded from the external simulation environment and are
  not redistributed by this repository.
- External dependencies and assets are not relicensed by this project's MIT License.
